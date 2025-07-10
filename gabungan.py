import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import json
import threading
import time
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from sqlalchemy import text

# Configuration
DB_URI = "postgresql://postgres:1234@localhost:5432/postgres"
VECTOR_DB_CONNECTION = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
COLLECTION_NAME = "my_docs"

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Page configuration
st.set_page_config(
    page_title="IT Support Ticketing System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .ticket-number {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 20px 0;
        font-size: 1.2em;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border: 1px solid #c3e6cb;
    }
    .status-pending {
        color: #ffc107;
        font-weight: bold;
    }
    .status-done {
        color: #28a745;
        font-weight: bold;
    }
    .status-processing {
        color: #007bff;
        font-weight: bold;
    }
    .priority-high {
        color: #dc3545;
        font-weight: bold;
    }
    .priority-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .priority-low {
        color: #28a745;
        font-weight: bold;
    }
    .ticket-card {
        background-color: #6a756d;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# Database connection
def get_db_connection():
    try:
        connection = psycopg2.connect(
            host="localhost",
            database="postgres",
            user="postgres",
            password="1234",
            port="5432"
        )
        return connection
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

# Initialize RAG components
@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system components"""
    try:
        chat_model = ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            model='gemini-2.5-pro',
            temperature=0.9
        )
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=GOOGLE_API_KEY
        )
        
        db = SQLDatabase.from_uri(DB_URI, schema="data")
        
        # Load documents from database
        def load_documents_from_db():
            with db._engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT
                        "ID Tiket",
                        "Keluhan",
                        "Prioritas",
                        "Justifikasi Prioritas",
                        "Unit Penanggung Jawab",
                        "Solusi Awal (untuk Pelanggan)"
                    FROM data.dataset_dummy_ticketing;
                """))
                
                rows = result.fetchall()
                docs = []
                for row in rows:
                    (
                        ticket_id,
                        keluhan,
                        prioritas,
                        justifikasi_prioritas,
                        unit_penanggung_jawab,
                        solusi_awal
                    ) = row
                    
                    doc = Document(
                        page_content=keluhan,
                        metadata={
                            "ticket_id": ticket_id,
                            "prioritas": prioritas,
                            "justifikasi_prioritas": justifikasi_prioritas,
                            "unit_penanggung_jawab": unit_penanggung_jawab,
                            "solusi_awal": solusi_awal
                        }
                    )
                    docs.append(doc)
                
                return docs
        
        # Initialize vector store
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=VECTOR_DB_CONNECTION,
            use_jsonb=True,
        )
        
        # Load and add documents
        docs = load_documents_from_db()
        vector_store.add_documents(
            docs,
            ids=[str(doc.metadata["ticket_id"]) for doc in docs]
        )
        
        # Create RAG chain
        retriever = vector_store.as_retriever()
        
        system_prompt = (
            "Anda adalah asisten AI yang sangat efisien untuk tim dukungan IT. "
            "Tugas utama Anda adalah menganalisis tiket masalah baru yang diberikan oleh pengguna, "
            "dan berdasarkan deskripsi tiket tersebut serta konteks historis yang relevan "
            "yang diambil dari basis pengetahuan, tentukan informasi berikut:\n\n"
            "1. **issue**: Salin ulang pertanyaan/keluhan dari pengguna.\n"
            "2. **priority**: Pilih salah satu dari P1 (Kritis), P2 (Tinggi), P3 (Sedang), atau P4 (Rendah).\n"
            "3. **unit**: Rekomendasikan tim atau departemen yang paling sesuai untuk menangani tiket ini.\n"
            "4. **solution**: Solusi awal yang dapat membantu Pengguna.\n"
            "5. **justification**: Justifikasi kenapa keluhan tersebut memiliki prioritas tertentu.\n"
            "\n"
            "Jika informasi tidak ditemukan, nyatakan 'Tidak Ada' atau 'Tidak Diketahui' pada poin yang bersangkutan."
            "\n\n"
            "Konteks dari tiket serupa:\n{context}\n\n"
            "Tolong format jawaban dalam bentuk JSON dengan key: issue, priority, unit, solution, justification"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain, db
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None, None

# Database functions
def save_ticket(question):
    """Save ticket to database"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return None
            
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO data.rag_queries (question, status)
                VALUES (%s, %s)
                RETURNING id
            """, (question, 'pending'))
            ticket_id = cur.fetchone()[0]
            conn.commit()
            return ticket_id
    except Exception as e:
        st.error(f"Error saving ticket: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close()

def get_all_tickets():
    """Get all tickets from database"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return []
            
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT rq.id, rq.question, rq.status,
                       rl.solution, rl.sources
                FROM data.rag_queries rq
                LEFT JOIN data.rag_logs rl ON rq.question = rl.issue
                ORDER BY rq.id DESC
            """)
            return cur.fetchall()
    except Exception as e:
        st.error(f"Error fetching tickets: {e}")
        return []
    finally:
        if conn:
            conn.close()

def process_single_ticket(ticket_id, question, rag_chain, db):
    """Process a single ticket using RAG"""
    try:
        # Run RAG
        result = rag_chain.invoke({"input": question})
        answer = result['answer']
        sources = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in result['context']
        ]
        
        # Save to database
        with db._engine.begin() as conn:
            # Insert into rag_logs
            conn.execute(text("""
                INSERT INTO data.rag_logs (issue, solution, sources)
                VALUES (:q, :a, :s)
            """), {
                "q": question,
                "a": answer,
                "s": json.dumps(sources)
            })
            
            # Update query status
            conn.execute(text("""
                UPDATE data.rag_queries
                SET status = 'done'
                WHERE id = :id
            """), {"id": ticket_id})
        
        return True, answer
        
    except Exception as e:
        return False, str(e)

def update_ticket_status(ticket_id, new_status):
    """Update ticket status"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False
            
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE data.rag_queries
                SET status = %s
                WHERE id = %s
            """, (new_status, ticket_id))
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Error updating ticket status: {e}")
        return False
    finally:
        if conn:
            conn.close()

# Sidebar for navigation
def render_sidebar():
    """Render sidebar navigation"""
    st.sidebar.title("🎫 IT Support System")
    
    # User role selection
    user_role = st.sidebar.selectbox(
        "Select Role",
        ["User", "Admin"],
        help="Choose your role to access appropriate features"
    )
    
    if user_role == "User":
        page = st.sidebar.selectbox(
            "Navigation",
            ["Submit Ticket", "Track Tickets"]
        )
    else:
        page = st.sidebar.selectbox(
            "Navigation",
            ["Dashboard", "Manage Tickets", "Process Tickets"]
        )
    
    return user_role, page

# User interface functions
def render_submit_ticket():
    """Render ticket submission form"""
    st.markdown('<h1 class="main-header">🎫 Submit Support Ticket</h1>', unsafe_allow_html=True)
    
    with st.form("ticket_form", clear_on_submit=True):
        title = st.text_input(
            "Title *", 
            placeholder="e.g., Cannot login to account",
            help="Provide a short title describing your issue"
        )
        
        description = st.text_area(
            "Description *", 
            placeholder="Explain your issue in detail...",
            help="Provide complete description of the problem you're experiencing",
            height=120
        )
        
        submitted = st.form_submit_button("🚀 Submit Ticket", use_container_width=True)
        
        if submitted:
            if title and description:
                full_question = f"{title}: {description}"
                ticket_id = save_ticket(full_question)
                
                if ticket_id:
                    st.markdown(f'<div class="success-message">✅ Ticket created successfully!</div>', 
                               unsafe_allow_html=True)
                    st.markdown(f'<div class="ticket-number">🎫 Ticket ID: {ticket_id}</div>', 
                               unsafe_allow_html=True)
                    st.markdown('<span class="status-pending">⏳ Status: Pending</span>', 
                               unsafe_allow_html=True)
                    st.info("💡 Save your ticket ID for future reference.")
                else:
                    st.error("❌ Failed to create ticket. Please try again.")
            else:
                st.error("⚠️ Please fill in all required fields (*)")

def render_track_tickets():
    """Render ticket tracking interface"""
    st.markdown('<h1 class="main-header">📊 Track Your Tickets</h1>', unsafe_allow_html=True)
    
    tickets = get_all_tickets()
    
    if not tickets:
        st.info("No tickets found.")
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox("Filter by Status", ["All", "pending", "done", "processing"])
    with col2:
        search_term = st.text_input("Search tickets", placeholder="Search by ticket content...")
    
    filtered_tickets = tickets
    if status_filter != "All":
        filtered_tickets = [t for t in filtered_tickets if t['status'] == status_filter]
    
    if search_term:
        filtered_tickets = [t for t in filtered_tickets if search_term.lower() in t['question'].lower()]
    
    # Display tickets
    for ticket in filtered_tickets:
        with st.expander(f"Ticket #{ticket['id']} - {ticket['status'].upper()}"):
            st.write(f"**Ticket ID:** {ticket['id']}")
            st.write(f"**Question:** {ticket['question']}")
            
            # Status styling
            status_class = {
                'pending': 'status-pending',
                'done': 'status-done',
                'processing': 'status-processing'
            }.get(ticket['status'], '')
            
            st.markdown(f'<span class="{status_class}">Status: {ticket["status"].upper()}</span>', 
                       unsafe_allow_html=True)
            
            if ticket['solution']:
                st.write("**Solution:**")
                try:
                    solution_data = json.loads(ticket['solution'])
                    st.json(solution_data)
                except:
                    st.write(ticket['solution'])

# Admin interface functions
def render_admin_dashboard():
    """Render admin dashboard"""
    st.markdown('<h1 class="main-header">📈 Admin Dashboard</h1>', unsafe_allow_html=True)
    
    tickets = get_all_tickets()
    
    if not tickets:
        st.info("No tickets in system.")
        return
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    total_tickets = len(tickets)
    pending_tickets = len([t for t in tickets if t['status'] == 'pending'])
    done_tickets = len([t for t in tickets if t['status'] == 'done'])
    processing_tickets = len([t for t in tickets if t['status'] == 'processing'])
    
    with col1:
        st.metric("Total Tickets", total_tickets)
    with col2:
        st.metric("Pending", pending_tickets)
    with col3:
        st.metric("Processing", processing_tickets)
    with col4:
        st.metric("Completed", done_tickets)
    
    # Recent tickets
    st.markdown("### Recent Tickets")
    recent_tickets = tickets[:10]
    
    for ticket in recent_tickets:
        with st.container():
            st.markdown(f"""
            <div class="ticket-card">
                <strong>Ticket #{ticket['id']}</strong> - 
                <span class="status-{ticket['status']}">
                    {ticket['status'].upper()}
                </span><br>
                <small>Ticket ID: {ticket['id']}</small><br>
                {ticket['question'][:100]}...
            </div>
            """, unsafe_allow_html=True)

def render_manage_tickets():
    """Render ticket management interface"""
    st.markdown('<h1 class="main-header">🔧 Manage Tickets</h1>', unsafe_allow_html=True)
    
    tickets = get_all_tickets()
    
    if not tickets:
        st.info("No tickets to manage.")
        return
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox("Filter by Status", ["All", "pending", "done", "processing"])
    with col2:
        search_term = st.text_input("Search", placeholder="Search tickets...")
    
    filtered_tickets = tickets
    if status_filter != "All":
        filtered_tickets = [t for t in filtered_tickets if t['status'] == status_filter]
    
    if search_term:
        filtered_tickets = [t for t in filtered_tickets if search_term.lower() in t['question'].lower()]
    
    # Ticket management
    for ticket in filtered_tickets:
        with st.expander(f"Ticket #{ticket['id']} - {ticket['status'].upper()}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Ticket ID:** {ticket['id']}")
                st.write(f"**Question:** {ticket['question']}")
                
                if ticket['solution']:
                    st.write("**Solution:**")
                    try:
                        solution_data = json.loads(ticket['solution'])
                        st.json(solution_data)
                    except:
                        st.write(ticket['solution'])
            
            with col2:
                new_status = st.selectbox(
                    "Status",
                    ["pending", "processing", "done"],
                    index=["pending", "processing", "done"].index(ticket['status']),
                    key=f"status_{ticket['id']}"
                )
                
                if st.button(f"Update", key=f"update_{ticket['id']}"):
                    if update_ticket_status(ticket['id'], new_status):
                        st.success("Status updated!")
                        st.rerun()
                    else:
                        st.error("Failed to update status.")

def render_process_tickets():
    """Render ticket processing interface"""
    st.markdown('<h1 class="main-header">⚙️ Process Tickets</h1>', unsafe_allow_html=True)
    
    # Initialize RAG system
    rag_chain, db = initialize_rag_system()
    
    if not rag_chain:
        st.error("Failed to initialize RAG system. Please check your configuration.")
        return
    
    # Get pending tickets
    tickets = get_all_tickets()
    pending_tickets = [t for t in tickets if t['status'] == 'pending']
    
    if not pending_tickets:
        st.info("No pending tickets to process.")
        return
    
    st.write(f"Found {len(pending_tickets)} pending tickets.")
    
    # Process options
    col1, col2 = st.columns(2)
    with col1:
        process_limit = st.number_input("Number of tickets to process", min_value=1, max_value=len(pending_tickets), value=min(5, len(pending_tickets)))
    with col2:
        if st.button("🔄 Process Tickets", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, ticket in enumerate(pending_tickets[:process_limit]):
                status_text.text(f"Processing ticket #{ticket['id']}...")
                
                # Update status to processing
                update_ticket_status(ticket['id'], 'processing')
                
                # Process ticket
                success, result = process_single_ticket(ticket['id'], ticket['question'], rag_chain, db)
                
                if success:
                    st.success(f"✅ Processed ticket #{ticket['id']}")
                else:
                    st.error(f"❌ Failed to process ticket #{ticket['id']}: {result}")
                    update_ticket_status(ticket['id'], 'pending')
                
                progress_bar.progress((i + 1) / process_limit)
            
            status_text.text("Processing completed!")
            st.rerun()
    
    # Show pending tickets
    st.markdown("### Pending Tickets")
    for ticket in pending_tickets[:10]:
        with st.expander(f"Ticket #{ticket['id']}"):
            st.write(f"**Ticket ID:** {ticket['id']}")
            st.write(f"**Question:** {ticket['question']}")
            
            if st.button(f"Process this ticket", key=f"process_single_{ticket['id']}"):
                with st.spinner("Processing..."):
                    update_ticket_status(ticket['id'], 'processing')
                    success, result = process_single_ticket(ticket['id'], ticket['question'], rag_chain, db)
                    
                    if success:
                        st.success("✅ Ticket processed successfully!")
                    else:
                        st.error(f"❌ Processing failed: {result}")
                        update_ticket_status(ticket['id'], 'pending')
                    
                    st.rerun()

# Main application
def main():
    """Main application function"""
    user_role, page = render_sidebar()
    
    if user_role == "User":
        if page == "Submit Ticket":
            render_submit_ticket()
        elif page == "Track Tickets":
            render_track_tickets()
    
    else:  # Admin
        if page == "Dashboard":
            render_admin_dashboard()
        elif page == "Manage Tickets":
            render_manage_tickets()
        elif page == "Process Tickets":
            render_process_tickets()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>Need help? Email: support@company.com | © 2024 IT Support System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()