import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ticketing_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="IT Support Ticketing System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS STYLING
# =============================================================================

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
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #007bff;
    }
    .pending-ticket-card {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #ffc107;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def get_db_connection():
    """Create database connection with proper error handling"""
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        return connection
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        st.error(f"Database connection failed: {e}")
        return None

def save_ticket(question):
    """Save ticket to database with proper error handling"""
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
            logger.info(f"Ticket {ticket_id} saved successfully")
            return ticket_id
            
    except psycopg2.Error as e:
        logger.error(f"Error saving ticket: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close()

def get_all_tickets():
    """Get all tickets from database with proper error handling"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return []
            
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT rq.id, rq.question, rq.status,
                       rl.solution, rl.sources, rl.created_at
                FROM data.rag_queries rq
                LEFT JOIN data.rag_logs rl ON rq.id = rl.query_id
                ORDER BY rq.id DESC
            """)
            tickets = cur.fetchall()
            logger.info(f"Retrieved {len(tickets)} tickets")
            return tickets
            
    except psycopg2.Error as e:
        logger.error(f"Error fetching tickets: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_rag_logs():
    """Get all RAG logs from database with proper error handling"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return []
            
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT rl.query_id, rl.issue, rl.solution, rl.sources, rl.created_at
                FROM data.rag_logs rl
                ORDER BY rl.created_at DESC
            """)
            logs = cur.fetchall()
            logger.info(f"Retrieved {len(logs)} RAG logs")
            return logs
            
    except psycopg2.Error as e:
        logger.error(f"Error fetching RAG logs: {e}")
        return []
    finally:
        if conn:
            conn.close()

def update_ticket_status(ticket_id, new_status):
    """Update ticket status with proper error handling"""
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
            logger.info(f"Ticket {ticket_id} status updated to {new_status}")
            return True
            
    except psycopg2.Error as e:
        logger.error(f"Error updating ticket status: {e}")
        return False
    finally:
        if conn:
            conn.close()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_solution_json(solution_data):
    """Parse solution JSON data with robust error handling"""
    try:
        if isinstance(solution_data, dict):
            return solution_data
        
        if isinstance(solution_data, str):
            return json.loads(solution_data)
        
        return {
            'issue': 'Unknown',
            'priority': 'P3 (Sedang)',
            'unit': 'Unknown',
            'solution': 'No solution available',
            'justification': 'No justification available'
        }
        
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.error(f"Error parsing solution JSON: {e}")
        return {
            'issue': 'Parse Error',
            'priority': 'P3 (Sedang)',
            'unit': 'Unknown',
            'solution': 'Error parsing solution',
            'justification': 'JSON parsing failed'
        }

# =============================================================================
# CHART FUNCTIONS
# =============================================================================

def create_priority_chart(df):
    """Create priority distribution chart"""
    if df.empty:
        return None
    
    priority_counts = df['priority'].value_counts()
    
    fig = px.pie(
        values=priority_counts.values,
        names=priority_counts.index,
        title="Distribusi Tingkat Prioritas Tiket",
        color_discrete_map={
            'P1 (Kritis)': '#dc3545',
            'P2 (Tinggi)': '#fd7e14',
            'P3 (Sedang)': '#ffc107',
            'P4 (Rendah)': '#28a745'
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        showlegend=True,
        height=400,
        font=dict(size=12)
    )
    
    return fig

def create_unit_chart(df):
    """Create unit distribution chart"""
    if df.empty:
        return None
    
    unit_counts = df['unit'].value_counts().head(10)
    
    fig = px.bar(
        x=unit_counts.values,
        y=unit_counts.index,
        orientation='h',
        title="Top 10 Unit Penanganan Tiket",
        labels={'x': 'Jumlah Tiket', 'y': 'Unit'}
    )
    
    fig.update_layout(
        height=400,
        font=dict(size=12),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_status_chart(tickets):
    """Create ticket status distribution chart"""
    if not tickets:
        return None
    
    status_counts = {}
    for ticket in tickets:
        status = ticket['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    fig = px.bar(
        x=list(status_counts.keys()),
        y=list(status_counts.values()),
        title="Distribusi Status Tiket",
        labels={'x': 'Status', 'y': 'Jumlah Tiket'},
        color_discrete_map={
            'pending': '#ffc107',
            'processing': '#007bff',
            'done': '#28a745'
        }
    )
    
    fig.update_layout(
        height=400,
        font=dict(size=12)
    )
    
    return fig

def filter_rag_logs(df, priority_filter, unit_filter, search_term):
    """Filter RAG logs based on criteria"""
    filtered_df = df.copy()
    
    if priority_filter != "All":
        filtered_df = filtered_df[filtered_df['priority'] == priority_filter]
    
    if unit_filter != "All":
        filtered_df = filtered_df[filtered_df['unit'] == unit_filter]
    
    if search_term:
        search_mask = (
            filtered_df['issue'].str.contains(search_term, case=False, na=False) |
            filtered_df['justification'].str.contains(search_term, case=False, na=False)
        )
        filtered_df = filtered_df[search_mask]
    
    return filtered_df

# =============================================================================
# UI FUNCTIONS
# =============================================================================

def render_sidebar():
    """Render sidebar navigation"""
    st.sidebar.title("🎫 IT Support System")
    
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
        page = "Dashboard"  # Only one page for admin
    
    return user_role, page

def render_submit_ticket():
    """Render ticket submission form"""
    st.markdown('<h1 class="main-header">🎫 Submit Support Ticket</h1>', unsafe_allow_html=True)
    
    with st.form("ticket_form", clear_on_submit=True):
        st.subheader("Describe Your Issue")
        
        # Input fields
        question = st.text_area(
            "Please describe your problem or question in detail:",
            height=150,
            placeholder="Example: I can't access my email account. When I try to log in, I get an error message 'Invalid credentials'..."
        )

        # Submit button
        submitted = st.form_submit_button("Submit Ticket", type="primary")
        
        if submitted:
            if not question.strip():
                st.error("Please describe your issue before submitting.")
                return
            
            # Save ticket to database
            with st.spinner("Submitting your ticket..."):
                ticket_id = save_ticket(question)
                
                if ticket_id:
                    st.success("Ticket submitted successfully!")
                    st.markdown(f"""
                    <div class="ticket-number">
                        🎫 Your Ticket ID: <strong>#{ticket_id}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("Your ticket has been submitted and will be processed shortly. You can track its status using the ticket ID above.")
                else:
                    st.error("Failed to submit ticket. Please try again.")

def render_track_tickets():
    """Render ticket tracking interface"""
    st.markdown('<h1 class="main-header">🔍 Track Your Tickets</h1>', unsafe_allow_html=True)
    
    # Search by ticket ID
    col1, col2 = st.columns([3, 1])
    with col1:
        ticket_id = st.text_input("Enter Ticket ID to search:", placeholder="e.g., 123")
    with col2:
        search_button = st.button("Search", type="primary")
    
    # Get all tickets
    tickets = get_all_tickets()
    
    if search_button and ticket_id:
        # Filter by specific ticket ID
        filtered_tickets = [t for t in tickets if str(t['id']) == ticket_id]
        if not filtered_tickets:
            st.warning(f"No ticket found with ID: {ticket_id}")
            return
        tickets = filtered_tickets
    
    if not tickets:
        st.info("No tickets found.")
        return
    
    # Display tickets
    st.subheader("Your Tickets")
    
    for ticket in tickets:
        with st.container():
            st.markdown(f"""
            <div class="ticket-card">
                <h4>🎫 Ticket #{ticket['id']}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                status_class = f"status-{ticket['status']}"
                st.markdown(f"**Status:** <span class='{status_class}'>{ticket['status'].upper()}</span>", 
                          unsafe_allow_html=True)
            with col2:
                if ticket['created_at']:
                    st.write(f"**Created:** {ticket['created_at'].strftime('%Y-%m-%d %H:%M')}")
            with col3:
                if ticket['status'] == 'done' and ticket['solution']:
                    st.write("**Status:** ✅ Resolved")
                elif ticket['status'] == 'processing':
                    st.write("**Status:** ⏳ Processing")
                else:
                    st.write("**Status:** ⏰ Pending")
            
            # Question
            st.write("**Issue Description:**")
            st.text_area("", value=ticket['question'], height=100, disabled=True, key=f"question_{ticket['id']}")
            
            # Solution (if available)
            if ticket['solution']:
                st.write("**Solution:**")
                try:
                    solution_data = parse_solution_json(ticket['solution'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        priority_class = f"priority-{solution_data.get('priority', 'medium').lower()}"
                        st.markdown(f"**Priority:** <span class='{priority_class}'>{solution_data.get('priority', 'N/A')}</span>", 
                                  unsafe_allow_html=True)
                    with col2:
                        st.write(f"**Assigned to:** {solution_data.get('unit', 'N/A')}")
                    
                    st.write("**Solution Details:**")
                    st.info(solution_data.get('solution', 'No solution provided'))
                    
                    st.write("**Justification:**")
                    st.text_area("", value=solution_data.get('justification', 'No justification provided'), 
                               height=80, disabled=True, key=f"justification_{ticket['id']}")
                    
                except Exception as e:
                    st.error(f"Error displaying solution: {e}")
            
            st.divider()

def render_admin_dashboard():
    """Render comprehensive admin dashboard"""
    st.markdown('<h1 class="main-header">📈 Admin Dashboard</h1>', unsafe_allow_html=True)
    
    # Get all tickets and RAG logs
    tickets = get_all_tickets()
    rag_logs = get_rag_logs()
    
    if not tickets:
        st.info('No tickets in system.')
        return
    
    # =============================================================================
    # METRICS SECTION
    # =============================================================================
    st.subheader("📊 Ticket Status Overview")
    
    # Calculate metrics
    total_tickets = len(tickets)
    pending_tickets = len([t for t in tickets if t['status'] == 'pending'])
    processing_tickets = len([t for t in tickets if t['status'] == 'processing'])
    done_tickets = len([t for t in tickets if t['status'] == 'done'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tickets", total_tickets)
    with col2:
        st.metric("Pending", pending_tickets, delta=f"{pending_tickets} need attention" if pending_tickets > 0 else "All caught up!")
    with col3:
        st.metric("Processing", processing_tickets)
    with col4:
        st.metric("Done", done_tickets)
    
    # Status distribution chart
    status_chart = create_status_chart(tickets)
    if status_chart:
        st.plotly_chart(status_chart, use_container_width=True)
    
    st.markdown("---")
    
    # =============================================================================
    # PENDING TICKETS SECTION
    # =============================================================================
    st.subheader("⏰ Pending Tickets - Need Attention")
    
    # Get pending tickets
    pending_tickets_list = [t for t in tickets if t['status'] == 'pending']
    
    if pending_tickets_list:
        st.warning(f"You have {len(pending_tickets_list)} pending tickets that need attention!")
        
        # Display pending tickets with management options
        for ticket in pending_tickets_list:
            with st.container():
                st.markdown(f"""
                <div class="pending-ticket-card">
                    <h4>🎫 Ticket #{ticket['id']} - ⏰ PENDING</h4>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Issue Description:**")
                    st.text_area("", value=ticket['question'], height=100, disabled=True, key=f"pending_desc_{ticket['id']}")
                    
                    if ticket['created_at']:
                        st.write(f"**Created:** {ticket['created_at'].strftime('%Y-%m-%d %H:%M')}")
                
                with col2:
                    st.write("**Quick Actions:**")
                    
                    # Status update
                    new_status = st.selectbox(
                        "Update Status:",
                        ["pending", "processing", "done"],
                        index=0,
                        key=f"pending_status_{ticket['id']}"
                    )
                    
                    col_update, col_view = st.columns(2)
                    with col_update:
                        if st.button("Update", key=f"update_pending_{ticket['id']}", type="primary"):
                            if update_ticket_status(ticket['id'], new_status):
                                st.success(f"Status updated to {new_status}")
                                st.rerun()
                            else:
                                st.error("Failed to update status")
                    
                    with col_view:
                        if st.button("View Details", key=f"view_pending_{ticket['id']}"):
                            st.info(f"Ticket #{ticket['id']} details expanded above")
                
                st.divider()
    else:
        st.success("🎉 No pending tickets! All tickets are being processed or completed.")
    
    st.markdown("---")
    
    # =============================================================================
    # RAG LOGS ANALYSIS SECTION
    # =============================================================================
    if rag_logs:
        st.subheader("📋 RAG Logs Analysis")
        
        # Convert to DataFrame and process
        df_data = []
        for log in rag_logs:
            parsed_solution = parse_solution_json(log['solution'])
            
            df_data.append({
                'query_id': log['query_id'],
                'issue': log['issue'],
                'priority': parsed_solution['priority'],
                'unit': parsed_solution['unit'],
                'justification': parsed_solution['justification'],
                'solution': parsed_solution['solution'],
                'created_at': log['created_at']
            })
        
        df = pd.DataFrame(df_data)
        
        # Sidebar filters
        st.sidebar.markdown("### 🔍 Filter RAG Logs")
        
        # Priority filter
        priorities = ['All'] + sorted(df['priority'].unique().tolist())
        selected_priority = st.sidebar.selectbox("Tingkat Prioritas:", priorities)
        
        # Unit filter
        units = ['All'] + sorted(df['unit'].unique().tolist())
        selected_unit = st.sidebar.selectbox("Unit:", units)
        
        # Search functionality
        search_term = st.sidebar.text_input("🔍 Cari:", placeholder="Cari berdasarkan issue atau justifikasi...")
        
        # Apply filters
        filtered_df = filter_rag_logs(df, selected_priority, selected_unit, search_term)
        
        # Charts section
        col1, col2 = st.columns(2)
        
        with col1:
            priority_chart = create_priority_chart(filtered_df)
            if priority_chart:
                st.plotly_chart(priority_chart, use_container_width=True)
        
        with col2:
            unit_chart = create_unit_chart(filtered_df)
            if unit_chart:
                st.plotly_chart(unit_chart, use_container_width=True)
        
        # Display main table
        st.subheader(f"📋 Daftar Tiket RAG ({len(filtered_df)} tiket)")
        
        if len(filtered_df) > 0:
            # Prepare display dataframe
            display_df = filtered_df.copy()
            
            # Format datetime columns
            display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Display table with custom column configuration
            st.dataframe(
                display_df[['query_id', 'issue', 'priority', 'unit', 'justification', 'created_at']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "query_id": st.column_config.NumberColumn("Query ID", width="small"),
                    "issue": st.column_config.TextColumn("Issue", width="large"),
                    "priority": st.column_config.SelectboxColumn(
                        "Tingkat Prioritas",
                        options=['P1 (Kritis)', 'P2 (Tinggi)', 'P3 (Sedang)', 'P4 (Rendah)'],
                        width="medium"
                    ),
                    "unit": st.column_config.TextColumn("Unit", width="medium"),
                    "justification": st.column_config.TextColumn("Justifikasi Prioritas", width="large"),
                    "created_at": st.column_config.TextColumn("Dibuat", width="small")
                }
            )
            
            # Detail view section
            st.markdown("---")
            st.subheader('🔍 Detail Tiket "DONE"')
            
            # Select ticket for detail view
            query_ids = filtered_df['query_id'].tolist()
            selected_query_id = st.selectbox("Pilih Query ID untuk melihat detail:", query_ids)
            
            if selected_query_id:
                selected_ticket = filtered_df[filtered_df['query_id'] == selected_query_id].iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Query ID:**", selected_ticket['query_id'])
                    st.write("**Issue:**", selected_ticket['issue'])
                    st.write("**Tingkat Prioritas:**", selected_ticket['priority'])
                
                with col2:
                    st.write("**Unit:**", selected_ticket['unit'])
                    st.write("**Justifikasi Prioritas:**", selected_ticket['justification'])
                    st.write("**Created At:**", selected_ticket['created_at'])
                
                # Display solution details
                st.markdown("---")
                st.subheader("💡 Solusi")
                st.write(selected_ticket['solution'])
        else:
            st.info("Tidak ada tiket yang sesuai dengan filter yang dipilih.")
    else:
        st.info("No RAG logs found in system.")
    
    st.markdown("---")
    
    # =============================================================================
    # ALL TICKETS MANAGEMENT SECTION
    # =============================================================================
    st.subheader("🎫 All Tickets Management")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox("Filter by Status:", ["All", "pending", "processing", "done"])
    with col2:
        ticket_search = st.text_input("Search in tickets:", placeholder="Search by question content...")
    
    # Apply filters
    filtered_tickets = tickets
    if status_filter != "All":
        filtered_tickets = [t for t in filtered_tickets if t['status'] == status_filter]
    
    if ticket_search:
        filtered_tickets = [t for t in filtered_tickets if ticket_search.lower() in t.get('question', '').lower()]
    
    # Display filtered tickets
    if filtered_tickets:
        st.write(f"Showing {len(filtered_tickets)} ticket(s)")
        
        for ticket in filtered_tickets:
            with st.expander(f"🎫 Ticket #{ticket['id']} - {ticket['status'].upper()}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Ticket ID:** {ticket['id']}")
                    st.write(f"**Status:** {ticket['status']}")
                    if ticket['created_at']:
                        st.write(f"**Created:** {ticket['created_at'].strftime('%Y-%m-%d %H:%M')}")
                
                with col2:
                    # Manual status update
                    new_status = st.selectbox(
                        "Update Status:",
                        ["pending", "processing", "done"],
                        index=["pending", "processing", "done"].index(ticket['status']),
                        key=f"all_status_{ticket['id']}"
                    )
                    
                    if st.button("Update Status", key=f"update_all_{ticket['id']}"):
                        if update_ticket_status(ticket['id'], new_status):
                            st.success(f"Status updated to {new_status}")
                            st.rerun()
                        else:
                            st.error("Failed to update status")
                
                st.write("**Issue Description:**")
                st.text_area("Issue Description", value=ticket['question'], height=100, disabled=True, key=f"all_desc_{ticket['id']}", label_visibility="collapsed")
                
                # Show solution if available
                if ticket['solution']:
                    st.write("**Solution:**")
                    try:
                        solution_data = parse_solution_json(ticket['solution'])
                        st.json(solution_data)
                    except Exception as e:
                        st.error(f"Error parsing solution: {e}")
                        st.text(ticket['solution'])
    else:
        st.info("No tickets match your filter criteria.")

def main():
    """Main application function"""
    # Render sidebar
    user_role, page = render_sidebar()
    
    # Route to appropriate page
    if user_role == "User":
        if page == "Submit Ticket":
            render_submit_ticket()
        elif page == "Track Tickets":
            render_track_tickets()
    else:  # Admin - only Dashboard
        render_admin_dashboard()

if __name__ == "__main__":
    main()