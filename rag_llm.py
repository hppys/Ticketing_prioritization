import os
import json
import logging
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sqlalchemy import text

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database Configuration
DB_URI = "postgresql://postgres:1234@localhost:5432/postgres" # konfigurasi ke db
VECTOR_DB_CONNECTION = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain" # konfigurasi ke db untuk proses llm 
COLLECTION_NAME = "my_docs"

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('D:\\interface\\ticketing_system_cron.log'),
        logging.StreamHandler()
    ]
) # untuk mencatat semua proses ketika code di run, log akan membuar file sendiri dan akan mencatat semuanya
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv() 
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
# megambil config db pada .env


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class TicketAnalysis(BaseModel):
    issue: str = Field(description="Salin ulang pertanyaan/keluhan dari pengguna")
    priority: str = Field(description="Pilih salah satu: P1 (Kritis), P2 (Tinggi), P3 (Sedang), P4 (Rendah)")
    unit: str = Field(description="Rekomendasikan tim atau departemen yang paling sesuai")
    solution: str = Field(description="Solusi awal yang dapat membantu pengguna")
    justification: str = Field(description="Justifikasi kenapa keluhan tersebut memiliki prioritas tertentu")
# membuat base model yang akan menjadi format baku untuk output yang diinginkan dari llm 

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
        ) # hubungkan / connect ke db 
        return connection # jika connect kembalikan hasiilnya (koneksi db)
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}") # jika error, error akan di cata di log
        return None # kembalikan none

def update_ticket_status(ticket_id, new_status):
    """Update ticket status with proper error handling"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False # saat mengambil nilai dari get_db_connection() none, hasilnya akan false 
            
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
        logger.error(f"Error updating ticket status for ticket {ticket_id}: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def save_rag_result(ticket_id, issue, solution, sources):
    """Save RAG processing result to database"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False
            
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO data.rag_logs (query_id, issue, solution, sources)
                VALUES (%s, %s, %s, %s)
            """, (ticket_id, issue, json.dumps(solution, ensure_ascii=False), json.dumps(sources, ensure_ascii=False)))
            conn.commit()
            logger.info(f"RAG result saved for ticket {ticket_id}")
            return True
            
    except psycopg2.Error as e:
        logger.error(f"Error saving RAG result for ticket {ticket_id}: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

# =============================================================================
# RAG SYSTEM INITIALIZATION
# =============================================================================

def initialize_rag_system():
    """Initialize RAG system components for a non-interactive script."""
    try:
        chat_model = ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            model='gemini-2.5-pro',
            temperature=0.3
        )
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        db = SQLDatabase.from_uri(DB_URI, schema="data")
        
        def load_documents_from_db():
            with db._engine.connect() as conn:
                try:
                    result = conn.execute(text("""
                        SELECT DISTINCT "ID Tiket", "Keluhan", "Prioritas", "Justifikasi Prioritas", "Unit Penanggung Jawab", "Solusi Awal (untuk Pelanggan)"
                        FROM data.dataset_dummy_ticketing
                        WHERE "Keluhan" IS NOT NULL AND "Prioritas" IS NOT NULL;
                    """))
                    
                    rows = result.fetchall()
                    docs = [
                        Document(
                            page_content=row[1],
                            metadata={
                                "ticket_id": row[0], 
                                "prioritas": row[2], 
                                "justifikasi_prioritas": row[3] or "",
                                "unit_penanggung_jawab": row[4] or "", 
                                "solusi_awal": row[5] or ""
                            }
                        ) for row in rows
                    ]
                    logger.info(f"Loaded {len(docs)} documents from database")
                    return docs
                except Exception as e:
                    logger.error(f"Error loading documents from database: {e}")
                    return []
        
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=VECTOR_DB_CONNECTION,
            use_jsonb=True,
        )
        
        docs = load_documents_from_db()
        if docs:
            try:
                # Opsi: Hapus dan buat ulang untuk memastikan data terbaru.
                vector_store.delete_collection()
                vector_store.create_collection()
                
                batch_size = 100
                for i in range(0, len(docs), batch_size):
                    batch = docs[i:i + batch_size]
                    vector_store.add_documents(batch)
                logger.info(f"Successfully added {len(docs)} documents to vector store")
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {e}")
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        parser = JsonOutputParser(pydantic_object=TicketAnalysis)
        
        system_prompt = (
            "Anda adalah asisten AI yang sangat efisien untuk tim dukungan IT. "
            "Tugas utama Anda adalah menganalisis tiket masalah baru yang diberikan oleh pengguna, "
            "dan berdasarkan deskripsi tiket tersebut serta konteks historis yang relevan "
            "yang diambil dari basis pengetahuan, tentukan informasi berikut:\n\n"
            "1. **issue**: Salin ulang pertanyaan/keluhan dari pengguna.\n"
            "2. **priority**: Pilih salah satu dari P1 (Kritis), P2 (Tinggi), P3 (Sedang), atau P4 (Rendah).\n"
            "3. **unit**: Rekomendasikan tim atau departemen yang paling sesuai untuk menangani tiket ini.\n"
            "4. **solution**: Solusi awal yang dapat membantu pengguna.\n"
            "5. **justification**: Justifikasi kenapa keluhan tersebut memiliki prioritas tertentu.\n"
            "\n"
            "Analisis berdasarkan konteks tiket serupa berikut:\n{context}\n\n"
            "{format_instructions}\n\n"
            "PENTING: Pastikan output dalam format JSON yang valid dengan struktur yang tepat."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())
        
        question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        def parse_rag_output(inputs):
            result = None
            try:
                result = rag_chain.invoke(inputs)
                parsed_answer = parser.parse(result['answer'])
                
                if not isinstance(parsed_answer, dict):
                    raise ValueError("Parsed result is not a dictionary")
                
                required_fields = ['issue', 'priority', 'unit', 'solution', 'justification']
                for field in required_fields:
                    if field not in parsed_answer or not parsed_answer[field]:
                        parsed_answer[field] = 'Tidak Diketahui'
                
                valid_priorities = ['P1 (Kritis)', 'P2 (Tinggi)', 'P3 (Sedang)', 'P4 (Rendah)']
                if parsed_answer['priority'] not in valid_priorities:
                    parsed_answer['priority'] = 'P3 (Sedang)' # Default priority
                
                return {'answer': parsed_answer, 'context': result['context']}
            except Exception as e:
                raw_answer = result.get('answer', 'N/A') if result else 'N/A'
                logger.error(f"RAG parsing error: {e}. Raw answer: {raw_answer}")
                return {
                    'answer': {
                        'issue': inputs.get('input', 'Tidak Diketahui'), 'priority': 'P3 (Sedang)', 'unit': 'IT Support',
                        'solution': 'Tiket sedang diproses, mohon menunggu respons dari tim terkait.',
                        'justification': 'Prioritas sedang karena memerlukan analisis lebih lanjut.'
                    },
                    'context': [], 'parse_error': str(e)
                }
        
        logger.info("RAG system initialized successfully")
        return parse_rag_output, db
        
    except Exception as e:
        logger.error(f"Fatal error initializing RAG system: {e}", exc_info=True)
        return None, None

# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def process_single_ticket(ticket_id, question, rag_chain_parser):
    """Process a single ticket using the RAG chain parser."""
    try:
        logger.info(f"Processing ticket {ticket_id}: {question[:50]}...")
        
        if not update_ticket_status(ticket_id, 'processing'):
            return False, "Failed to update ticket status to 'processing'"
        
        result = rag_chain_parser({"input": question})
        parsed_answer = result['answer']
        
        # Jika ada error parsing, 'parse_error' akan ada di dalam result
        if 'parse_error' in result:
             raise Exception(f"Parsing failed: {result['parse_error']}")

        sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in result['context']]
        
        if save_rag_result(ticket_id, question, parsed_answer, sources):
            if update_ticket_status(ticket_id, 'done'):
                logger.info(f"Successfully processed ticket {ticket_id}")
                return True, parsed_answer
            else:
                logger.error(f"Failed to update ticket {ticket_id} status to 'done'")
                return False, "Failed to update ticket status to 'done'"
        else:
            logger.error(f"Failed to save RAG result for ticket {ticket_id}")
            update_ticket_status(ticket_id, 'pending') # Revert status
            return False, "Failed to save RAG result"
            
    except Exception as e:
        logger.error(f"Error processing ticket {ticket_id}: {e}", exc_info=True)
        update_ticket_status(ticket_id, 'pending') # Revert status
        return False, str(e)

def process_tickets_batch(rag_chain_parser, max_tickets=5):
    """Process a batch of pending tickets."""
    try:
        conn = get_db_connection()
        if conn is None:
            return {'error': 'Database connection failed'}
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, question FROM data.rag_queries
                WHERE status = 'pending' ORDER BY id ASC LIMIT %s
            """, (max_tickets,))
            pending_tickets = cur.fetchall()
        
        conn.close()
        
        if not pending_tickets:
            return {'message': 'No pending tickets found'}
        
        results = []
        for ticket in pending_tickets:
            success, result_data = process_single_ticket(
                ticket['id'], ticket['question'], rag_chain_parser
            )
            results.append({
                'ticket_id': ticket['id'],
                'success': success,
                'result': result_data if success else f"Error: {result_data}"
            })
        return {'results': results}
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}", exc_info=True)
        return {'error': str(e)}

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

def run_job():
    """Main function to be executed by the scheduler."""
    logger.info("Cron job started: Processing pending tickets.")

    rag_chain_parser, db = initialize_rag_system()
    if not rag_chain_parser:
        logger.error("Failed to initialize RAG system. Aborting job.")
        return

    try:
        max_tickets_to_process = 10
        result = process_tickets_batch(rag_chain_parser, max_tickets=max_tickets_to_process)

        if 'error' in result:
            logger.error(f"Batch processing failed with error: {result['error']}")
        elif 'message' in result:
            logger.info(f"Batch processing status: {result['message']}")
        elif 'results' in result:
            processed_count = len(result['results'])
            success_count = sum(1 for r in result['results'] if r['success'])
            logger.info(f"Batch processing finished. Processed: {processed_count}, Successful: {success_count}.")
            for res in result['results']:
                if not res['success']:
                    logger.warning(f"Failed ticket ID {res['ticket_id']}: {res['result']}")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the job: {e}", exc_info=True)

    logger.info("Cron job finished.")


if __name__ == "__main__":
    run_job()
