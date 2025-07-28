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
from sqlalchemy.engine.url import make_url
# =============================================================================
# CONFIGURATION
# =============================================================================

# Database Configuration
VECTOR_DB_CONNECTION = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain" # konfigurasi db vectorpg
COLLECTION_NAME = "my_docs"


# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('D:\\interface\\ticketing_system_cron.log'),
        logging.StreamHandler()
    ]
) 
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv() 
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class TicketAnalysis(BaseModel):
    issue: str = Field(description="Salin ulang pertanyaan/keluhan dari pengguna")
    priority: str = Field(description="Pilih salah satu: P1 (Kritis), P2 (Tinggi), P3 (Sedang), P4 (Rendah)")
    unit: str = Field(description="Rekomendasikan tim atau departemen yang paling sesuai")
    solution: str = Field(description="solution awal yang dapat membantu pengguna")
    justification: str = Field(description="Justifikasi kenapa keluhan tersebut memiliki prioritas tertentu")
# membuat base model yang akan menjadi format baku untuk output yang diinginkan dari llm 

# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def get_db_connection():
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
        return None


def update_ticket_status(ticket_id, new_status):
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False
            
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE data.rag_queries
                SET status = %s
                WHERE ticket_id = %s
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
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False
            
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO data.rag_logs (ticket_id, question, solution, sources)
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
def load_documents_from_db():
        conn = get_db_connection()
        if conn is None:
            logger.error("Error loading documents from database: {e}")
            return []
        
        try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT DISTINCT "ticket_id", "question", "priority", "justification", "unit", "solution"
                        FROM data.dataset_dummy_ticketing
                        WHERE "question" IS NOT NULL AND "priority" IS NOT NULL;
                    """)
                    rows = cur.fetchall()
                    docs = [
                        Document(
                            page_content=row["question"],
                            metadata={
                                "ticket_id": row["ticket_id"], 
                                "priority": row["priority"], 
                                "justification": row["justification"] or "",
                                "unit": row["unit"] or "", 
                                "solution": row["solution"] or ""
                            }
                        ) for row in rows
                    ]
                    logger.info(f"Loaded {len(docs)} documents from database")
                    return docs
        except Exception as e:
                    logger.error(f"Error loading documents from database: {e}")
                    return []
        finally:
            conn.close()

def refresh_vector_store(vector_store, batch_size=100):
    docs = load_documents_from_db()
    if not docs:
        logger.warning("No documents loaded. Skipping vector store update.")
        return

    try:
        vector_store.delete_collection()
        vector_store.create_collection()

        for i in range(0, len(docs), batch_size):
            vector_store.add_documents(docs[i:i + batch_size])
        logger.info(f"Successfully refreshed vector store with {len(docs)} documents.")
    except Exception as e:
        logger.error(f"Error refreshing vector store: {e}")

def convert_sqlalchemy_dsn_to_psycopg2(dsn):
    url = make_url(dsn)
    return f"dbname={url.database} user={url.username} password={url.password} host={url.host} port={url.port}"


def initialize_rag_pipeline():
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

        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=VECTOR_DB_CONNECTION,
            use_jsonb=True,
        )

        try:
            psycopg2_dsn = convert_sqlalchemy_dsn_to_psycopg2(VECTOR_DB_CONNECTION)
            with psycopg2.connect(psycopg2_dsn) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = "
                        "(SELECT uuid FROM langchain_pg_collection WHERE name = %s)",
                        (COLLECTION_NAME,)
                        )

                    result = cursor.fetchone()
                    embedding_count = result[0] if result else 0

                    if embedding_count == 0:
                        logger.warning(f"Vector store '{COLLECTION_NAME}' masih kosong. Lakukan inisialisasi dokumen terlebih dahulu.")
                        refresh_vector_store(vector_store, batch_size=100)
        except Exception as ve:
            logger.error(f"Gagal mengecek isi vector store: {ve}", exc_info=True)

        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        parser = JsonOutputParser(pydantic_object=TicketAnalysis)

        system_prompt = (
            "Anda adalah asisten AI yang sangat efisien untuk tim dukungan IT. "
            "Tugas utama Anda adalah menganalisis tiket masalah baru yang diberikan oleh pengguna, "
            "dan berdasarkan deskripsi tiket tersebut serta konteks historis yang relevan "
            "yang diambil dari basis pengetahuan, tentukan informasi berikut:\n\n"
            "1. *issue*: Salin ulang pertanyaan/keluhan dari pengguna.\n"
            "2. *priority*: Pilih salah satu dari P1 (Kritis), P2 (Tinggi), P3 (Sedang), atau P4 (Rendah).\n"
            "3. *unit*: Rekomendasikan tim atau departemen yang paling sesuai untuk menangani tiket ini.\n"
            "4. *solution*: Solusi awal yang dapat membantu pengguna.\n"
            "5. *justification*: Justifikasi kenapa keluhan tersebut memiliki prioritas tertentu.\n\n"
            "Analisis berdasarkan konteks tiket serupa berikut:\n{context}\n\n"
            "{format_instructions}\n\n"
            "PENTING: Pastikan output dalam format JSON yang valid dengan struktur yang tepat."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
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

                # Ensure all required fields are filled
                for field in ['issue', 'priority', 'unit', 'solution', 'justification']:
                    if field not in parsed_answer or not parsed_answer[field]:
                        parsed_answer[field] = 'Tidak Diketahui'

                valid_priorities = ['P1 (Kritis)', 'P2 (Tinggi)', 'P3 (Sedang)', 'P4 (Rendah)']
                if parsed_answer['priority'] not in valid_priorities:
                    parsed_answer['priority'] = 'P3 (Sedang)'

                return {'answer': parsed_answer, 'context': result['context']}
            except Exception as e:
                raw_answer = result.get('answer', 'N/A') if result else 'N/A'
                logger.error(f"RAG parsing error: {e}. Raw answer: {raw_answer}")
                return {
                    'answer': {
                        'issue': inputs.get('input', 'Tidak Diketahui'),
                        'priority': 'P3 (Sedang)',
                        'unit': 'IT Support',
                        'solution': 'Tiket sedang diproses, mohon menunggu respons dari tim terkait.',
                        'justification': 'Prioritas sedang karena memerlukan analisis lebih lanjut.'
                    },
                    'context': [], 'parse_error': str(e)
                }

        logger.info("RAG pipeline initialized successfully")
        return parse_rag_output, vector_store

    except Exception as e:
        logger.error(f"Fatal error initializing RAG pipeline: {e}", exc_info=True)
        return None, None


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def process_single_ticket(ticket_id, question, rag_chain_parser):
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
                SELECT ticket_id, question FROM data.rag_queries
                WHERE status = 'pending' ORDER BY ticket_id ASC LIMIT %s
            """, (max_tickets,))
            pending_tickets = cur.fetchall()
        
        conn.close()
        
        if not pending_tickets:
            return {'message': 'No pending tickets found'}
        
        results = []
        for ticket in pending_tickets:
            success, result_data = process_single_ticket(
                ticket['ticket_id'], ticket['question'], rag_chain_parser
            )
            results.append({
                'ticket_id': ticket['ticket_id'],
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
    logger.info("Starting to  processing pending tickets.")

    rag_chain_parser = initialize_rag_pipeline()
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

    logger.info("Finished.")


if __name__ == "__main__":
    run_job()
