
from flask import Flask, flash, request, jsonify, render_template, session, redirect, url_for
from functools import wraps
import bcrypt
import jwt
import datetime
import psycopg2
import psycopg2.extras
import pgvector.psycopg2
from sentence_transformers import SentenceTransformer, CrossEncoder
import pdfplumber
import io
import time
import nltk
import os
import json
import requests
import uuid
from dotenv import load_dotenv
import fitz
import numpy as np
from typing import Dict, Any
from nltk.tokenize import sent_tokenize
# CRITICAL FIX: Handling Decimals for JSON
from decimal import Decimal
from flask.json.provider import DefaultJSONProvider

# LOAD SECRETS
load_dotenv()


# --- 1. CUSTOM JSON PROVIDER (Prevents Admin Panel Crashes) ---
# This fixes the errors you had with Latitude/Longitude and Numpy arrays
class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super().default(obj)


app = Flask(__name__)
app.json = CustomJSONProvider(app)

# CONFIG FROM .ENV
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "vigilance_bot")
DB_USER = os.getenv("DB_USER", "postgres")
VAULT_KEY = os.getenv("VAULT_KEY", "CyberVigilance2025")

SYSTEM_PROMPT = """
You are Vig-Bot, a professional AI assistant for Kerala Cyber Laws. 

CRITICAL RULES:
1. DO NOT greet with "Hello! I am Cy-Bot..." UNLESS the user's first message is a greeting
2. Answer ONLY using the provided context - NEVER use general knowledge
3. Keep answers under 150 words
4. Use ONLY HTML formatting:  <p>, <strong>, <ul>, <li> tags
5. Always cite the source in brackets:  [Source: Section 66, IT Act 2000]
6. If information is incomplete, say:  "Based on available information..."
7. Never invent laws or punishments
8.When a PDF is uploaded, first determine whether its content is related to cyber laws,cybercrime, IT Act, digital privacy, online fraud, or related legal frameworks. If it is relevant, extract and use the information to answer the user’s cyber-related query. If the PDF is not relevant, clearly acknowledge that the document is unrelated to cyber laws and do not rely on it. In cases where the user’s question is cyber-related but the uploaded PDF is unrelated (or vice versa), answer the cyber-related query independently while explicitly noting that the uploaded document is not applicable.



RESPONSE FORMAT (MANDATORY):
- Start with:  "Based on Kerala's cybFer laws..." or "According to the IT Act..."
- Use <p> tags for paragraphs (max 3 sentences each)
- Use <ul><li> for lists
- End with disclaimer:  <p><em>I am an AI assistant, not a lawyer. For official matters, consult authorities.</em></p>

SCOPE (ONLY answer these):
- Kerala cyber laws
- Information Technology Act, 2000
- Indian Penal Code sections on cybercrime
- DPDP Act
- Cybercrime reporting procedures

OUT OF SCOPE (Politely refuse):
- General knowledge questions
- Personal legal advice
- Politics, medical advice
- Non-cyber law topics

TONE: 
- Professional, formal, empathetic
- Clear and concise
- Non-judgmental, supportive for victims

END GOAL
--------
Your goal is to make cyber laws understandable and accessible to every citizen
without replacing legal professionals.

Always prioritize accuracy over completeness."""

# --- NEW: MALAYALAM SYSTEM PROMPT ---
MALAYALAM_SYSTEM_PROMPT = """
[YOUR ORIGINAL PROMPT HERE - COPY PASTE THE ENTIRE SYSTEM_PROMPT ABOVE]

**CRUCIAL MALAYALAM OUTPUT INSTRUCTION:**
- You must respond **only in Malayalam**.
- The main body of your explanation, greetings, and all conversational text must be in Malayalam.
- **CRITICAL EXCEPTION:** Do not translate specific legal acts, section numbers, or technical terms. Keep them in their original English form.
    - Examples: "Information Technology Act, 2000", "Section 66A", "IPC", "Cyber Appellate Tribunal", "Phishing".
- Format your response in valid HTML, just as described in the original prompt.
- Your persona and all other rules from the original prompt remain the same.
"""

# --- NEW: SENIOR CITIZEN MODE INSTRUCTION ---
SENIOR_CITIZEN_INSTRUCTION = """
**IMPORTANT: SENIOR CITIZEN MODE ACTIVATED.**
You MUST adhere to the following rules in addition to all previous instructions:
- Use very simple, short, and clear sentences.
- Explain complex legal terms in a very easy-to-understand way.
- Be extra patient, empathetic, and encouraging in your tone.
- Break down any procedures into simple, numbered steps.
- Avoid using jargon. If a legal term is necessary, explain it immediately in parentheses.
- Your goal is to make the user feel safe, understood, and confident.
"""

# --- ADMIN USERS ---
ADMIN_USERS = {
    "admin": "admin123"
}

# --- GLOBAL STORES ---
session_pdf_store: Dict[str, Any] = {}

# Fix NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# --- EMBEDDING MODELS ---
try:
    print("Loading Embedding Model...")
    # Using all-MiniLM-L6-v2 to match your existing database embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Loading Re-Ranker Model...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    reranker = None


# --- HELPERS ---
def get_db_connection():
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
        pgvector.psycopg2.register_vector(conn)
        return conn
    except Exception as e:
        print(f"Database error: {e}")
        return None


def create_embedding_vector(text):
    if not text: return None
    clean_text = text.replace("'", "").replace('"', "").strip()
    return model.encode([clean_text], convert_to_tensor=False)[0].tolist()


def chunk_text(text, chunk_size=500):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current_chunk_length + sentence_words <= chunk_size:
            current_chunk.append(sentence)
            current_chunk_length += sentence_words
        else:
            chunks.append(" ".join(current_chunk))
            overlap_sentences = current_chunk[-3:]
            current_chunk = overlap_sentences + [sentence]
            current_chunk_length = sum(len(s.split()) for s in current_chunk)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['username']
        except Exception:
            return jsonify({'message': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)

    return decorated


def log_event(conn, table, rec_id, action, old_val, new_val, user):
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO audit_logs (table_name, record_id, action_type, old_data, new_data, changed_by) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (table, rec_id, action, json.dumps(old_val, default=str), json.dumps(new_val, default=str), user))
    except Exception as e:
        print(f"Audit Logging Error: {e}")


# --- NEW HELPER: AI STRUCTURED EXTRACTION ---
# --- REPLACEMENT FUNCTION FOR app.py ---

def extract_legal_data_with_llm(text):
    """
    Uses Llama 3.1 to analyze the PDF text.
    Includes a 'Cleaning' step to handle AI formatting errors.
    """
    # 1. Read more text to ensure we catch the section
    input_text = text[:4500]

    prompt = f"""
    You are a legal expert AI. Analyze the text below.

    TASK: Extract these details into a valid JSON object.

    1. "section": The Section Number (e.g. "Section 66A").
    2. "chapter": The Chapter Heading (e.g. "Chapter XI").
    3. "section_name": The Title of the section.
    4. "description": Write a clear 2-3 sentence SUMMARY of the legal provision. Do not copy the whole text.
    5. "punishment": Search for penalties (years of jail, fine amounts). If found, summarize them. If none, write "Not Specified".

    CRITICAL: Return ONLY the JSON object. Do not write explanations.

    TEXT:
    {input_text}
    """

    for attempt in range(2):
        try:
            print(f"AI Extraction Attempt {attempt + 1}...")
            response = requests.post('http://localhost:11434/api/generate',
                                     json={
                                         "model": "llama3.1",
                                         "prompt": prompt,
                                         "stream": False,
                                         "format": "json",
                                         "options": {"temperature": 0.1, "num_ctx": 4096}
                                     }, timeout=60)

            raw_response = response.json()['response']

            # --- NEW: CLEAN THE AI OUTPUT ---
            # Sometimes AI adds markdown ```json ... ```. We remove it.
            clean_json = raw_response.replace("```json", "").replace("```", "").strip()

            # Parse the clean string
            result = json.loads(clean_json)

            # Check if it actually found a section
            if result.get('section'):
                return result

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(1)

    return None


# --- ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        # .strip() removes accidental spaces, .lower() ignores Capital Letters
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '').strip()
        
        print(f"Login Attempt - User: '{username}', Pass: '{password}'") # Look at your terminal for this!

        if username == "admin" and password == "admin123":
            session.clear()
            session['admin_logged_in'] = True
            
            token = jwt.encode({
                'username': username,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            }, app.config['SECRET_KEY'], algorithm="HS256")
            
            session['auth_token'] = token
            return redirect(url_for('view_admin'))
        else:
            return f"Login Failed. You typed User: '{username}' and Pass: '{password}'"
            
    return render_template('login.html')

@app.route('/api/logout_log', methods=['POST'])
@token_required
def logout_log(current_user):
    conn = get_db_connection()
    log_event(conn, 'auth', 0, 'LOGOUT', None, {"status": "success"}, current_user)
    conn.commit()
    conn.close()
    return jsonify({"success": True}), 200


@app.route('/api/query', methods=['POST'])
def chatbot_query():
    data = request.get_json()
    user_query = data.get('query', '').strip()
    language = data.get('language', 'en')
    mode = data.get('mode', 'normal')
    session_pdf_id = data.get('session_pdf_id')  # Get session PDF ID if provided

    if not user_query:
        return jsonify({"message": "Query required"}), 400

    # --- Greeting Logic ---
    greetings = ['hi', 'hello', 'hey', 'good morning', 'who are you', 'what are you', 'നമസ്കാരം']
    if any(word in user_query.lower() for word in greetings):
        try:
            base_prompt = MALAYALAM_SYSTEM_PROMPT if language == 'ml' else SYSTEM_PROMPT
            final_prompt = (SENIOR_CITIZEN_INSTRUCTION + base_prompt) if mode == 'senior' else base_prompt
            full_prompt = f"{final_prompt}\n\nUSER QUERY: {user_query}\n\nINSTRUCTION: The user is greeting you. Respond warmly as Cy-Bot."

            # To this:
            ai_res = requests.post('http://localhost:11434/api/generate',
                                   json={"model": "llama3.1", "prompt": full_prompt, "stream": False, "options": {
                                       "num_predict": 400, "temperature": 0.3, "top_p": 0.9, "num_ctx": 4096
                                   }}, timeout=300)  # Increased to 300 seconds

            res_data = ai_res.json()
            return jsonify({"response": res_data.get('response', "Hello!"), "relevant_sections": []}), 200
        except Exception as e:
            print(f"Greeting Error: {e}")
            fallback = "നമസ്കാരം!" if language == 'ml' else "Hello! I am Cy-Bot. How can I assist you today?"
            return jsonify({"response": fallback, "relevant_sections": []}), 200

    conn = get_db_connection()
    if conn is None:
        return jsonify({"message": "Database error"}), 500
    cur = conn.cursor()

    try:
        query_vector = create_embedding_vector(user_query)
        fts_keywords = " | ".join(user_query.split())

        # --- UPDATED SQL: Searches Meta AND Text ---
        global_rrf_sql = """
        WITH all_content AS (
            SELECT 
                'Law' as type, 
                s.sec_no || ': ' || s.title as meta, 
                cn.text_content as text, 
                cn.id::text as doc_id
            FROM content_nodes cn
            JOIN sections s ON cn.section_id = s.id
            WHERE cn.is_active = TRUE
        ),
        keyword_search AS (
            SELECT doc_id, ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(to_tsvector('english', meta || ' ' || text), websearch_to_tsquery('english', %s)) DESC
            ) as k_rank
            FROM all_content
            WHERE to_tsvector('english', meta || ' ' || text) @@ websearch_to_tsquery('english', %s)
        ),
        vector_search AS (
            SELECT id::text as doc_id, ROW_NUMBER() OVER (ORDER BY embedding <-> %s::vector ASC) as v_rank
            FROM content_nodes
            WHERE (embedding <-> %s::vector) < 1.5
        )
        SELECT ac.type, ac.meta, ac.text, 
            (COALESCE(1.0 / (60 + k_rank), 0) + COALESCE(1.0 / (60 + v_rank), 0)) as rrf_score
        FROM all_content ac
        LEFT JOIN keyword_search ks ON ac.doc_id = ks.doc_id
        LEFT JOIN vector_search vs ON ac.doc_id = vs.doc_id
        WHERE k_rank IS NOT NULL OR v_rank IS NOT NULL
        ORDER BY rrf_score DESC LIMIT 5;
        """

        cur.execute(global_rrf_sql, (fts_keywords, fts_keywords, query_vector, query_vector))
        raw_results = cur.fetchall()

        combined_results = []
        for r in raw_results:
            combined_results.append({"type": r[0], "meta": r[1], "text": r[2], "rrf_score": float(r[3])})

        if reranker and combined_results:
            pairs = [[user_query, res['text']] for res in combined_results]
            scores = reranker.predict(pairs)
            for i, res in enumerate(combined_results): res['rerank_score'] = float(scores[i])
            top_results = sorted(combined_results, key=lambda x: x['rerank_score'], reverse=True)[:3]
        else:
            top_results = combined_results[:3]

        # Build context from database results
        combined_context = ""
        sources = []
        for res in top_results:
            label = f"{res['type']}: {res['meta']}"
            combined_context += f"SOURCE: {label}\nCONTENT: {res['text']}\n\n"
            sources.append({"source": label, "relevance": "High", "context": res['text'][:200] + "..."})

        # --- NEW: Add session PDF content if provided ---
        if session_pdf_id and session_pdf_id in session_pdf_store:
            pdf_data = session_pdf_store[session_pdf_id]
            # Add PDF content as additional context
            combined_context += f"UPLOADED DOCUMENT ({pdf_data['filename']}):\n{pdf_data['text'][:2000]}\n\n"
            # Add to sources list
            sources.insert(0, {
                "source": f"Uploaded Document: {pdf_data['filename']}",
                "relevance": "High",
                "context": pdf_data['text'][:200] + "..."
            })

        if not combined_context:
            fallback = "<p>വിവരങ്ങൾ ലഭ്യമല്ല.</p>" if language == 'ml' else "<p>I could not find verified information in the database.</p>"
            return jsonify({"response": fallback, "relevant_sections": []})

        base_prompt = MALAYALAM_SYSTEM_PROMPT if language == 'ml' else SYSTEM_PROMPT
        final_prompt = (SENIOR_CITIZEN_INSTRUCTION + base_prompt) if mode == 'senior' else base_prompt

        full_prompt = f"{final_prompt}\n\nCONTEXT:\n{combined_context}\n\nUSER QUERY: {user_query}"

        # --- IMPROVED OLLAMA REQUEST WITH BETTER ERROR HANDLING ---
        try:
            print(f"Sending request to Ollama with prompt length: {len(full_prompt)}")
            ai_res = requests.post('http://localhost:11434/api/generate',
                                   json={"model": "llama3.1", "prompt": full_prompt, "stream": False, "options": {
                                       "num_predict": 1500, "temperature": 0.3, "top_p": 0.9, "num_ctx": 4096
                                   }}, timeout=120)

            # Check if request was successful
            if ai_res.status_code != 200:
                error_msg = f"Ollama returned status code {ai_res.status_code}: {ai_res.text}"
                print(error_msg)
                return jsonify({"message": "Ollama service error", "error": error_msg}), 503

            data = ai_res.json()
            if 'response' in data:
                return jsonify({"response": data['response'], "relevant_sections": sources}), 200
            else:
                error_msg = f"Ollama response missing 'response' field: {data}"
                print(error_msg)
                return jsonify({"message": "Invalid Ollama response", "error": error_msg}), 500

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Cannot connect to Ollama at localhost:11434: {str(e)}"
            print(error_msg)
            return jsonify({"message": "Ollama service not running", "error": error_msg}), 503
        except requests.exceptions.Timeout as e:
            error_msg = f"Ollama request timed out: {str(e)}"
            print(error_msg)
            return jsonify({"message": "Ollama request timed out", "error": error_msg}), 503
        except Exception as e:
            error_msg = f"Unexpected error with Ollama: {str(e)}"
            print(error_msg)
            return jsonify({"message": "Ollama error", "error": error_msg}), 500

    except Exception as e:
        print(f"Search Error: {e}")
        return jsonify({"message": "Search error", "error": str(e)}), 500
    finally:
        conn.close()


@app.route('/api/chat/upload_session', methods=['POST'])
def upload_chat_pdf():
    """
    Handle PDF uploads for chat sessions.
    Stores the PDF content temporarily with a session ID.
    """
    if 'file' not in request.files:
        return jsonify({"message": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No file selected"}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"message": "Only PDF files are allowed"}), 400

    try:
        # Extract text from PDF
        full_text = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)

        if not full_text:
            return jsonify({"message": "Could not extract text from PDF"}), 400

        # Generate a unique session ID
        session_id = str(uuid.uuid4())

        # Store in global session store with timestamp
        session_pdf_store[session_id] = {
            'filename': file.filename,
            'text': '\n'.join(full_text),
            'timestamp': datetime.datetime.now()
        }

        # Clean up old sessions (older than 1 hour)
        current_time = datetime.datetime.now()
        expired_sessions = [
            sid for sid, data in session_pdf_store.items()
            if (current_time - data['timestamp']).seconds > 3600
        ]
        for sid in expired_sessions:
            del session_pdf_store[sid]

        return jsonify({
            "session_id": session_id,
            "message": "PDF uploaded successfully"
        }), 200

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return jsonify({"message": "Error processing PDF"}), 500


@app.route('/api/chat/cleanup_sessions', methods=['POST'])
def cleanup_pdf_sessions():
    """Clean up expired PDF sessions (older than 1 hour)"""
    try:
        current_time = datetime.datetime.now()
        expired_sessions = [
            sid for sid, data in session_pdf_store.items()
            if (current_time - data['timestamp']).seconds > 3600
        ]
        for sid in expired_sessions:
            del session_pdf_store[sid]
        return jsonify({"message": f"Cleaned up {len(expired_sessions)} expired sessions"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- ADMIN ROUTES (Using Your Corrected Logic) ---
@app.route('/')
def index(): return render_template('chat.html')


@app.route('/chat')
def view_chat(): return render_template('chat.html')


@app.route('/admin')
def view_admin():
    # If this isn't True, we know for a fact why it's redirecting
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return render_template('admin.html')






# --- FUNCTION ---
@app.route('/api/admin/pdf/preview', methods=['POST'])
@token_required
def preview_pdf(current_user):
    if 'file' not in request.files: 
        return jsonify({"message": "No file"}), 400
    
    file = request.files['file']

    # 1. Extract Raw Text (The Raw Material)
    full_text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() or ""

    # 2. Call the "Chef" (Extraction Step)
    # This function uses the 'Context Object' logic internally to give us a tree structure
    extracted_data = extract_legal_data_with_llm(full_text)

    # Validation: We need at least a Law Code and Section No to file this
    if not extracted_data or not extracted_data.get('section_no'):
        return jsonify({
            "status": "error",
            "message": "Could not identify structure. Ensure PDF contains clear Section and Chapter headers."
        }), 200

    # 3. The "Librarian Check" (Database Lookup)
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # We need to see if this specific hierarchy exists: Law -> Chapter -> Section
    # We use the 'short_code' (e.g., 'BNSS') and 'sec_no' (e.g., '105')
    query = """
        SELECT 
            s.id as section_id, s.title as section_title, s.content as section_content,
            c.id as chapter_id, c.title as chapter_title,
            l.id as law_id, l.short_code
        FROM sections s
        JOIN chapters c ON s.chap_id = c.id
        JOIN legal_sources l ON c.law_id = l.id
        WHERE l.short_code = %s AND s.sec_no = %s
    """
    
    # We assume extracted_data has 'law_code' (e.g., BNSS) and 'section_no'
    law_code = extracted_data.get('law_code', 'BNSS') # Default to BNSS for now
    sec_no = extracted_data.get('section_no')

    cur.execute(query, (law_code, sec_no))
    existing_record = cur.fetchone()

    old_data_structure = None

    # 4. If it exists, fetch ALL related children (The "All Tables" Requirement)
    if existing_record:
        section_id = existing_record['section_id']
        
        # Fetch Subsections
        cur.execute("SELECT * FROM sub_sections WHERE section_id = %s ORDER BY id", (section_id,))
        existing_sub_sections = cur.fetchall()

        # Fetch Explanations
        cur.execute("SELECT * FROM explanations WHERE section_id = %s ORDER BY id", (section_id,))
        existing_explanations = cur.fetchall()

        # Fetch Illustrations
        cur.execute("SELECT * FROM illustrations WHERE section_id = %s ORDER BY id", (section_id,))
        existing_illustrations = cur.fetchall()

        # Fetch Compounding Rules (if any)
        cur.execute("SELECT * FROM compounding_rules WHERE bns_section_ref = %s", (section_id,))
        existing_rules = cur.fetchall()

        # Build the 'Old Data' tree to compare against
        old_data_structure = {
            "law_info": existing_record,
            "sub_sections": existing_sub_sections,
            "explanations": existing_explanations,
            "illustrations": existing_illustrations,
            "compounding_rules": existing_rules,
            # Add placeholders for schedule/forms if needed
            "first_schedule": [], 
            "form_templates": [] 
        }

    conn.close()

    # 5. Return Comparison
    return jsonify({
        "status": "update" if existing_record else "new",
        "law_code": law_code,
        "section_no": sec_no,
        "new_data": extracted_data, # The fresh tree from the PDF
        "old_data": old_data_structure, # The existing tree from DB (if any)
        "raw_text": full_text
    }), 200

@app.route('/api/admin/pdf/confirm/', methods=['POST'])
@token_required
def confirm_pdf_ingestion(current_user):
    data = request.get_json()
    # 'hierarchy' is the structured data we reviewed in Step 2
    hierarchy = data.get('new_data') 
    filename = data.get('filename')
    raw_text = data.get('raw_text')

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # 1. TABLE: uploaded_documents
        # We save the original source first
        cur.execute("INSERT INTO uploaded_documents (file_name, raw_pdf_data) VALUES (%s, %s) RETURNING document_id",
                    (filename, raw_text))
        doc_id = cur.fetchone()[0]

   
        # 2. TABLE: legal_sources
        cur.execute("""
        INSERT INTO legal_sources (short_code, full_name, year) 
         VALUES (%s, %s, %s) 
        ON CONFLICT (short_code) DO UPDATE SET short_code = EXCLUDED.short_code 
        RETURNING id
        """, (hierarchy.get('law_code', 'BNSS'), hierarchy.get('law_name', 'Law Name'), hierarchy.get('year', 2024))) # Added hierarchy.get('year')
        law_id = cur.fetchone()[0]

        # 3. TABLE: chapters
        cur.execute("INSERT INTO chapters (law_id, chap_no, title) VALUES (%s, %s, %s) RETURNING id",
                    (law_id, hierarchy.get('chap_no'), hierarchy.get('chap_title')))
        chap_id = cur.fetchone()[0]

        # 4. TABLE: sections
        # We generate an embedding for the main section text for search
        sec_text = hierarchy.get('content', '')
        sec_emb = create_embedding_vector(sec_text)
        cur.execute("""INSERT INTO sections (chap_id, sec_no, title, content, embedding, document_id) 
                       VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
                    (chap_id, hierarchy.get('sec_no'), hierarchy.get('sec_title'), sec_text, sec_emb, doc_id))
        sec_id = cur.fetchone()[0]

        # 5. TABLE: sub_sections (The Children)
        for sub in hierarchy.get('sub_sections', []):
            sub_emb = create_embedding_vector(sub.get('text', ''))
            cur.execute("INSERT INTO sub_sections (section_id, sub_sec, text_content, embedding) VALUES (%s, %s, %s, %s) RETURNING id",
                        (sec_id, sub.get('sub_no'), sub.get('text'), sub_emb))
            sub_id = cur.fetchone()[0]

            # 6. TABLE: explanations (linked to Subsection)
            for exp_text in sub.get('explanations', []):
                cur.execute("INSERT INTO explanations (section_id, subsec_id, text_content) VALUES (%s, %s, %s)",
                            (sec_id, sub_id, exp_text))

        # 7. TABLE: illustrations (linked to Section)
        for illus in hierarchy.get('illustrations', []):
            cur.execute("INSERT INTO illustrations (section_id, text_content) VALUES (%s, %s)",
                        (sec_id, illus))

        # 8. TABLE: compounding_rules
        if hierarchy.get('compounding'):
            rule = hierarchy['compounding']
            cur.execute("INSERT INTO compounding_rules (bns_section_ref, punishment_fine, can_compound) VALUES (%s, %s, %s)",
                        (sec_id, rule.get('fine'), rule.get('eligible')))

        # 9. TABLE: first_schedule
        # These are usually large tables or lists at the end of the law
        for schedule in hierarchy.get('schedules', []):
            cur.execute("""
                INSERT INTO first_schedule (law_id, schedule_no, content) 
                VALUES (%s, %s, %s)
            """, (law_id, schedule.get('no'), schedule.get('content')))

        # 10. TABLE: form_templates
        # These are the official application/warrant forms
        for form in hierarchy.get('forms', []):
            cur.execute("""
                INSERT INTO form_templates (id, form_no, title, content_html) 
                VALUES (%s, %s, %s, %s)
            """, (id, form.get('form_no'), form.get('title'), form.get('content_html')))

        conn.commit()
        return jsonify({"message": "Full Law Data (including Schedules and Forms) Injected!"}), 201
        
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# --- NEW ROUTE: SMART UPDATE (Saves Edited Comparison Data) ---
@app.route('/api/admin/smart_update', methods=['POST'])
@token_required
def smart_update_law(current_user):
    data = request.get_json()
    # The 'hierarchy' is the structured data tree we got from the AI
    hierarchy = data.get('hierarchy') 
    raw_text = data.get('raw_text')
    law_code = data.get('law_code', 'BNSS')

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    try:
        # 1. TABLE: uploaded_documents (The Source)
        cur.execute("""
            INSERT INTO uploaded_documents (file_name, raw_pdf_data) 
            VALUES (%s, %s) RETURNING document_id
        """, (f"Update_{law_code}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}", raw_text))
        doc_id = cur.fetchone()[0]

        # 2. TABLE: legal_sources (The Root)
        
        # 2. TABLE: legal_sources (The Root)
        cur.execute("""
        INSERT INTO legal_sources (short_code, full_name, year) 
        VALUES (%s, %s, %s) 
        ON CONFLICT (short_code) DO UPDATE SET full_name = EXCLUDED.full_name, year = EXCLUDED.year
         RETURNING id
        """, (law_code, hierarchy.get('law_name', 'Law Name'), hierarchy.get('year', 2024))) # Added hierarchy.get('year')
        law_id = cur.fetchone()[0]

        # 3. TABLE: chapters (The Folder)
        cur.execute("""
            INSERT INTO chapters (law_id, chap_no, title) 
            VALUES (%s, %s, %s) RETURNING id
        """, (law_id, hierarchy.get('chap_no'), hierarchy.get('chap_title')))
        chap_id = cur.fetchone()[0]

        # 4. LOOP THROUGH SECTIONS (The Documents)
        for sec in hierarchy.get('sections', []):
            
            # ARCHIVE LOGIC: Deactivate old versions of this section if they exist
            cur.execute("UPDATE sections SET is_active = FALSE WHERE sec_no = %s AND chap_id = %s", 
                        (sec['sec_no'], chap_id))

            # EMBEDDING: Create vector for the section heading/content
            sec_search_text = f"Section {sec['sec_no']}: {sec['title']}. {sec['content']}"
            sec_emb = create_embedding_vector(sec_search_text)

            # 5. TABLE: sections
            cur.execute("""
                INSERT INTO sections (chap_id, sec_no, title, content, embedding, document_id, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, TRUE) RETURNING id
            """, (chap_id, sec['sec_no'], sec['title'], sec['content'], sec_emb, doc_id))
            sec_id = cur.fetchone()[0]

            # 6. TABLE: sub_sections (The Paragraphs)
            for sub in sec.get('sub_sections', []):
                sub_emb = create_embedding_vector(sub['text_content'])
                cur.execute("""
                    INSERT INTO sub_sections (section_id, sub_sec, text_content, embedding)
                    VALUES (%s, %s, %s, %s) RETURNING id
                """, (sec_id, sub['sub_no'], sub['text_content'], sub_emb))
                subsec_id = cur.fetchone()[0]

                # 7. TABLE: explanations (Linked to Subsection if specific, else Section)
                for exp in sub.get('explanations', []):
                    cur.execute("""
                        INSERT INTO explanations (section_id, subsec_id, text_content)
                        VALUES (%s, %s, %s)
                    """, (sec_id, subsec_id, exp))

            # 8. TABLE: illustrations (Linked to Section)
            for illus in sec.get('illustrations', []):
                cur.execute("INSERT INTO illustrations (section_id, text_content) VALUES (%s, %s)",
                            (sec_id, illus))

            # 9. TABLE: compounding_rules
            if sec.get('compounding'):
                c = sec['compounding']
                cur.execute("""
                    INSERT INTO compounding_rules (bns_section_ref, punishment_fine, can_compound) 
                    VALUES (%s, %s, %s)
                """, (sec_id, c.get('fine'), c.get('eligible')))

        # 10. TABLES: first_schedule & form_templates (Top-level law attachments)
        if hierarchy.get('schedules'):
            for sch in hierarchy['schedules']:
                cur.execute("INSERT INTO first_schedule (law_id, schedule_no, content) VALUES (%s, %s, %s)",
                            (law_id, sch.get('no'), sch.get('text')))

        if hierarchy.get('forms'):
            for form in hierarchy['forms']:
                cur.execute("INSERT INTO form_templates (id, form_no, title, content_html) VALUES (%s, %s, %s, %s)",
                            (law_id, form.get('no'), form.get('title'), form.get('path')))

        conn.commit()
        return jsonify({"message": "Full Law Hierarchy Injected Successfully!"}), 200

    except Exception as e:
        conn.rollback()
        return jsonify({"message": "Database Update Failed", "error": str(e)}), 500
    finally:
        conn.close()


# --- UNIVERSAL CRUD (YOUR FIXED VERSION FOR ARRAYS/DECIMALS) ---
TABLE_CONFIG = {
    'legal_sources': 'id',
    'chapters': 'id',
    'sections': 'id',
    'sub_sections': 'id',
    'explanations': 'id',
    'illustrations': 'id',
    'compounding_rules': 'id',
    'first_schedule': 'id',
    'form_templates': 'id',
    'uploaded_documents': 'document_id'
}


@app.route('/api/admin/universal/<table_name>', methods=['GET', 'POST'])
@token_required
def universal_crud(current_user, table_name):
    if table_name not in TABLE_CONFIG: return jsonify({"message": "Invalid table"}), 400
    
    pk = TABLE_CONFIG[table_name]
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # --- GET: View Data ---
    if request.method == 'GET':
        try:
            # We try to show only Active records first (for versioned tables)
            cur.execute(f"SELECT * FROM {table_name} WHERE is_active = TRUE ORDER BY {pk} DESC LIMIT 500")
        except:
            # If the table doesn't have 'is_active' (like uploaded_documents), show everything
            conn.rollback()
            cur.execute(f"SELECT * FROM {table_name} ORDER BY {pk} DESC LIMIT 500")

        rows = [dict(r) for r in cur.fetchall()]
        return jsonify(rows)

    # --- POST: Create or Update Data ---
    if request.method == 'POST':
        data = request.get_json()

        # 1. VERSIONING LOOKUP MAP (Added your new tables here!)
        # This tells the code what "name" to look for to see if a row already exists
        lookup_map = {
            'legal_sources': 'short_code',
            'chapters': 'chap_no',
            'sections': 'sec_no',
            'sub_sections': 'sub_sec',   
        }
        
        lookup_col = lookup_map.get(table_name)
        
        # 2. ARCHIVE OLD VERSION (The Versioning Logic)
        if lookup_col and lookup_col in data:
            try:
                # Check if this item (e.g., Section 105) already exists and is active
                cur.execute(
                    f"SELECT {pk}, version_number FROM {table_name} WHERE {lookup_col} = %s AND is_active = TRUE",
                    (data[lookup_col],))
                existing = cur.fetchone()
                
                if existing:
                    # Mark the old one as inactive (Archive it)
                    cur.execute(f"UPDATE {table_name} SET is_active = FALSE WHERE {pk} = %s", (existing[0],))
                    data['version_number'] = (existing[1] or 1) + 1
                else:
                    data['version_number'] = 1
                
                data['is_active'] = True
            except:
                conn.rollback() # Table might not have versioning columns, that's okay

        # 3. DYNAMIC INSERT (The "Universal" Part)
        # We automatically build the query based on whatever columns are in 'data'
        cols = [k for k in data.keys() if k != pk]
        query = f"INSERT INTO {table_name} ({','.join(cols)}) VALUES ({','.join(['%s'] * len(cols))}) RETURNING {pk}"
        
        cur.execute(query, [data[k] for k in cols])
        new_id = cur.fetchone()[0]
        
        # 4. AUDIT LOG (Security)
        log_event(conn, table_name, new_id, 'CREATE', None, data, current_user)
        
        conn.commit()
        return jsonify({"message": "Successfully updated the database", "id": new_id}), 201

@app.route('/api/admin/universal/<table_name>/<int:record_id>', methods=['PUT', 'DELETE'])
@token_required
def universal_item_ops(current_user, table_name, record_id):
    # Check if the table is allowed
    if table_name not in TABLE_CONFIG: return jsonify({"message": "Invalid table"}), 400
    
    pk = TABLE_CONFIG.get(table_name)
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # 1. Fetch the OLD data (for logging/history)
    cur.execute(f"SELECT * FROM {table_name} WHERE {pk}=%s", (record_id,))
    old_record = cur.fetchone()
    if not old_record:
        return jsonify({"message": "Record not found"}), 404

    # --- MODE: UPDATE (PUT) ---
    if request.method == 'PUT':
        data = request.get_json()

        # Build the SQL UPDATE command dynamically
        # It creates a string like: "title=%s, content=%s"
        set_clause = ", ".join([f"{k}=%s" for k in data.keys() if k != pk])
        
        # We add the record_id at the end of the list for the WHERE clause
        values = [data[k] for k in data.keys() if k != pk] + [record_id]
        
        cur.execute(f"UPDATE {table_name} SET {set_clause} WHERE {pk}=%s", values)
        
        # Log the change
        log_event(conn, table_name, record_id, 'UPDATE', dict(old_record), data, current_user)
        conn.commit()
        return jsonify({"msg": "Successfully updated!"})

    # --- MODE: DELETE ---
    if request.method == 'DELETE':
        try:
            # Try a "Soft Delete" first (Hiding the data)
            cur.execute(f"UPDATE {table_name} SET is_active = FALSE WHERE {pk}=%s", (record_id,))
        except:
            # If no is_active column exists, do a "Hard Delete" (Erasing the data)
            conn.rollback() # Reset from the failed 'is_active' attempt
            cur.execute(f"DELETE FROM {table_name} WHERE {pk}=%s", (record_id,))
        
        log_event(conn, table_name, record_id, 'DELETE', dict(old_record), None, current_user)
        conn.commit()
        return jsonify({"msg": "Successfully deleted!"})

    conn.close()

# --- HISTORY VIEW ---
@app.route('/api/admin/versions/<table_name>', methods=['POST'])
@token_required
def view_versions(current_user, table_name):
    data = request.get_json()
    
    # Cleaned for BNSS project
    lookup_map = {
        'legal_sources': 'short_code',
        'chapters': 'chap_no',
        'sections': 'sec_no',
        'sub_sections': 'sub_sec'
    }
    
    col = lookup_map.get(table_name)
    val = data.get('identifier') # e.g., "105" or "BNSS"
    
    if not col:
        return jsonify({"message": "Versioning not supported for this table"}), 400

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Fetch all versions, newest first
    cur.execute(f"SELECT * FROM {table_name} WHERE {col} = %s ORDER BY version_number DESC", (val,))
    
    history = [dict(r) for r in cur.fetchall()]
    conn.close()
    
    return jsonify({"history": history}), 200


if __name__ == '__main__':
    app.run(debug=True)