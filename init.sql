-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create uploaded_documents table
CREATE TABLE IF NOT EXISTS uploaded_documents (
    document_id SERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,
    raw_pdf_data TEXT,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create cyber_laws table with versioning
CREATE TABLE IF NOT EXISTS cyber_laws (
    law_section_id SERIAL PRIMARY KEY,
    chapter TEXT,
    section TEXT NOT NULL,
    section_name TEXT,
    description TEXT,
    punishment TEXT,
    version_number INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    document_id INTEGER REFERENCES uploaded_documents(document_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(section, version_number)
);

-- Create law_embeddings table
CREATE TABLE IF NOT EXISTS law_embeddings (
    embedding_id SERIAL PRIMARY KEY,
    law_section_id INTEGER REFERENCES cyber_laws(law_section_id),
    section_text TEXT,
    embedding vector(384),  -- For all-MiniLM-L6-v2 model
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create scam_advisories table
CREATE TABLE IF NOT EXISTS scam_advisories (
    scam_id SERIAL PRIMARY KEY,
    scam_name TEXT NOT NULL,
    keywords TEXT[],
    modus_operandi TEXT,
    recent_intel TEXT,
    police_advice TEXT,
    version_number INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create cybercells table
CREATE TABLE IF NOT EXISTS cybercells (
    station_id SERIAL PRIMARY KEY,
    station_name TEXT NOT NULL,
    city TEXT,
    district TEXT,
    phone_number TEXT,
    email TEXT,
    website TEXT,
    services TEXT[],
    version_number INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create legal_guidance table
CREATE TABLE IF NOT EXISTS legal_guidance (
    crime_id SERIAL PRIMARY KEY,
    crime_type TEXT NOT NULL,
    applicable_laws TEXT[],
    punishment TEXT,
    jurisdiction TEXT,
    version_number INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create reporting_procedures table
CREATE TABLE IF NOT EXISTS reporting_procedures (
    report_id SERIAL PRIMARY KEY,
    crime_type TEXT NOT NULL,
    priority TEXT,
    procedure_text TEXT,
    contact_method TEXT,
    contact_detail TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create user_queries table
CREATE TABLE IF NOT EXISTS user_queries (
    query_id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    category TEXT,
    law_section TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create audit_logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id SERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    record_id INTEGER,
    action_type TEXT NOT NULL,
    old_data JSONB,
    new_data JSONB,
    changed_by TEXT,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_law_embeddings ON law_embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_cyber_laws_section ON cyber_laws(section);
CREATE INDEX IF NOT EXISTS idx_cyber_laws_active ON cyber_laws(is_active);
CREATE INDEX IF NOT EXISTS idx_audit_logs_table ON audit_logs(table_name);
CREATE INDEX IF NOT EXISTS idx_audit_logs_time ON audit_logs(changed_at);

-- Create text search indexes
CREATE INDEX IF NOT EXISTS idx_cyber_laws_search ON cyber_laws USING GIN (to_tsvector('english', description));
CREATE INDEX IF NOT EXISTS idx_scam_advisories_search ON scam_advisories USING GIN (to_tsvector('english', modus_operandi || ' ' || police_advice));