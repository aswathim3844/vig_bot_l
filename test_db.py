import os
import psycopg2
from dotenv import load_dotenv

# Load the settings from your .env file
load_dotenv()

try:
    # Attempt to connect using the variables from .env
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS")
    )
    print("✅ Success! VL is connected to the database.")
    
    # Check if our new columns exist
    cur = conn.cursor()
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='compounding_rules';")
    columns = [row[0] for row in cur.fetchall()]
    print(f"Found columns: {columns}")
    
    cur.close()
    conn.close()
except Exception as e:
    print(f"❌ Connection Failed: {e}")