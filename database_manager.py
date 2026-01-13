import sqlite3
import datetime
import os

DB_NAME = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create chats table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        timestamp DATETIME,
        role TEXT,
        content TEXT,
        visual_context TEXT,
        document_context TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

def save_message(session_id, role, content, visual_context="", document_context=""):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    timestamp = datetime.datetime.now()
    
    cursor.execute('''
    INSERT INTO chats (session_id, timestamp, role, content, visual_context, document_context)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (session_id, timestamp, role, content, visual_context, document_context))
    
    conn.commit()
    conn.close()

def get_session_history(session_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT role, content FROM chats WHERE session_id = ? ORDER BY timestamp ASC
    ''', (session_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return rows

def get_all_sessions():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT DISTINCT session_id, MIN(timestamp) as start_time FROM chats GROUP BY session_id ORDER BY start_time DESC
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    return rows
