"""
Chat history in a local SQLite database.

Every call opens its own connection, so the functions can be used from the
GUI thread and from worker threads.
"""
import sqlite3
import datetime
from contextlib import closing

DB_NAME = "chat_history.db"


def _connect():
    return closing(sqlite3.connect(DB_NAME, timeout=10))


def init_db():
    with _connect() as conn, conn:
        conn.execute('''
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
        columns = [row[1] for row in conn.execute('PRAGMA table_info(chats)')]
        if 'reply_to' not in columns:
            # id of the user message an assistant message answers (added later; NULL for old rows)
            conn.execute('ALTER TABLE chats ADD COLUMN reply_to INTEGER')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_chats_session ON chats(session_id, id)')


# Conversation order: each answer directly after the question it answers,
# even if further questions were sent before the answer arrived.
_CONVERSATION_ORDER = 'COALESCE(reply_to, id), id'
_CONVERSATION_ORDER_REVERSED = 'COALESCE(reply_to, id) DESC, id DESC'


def save_message(session_id, role, content, visual_context="", document_context="", reply_to=None):
    """Store a message and return its id. For answers, `reply_to` is the id of the question."""
    # str() keeps the same format the sqlite3 default adapter used before
    timestamp = str(datetime.datetime.now())
    with _connect() as conn, conn:
        cursor = conn.execute('''
        INSERT INTO chats (session_id, timestamp, role, content, visual_context, document_context, reply_to)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, timestamp, role, content, visual_context, document_context, reply_to))
        return cursor.lastrowid


def get_session_history(session_id):
    """All messages of a session in order: [(role, content), ...]."""
    with _connect() as conn:
        return conn.execute(
            f'SELECT role, content FROM chats WHERE session_id = ? ORDER BY {_CONVERSATION_ORDER}',
            (session_id,)).fetchall()


def get_context_history(session_id, current_message_id, max_messages=12):
    """
    Conversation so far, as context for answering the user message
    `current_message_id`: [(role, content), ...], oldest first.

    Excludes the current message and user messages sent after it (they are
    still waiting in the queue), but includes all answers already given.
    """
    with _connect() as conn:
        rows = conn.execute(f'''
        SELECT role, content FROM chats
        WHERE session_id = ? AND id != ? AND (role = 'assistant' OR id < ?)
        ORDER BY {_CONVERSATION_ORDER_REVERSED} LIMIT ?
        ''', (session_id, current_message_id, current_message_id, max_messages)).fetchall()
    return list(reversed(rows))


def get_all_sessions():
    """[(session_id, start_time, title), ...], most recently active first.
    The title is the first user message of the session."""
    with _connect() as conn:
        return conn.execute('''
        SELECT c.session_id,
               MIN(c.timestamp) AS start_time,
               (SELECT content FROM chats u
                 WHERE u.session_id = c.session_id AND u.role = 'user'
                 ORDER BY u.id LIMIT 1) AS title
        FROM chats c
        GROUP BY c.session_id
        ORDER BY MAX(c.id) DESC
        ''').fetchall()


def delete_session(session_id):
    with _connect() as conn, conn:
        conn.execute('DELETE FROM chats WHERE session_id = ?', (session_id,))
