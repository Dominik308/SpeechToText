import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import database_manager as db  # noqa: E402


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_NAME", str(tmp_path / "test.db"))
    db.init_db()


def test_save_and_history_order():
    a = db.save_message("s1", "user", "Frage 1")
    b = db.save_message("s1", "assistant", "Antwort 1")
    assert b > a
    db.save_message("s2", "user", "andere Sitzung")
    assert db.get_session_history("s1") == [("user", "Frage 1"), ("assistant", "Antwort 1")]


def test_context_history_with_queued_questions():
    # Q1, Q2, Q3 were sent quickly; the answers arrive afterwards.
    q1 = db.save_message("s", "user", "Q1")
    q2 = db.save_message("s", "user", "Q2")
    q3 = db.save_message("s", "user", "Q3")
    # While answering Q1 nothing came before it
    assert db.get_context_history("s", q1) == []
    db.save_message("s", "assistant", "A1", reply_to=q1)
    # Answering Q2: sees Q1 and A1, but neither itself nor the still-queued Q3
    assert db.get_context_history("s", q2) == [("user", "Q1"), ("assistant", "A1")]
    db.save_message("s", "assistant", "A2", reply_to=q2)
    # Answers are placed directly after their question
    expected = [("user", "Q1"), ("assistant", "A1"), ("user", "Q2"), ("assistant", "A2")]
    assert db.get_context_history("s", q3) == expected
    db.save_message("s", "assistant", "A3", reply_to=q3)
    assert db.get_session_history("s") == expected + [("user", "Q3"), ("assistant", "A3")]


def test_old_rows_without_reply_to_keep_insertion_order():
    db.save_message("s", "user", "Q1")
    db.save_message("s", "assistant", "A1")
    db.save_message("s", "user", "Q2")
    assert db.get_session_history("s") == [("user", "Q1"), ("assistant", "A1"), ("user", "Q2")]


def test_init_db_migrates_old_schema(tmp_path, monkeypatch):
    import sqlite3
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE chats (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, timestamp DATETIME,"
                 " role TEXT, content TEXT, visual_context TEXT, document_context TEXT)")
    conn.execute("INSERT INTO chats (session_id, role, content) VALUES ('old', 'user', 'alt')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "DB_NAME", path)
    db.init_db()
    db.init_db()  # idempotent
    q = db.save_message("old", "user", "neu")
    db.save_message("old", "assistant", "Antwort", reply_to=q)
    assert db.get_session_history("old") == [("user", "alt"), ("user", "neu"), ("assistant", "Antwort")]


def test_context_history_limit_keeps_latest():
    for i in range(10):
        q = db.save_message("s", "user", f"Q{i}")
        db.save_message("s", "assistant", f"A{i}", reply_to=q)
    current = db.save_message("s", "user", "now")
    hist = db.get_context_history("s", current, max_messages=4)
    assert hist == [("user", "Q8"), ("assistant", "A8"), ("user", "Q9"), ("assistant", "A9")]


def test_sessions_listing_and_delete():
    db.save_message("old", "user", "Erste Frage")
    db.save_message("old", "assistant", "Antwort")
    db.save_message("new", "user", "Neue Frage")
    db.save_message("old", "user", "Nachfrage")  # 'old' is now the most recently active
    sessions = db.get_all_sessions()
    assert [(s[0], s[2]) for s in sessions] == [("old", "Erste Frage"), ("new", "Neue Frage")]
    db.delete_session("old")
    assert [s[0] for s in db.get_all_sessions()] == ["new"]
    assert db.get_session_history("old") == []
