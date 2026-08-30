import sqlite3
import time

SCHEMA = """
create table if not exists events (
    id integer primary key autoincrement,
    at real not null,
    label text not null,
    confidence real not null,
    weight_g real not null,
    bin text not null,
    reason text not null
)
"""


def open_db(path):
    db = sqlite3.connect(path, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute(SCHEMA)
    db.commit()
    return db


def log_event(db, label, confidence, weight_g, bin_name, reason):
    db.execute(
        "insert into events (at, label, confidence, weight_g, bin, reason) values (?, ?, ?, ?, ?, ?)",
        (time.time(), label, confidence, weight_g, bin_name, reason),
    )
    db.commit()


def recent(db, limit=50):
    rows = db.execute("select * from events order by id desc limit ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def totals(db):
    rows = db.execute("select bin, count(*) as n from events group by bin").fetchall()
    counts = {row["bin"]: row["n"] for row in rows}
    counts["total"] = sum(counts.values())
    return counts


def diverted_fraction(db):
    counts = totals(db)
    if not counts.get("total"):
        return 0.0
    return round(counts.get("recycle", 0) / counts["total"], 4)
