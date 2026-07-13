"""Add the deterministic node-degree lookup required by the PrimeKG path scorer."""
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "indexes/primekg.sqlite3"

with sqlite3.connect(DATABASE) as connection:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='node_degrees'").fetchone()
    if not exists:
        connection.executescript("""
            CREATE TABLE node_degrees (id INTEGER PRIMARY KEY, degree INTEGER NOT NULL);
            INSERT INTO node_degrees
            SELECT n.id,
                   (SELECT count(*) FROM edges WHERE x=n.id) +
                   (SELECT count(*) FROM edges WHERE y=n.id)
            FROM nodes n;
            ANALYZE node_degrees;
        """)
    count = connection.execute("SELECT count(*) FROM node_degrees").fetchone()[0]
print(f"node_degrees={count}")
