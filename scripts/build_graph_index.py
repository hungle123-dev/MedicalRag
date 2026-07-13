import csv
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "primekg"
OUTPUT = ROOT / "indexes" / "primekg.sqlite3"


def batches(rows, size=50_000):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp.sqlite3")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    connection.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA temp_store=MEMORY;
        CREATE TABLE nodes (id INTEGER PRIMARY KEY, external_id TEXT, type TEXT, name TEXT, name_norm TEXT, source TEXT);
        CREATE TABLE edges (relation TEXT, display_relation TEXT, x INTEGER, y INTEGER);
    """)
    with (RAW / "nodes.tab").open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        for batch in batches(rows):
            connection.executemany(
                "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?)",
                [(int(r["node_index"]), r["node_id"], r["node_type"], r["node_name"], r["node_name"].casefold(), r["node_source"]) for r in batch],
            )
    with (RAW / "edges.csv").open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        count = 0
        for batch in batches(rows):
            connection.executemany(
                "INSERT INTO edges VALUES (?, ?, ?, ?)",
                [(r["relation"], r["display_relation"], int(r["x_index"]), int(r["y_index"])) for r in batch],
            )
            count += len(batch)
            if count % 1_000_000 == 0:
                print(f"loaded {count:,} edges")
    connection.executescript("""
        CREATE INDEX nodes_name_norm ON nodes(name_norm);
        CREATE INDEX edges_x ON edges(x);
        CREATE INDEX edges_y ON edges(y);
        CREATE INDEX edges_xy ON edges(x, y);
        CREATE TABLE node_degrees (id INTEGER PRIMARY KEY, degree INTEGER NOT NULL);
        INSERT INTO node_degrees
        SELECT n.id,
               (SELECT count(*) FROM edges WHERE x=n.id) +
               (SELECT count(*) FROM edges WHERE y=n.id)
        FROM nodes n;
        ANALYZE;
    """)
    connection.commit()
    connection.close()
    temporary.replace(OUTPUT)
    print(f"wrote {OUTPUT} with {count:,} edges")


if __name__ == "__main__":
    main()
