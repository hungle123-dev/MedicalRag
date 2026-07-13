from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import math
from contextlib import closing
from pathlib import Path


WORD = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")


class PrimeKGIndex:
    def __init__(self, database: Path):
        self.database = database

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.database.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def link(self, question: str, limit: int = 5) -> list[dict]:
        words = WORD.findall(question.casefold())
        phrases = {" ".join(words[start:end]) for start in range(len(words)) for end in range(start + 1, min(len(words), start + 6) + 1)}
        phrases = {phrase for phrase in phrases if len(phrase) >= 3}
        if not phrases:
            return []
        placeholders = ",".join("?" for _ in phrases)
        with closing(self.connect()) as connection:
            rows = connection.execute(
                f"SELECT id, external_id, type, name, source FROM nodes WHERE name_norm IN ({placeholders})",
                tuple(phrases),
            ).fetchall()
        cues = {
            "gene/protein": {"gene", "genes", "protein", "proteins"},
            "drug": {"drug", "drugs", "medication", "treatment", "treat"},
            "disease": {"disease", "disorder", "syndrome", "condition"},
            "effect/phenotype": {"symptom", "phenotype", "effect"},
        }
        word_set = set(words)
        maximal = [row for row in rows if not any(
            row["name"].casefold() != other["name"].casefold()
            and row["name"].casefold() in other["name"].casefold()
            for other in rows
        )]
        ranked = sorted(
            maximal,
            key=lambda row: (
                -int(bool(word_set & cues.get(row["type"], set()))),
                -len(row["name"]), row["name"], row["id"],
            ),
        )[:limit]
        return [dict(row) | {"confidence": 1.0, "method": "exact_primekg_name"} for row in ranked]

    def paths(self, seeds: list[dict], limit: int = 5, question: str | None = None, max_hops: int = 2) -> list[dict]:
        if not seeds:
            return []
        question = question or " ".join(seed["name"] for seed in seeds)
        paths = []
        with closing(self.connect()) as connection:
            for seed in seeds:
                rows = connection.execute(
                    """SELECT e.display_relation relation, x.id x_id, x.name x_name, x.type x_type,
                              y.id y_id, y.name y_name, y.type y_type
                       FROM edges e JOIN nodes x ON x.id=e.x JOIN nodes y ON y.id=e.y
                       WHERE e.x=? OR e.y=? LIMIT 200""",
                    (seed["id"], seed["id"]),
                ).fetchall()
                for row in rows:
                    canonical = {
                        "nodes": [{"id": str(row["x_id"]), "name": row["x_name"], "type": row["x_type"]},
                                  {"id": str(row["y_id"]), "name": row["y_name"], "type": row["y_type"]}],
                        "edges": [{"source_id": str(row["x_id"]), "relation": row["relation"],
                                   "target_id": str(row["y_id"]), "source_dataset": "PrimeKG"}],
                    }
                    identifier = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
                    endpoint = row["y_name"] if row["x_id"] == seed["id"] else row["x_name"]
                    endpoint_overlap = len(set(WORD.findall(endpoint.casefold())) & set(WORD.findall(question.casefold())))
                    relation_overlap = len(set(WORD.findall(row["relation"].casefold())) & set(WORD.findall(question.casefold())))
                    endpoint_id = row["y_id"] if row["x_id"] == seed["id"] else row["x_id"]
                    degree = connection.execute(
                        "SELECT (SELECT count(*) FROM edges WHERE x=?) + (SELECT count(*) FROM edges WHERE y=?)",
                        (endpoint_id, endpoint_id),
                    ).fetchone()[0]
                    score = seed["confidence"] + 0.5 * endpoint_overlap + 0.25 * relation_overlap - 0.05 * math.log1p(degree)
                    paths.append(canonical | {"id": f"primekg:path:{identifier}", "score": round(score, 6),
                                               "hop_count": 1, "provenance_valid": True})
            if len(seeds) > 1:
                for left in seeds:
                    for right in seeds:
                        if left["id"] == right["id"]:
                            continue
                        rows = connection.execute(
                            """SELECT e1.display_relation r1, e2.display_relation r2,
                                      a.id a_id,a.name a_name,a.type a_type,
                                      b.id b_id,b.name b_name,b.type b_type,
                                      c.id c_id,c.name c_name,c.type c_type
                               FROM edges e1 JOIN edges e2 ON e1.y=e2.x
                               JOIN nodes a ON a.id=e1.x JOIN nodes b ON b.id=e1.y JOIN nodes c ON c.id=e2.y
                               WHERE e1.x=? AND e2.y=? LIMIT 10""",
                            (left["id"], right["id"]),
                        ).fetchall()
                        for row in rows:
                            canonical = {
                                "nodes": [{"id": str(row["a_id"]), "name": row["a_name"], "type": row["a_type"]},
                                          {"id": str(row["b_id"]), "name": row["b_name"], "type": row["b_type"]},
                                          {"id": str(row["c_id"]), "name": row["c_name"], "type": row["c_type"]}],
                                "edges": [{"source_id": str(row["a_id"]), "relation": row["r1"], "target_id": str(row["b_id"]), "source_dataset": "PrimeKG"},
                                          {"source_id": str(row["b_id"]), "relation": row["r2"], "target_id": str(row["c_id"]), "source_dataset": "PrimeKG"}],
                            }
                            identifier = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
                            paths.append(canonical | {"id": f"primekg:path:{identifier}", "score": 1.25,
                                                       "hop_count": 2, "provenance_valid": True})
                        if max_hops >= 3:
                            rows = connection.execute(
                                """SELECT e1.display_relation r1,e2.display_relation r2,e3.display_relation r3,
                                          a.id a_id,a.name a_name,a.type a_type,b.id b_id,b.name b_name,b.type b_type,
                                          c.id c_id,c.name c_name,c.type c_type,d.id d_id,d.name d_name,d.type d_type
                                   FROM edges e1 JOIN edges e2 ON e1.y=e2.x JOIN edges e3 ON e2.y=e3.x
                                   JOIN nodes a ON a.id=e1.x JOIN nodes b ON b.id=e1.y
                                   JOIN nodes c ON c.id=e2.y JOIN nodes d ON d.id=e3.y
                                   WHERE e1.x=? AND e3.y=? AND a.id<>b.id AND b.id<>c.id AND c.id<>d.id
                                   LIMIT 10""", (left["id"], right["id"]),
                            ).fetchall()
                            for row in rows:
                                canonical = {
                                    "nodes": [{"id": str(row[f"{key}_id"]), "name": row[f"{key}_name"], "type": row[f"{key}_type"]}
                                              for key in ("a", "b", "c", "d")],
                                    "edges": [
                                        {"source_id": str(row["a_id"]), "relation": row["r1"], "target_id": str(row["b_id"]), "source_dataset": "PrimeKG"},
                                        {"source_id": str(row["b_id"]), "relation": row["r2"], "target_id": str(row["c_id"]), "source_dataset": "PrimeKG"},
                                        {"source_id": str(row["c_id"]), "relation": row["r3"], "target_id": str(row["d_id"]), "source_dataset": "PrimeKG"}],
                                }
                                identifier = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
                                paths.append(canonical | {"id": f"primekg:path:{identifier}", "score": 1.0,
                                                           "hop_count": 3, "provenance_valid": True})
        unique = {path["id"]: path for path in paths}
        return sorted(unique.values(), key=lambda path: (-path["score"], path["id"]))[:limit]
