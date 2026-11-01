from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .classes import Bin, WasteClass
from .config import Jurisdiction

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            INTEGER NOT NULL,
    predicted     TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    weight_g      REAL,
    bin           TEXT    NOT NULL,
    gated         INTEGER NOT NULL,
    image_path    TEXT,
    ground_truth  TEXT,
    mech_ok       INTEGER NOT NULL DEFAULT 1,
    model_version TEXT,
    jurisdiction  TEXT
);

CREATE INDEX IF NOT EXISTS events_ts ON events (ts);
CREATE INDEX IF NOT EXISTS events_unlabelled ON events (ground_truth) WHERE ground_truth IS NULL;
"""


@dataclass(frozen=True)
class Event:
    ts: int
    predicted: WasteClass
    confidence: float
    bin: Bin
    gated: bool
    weight_g: float | None = None
    image_path: str | None = None
    ground_truth: WasteClass | None = None
    mech_ok: bool = True
    model_version: str | None = None
    jurisdiction: str | None = None


class EventStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def append(self, event: Event) -> int:
        cursor = self._connection.execute(
            """
            INSERT INTO events (
                ts, predicted, confidence, weight_g, bin, gated,
                image_path, ground_truth, mech_ok, model_version, jurisdiction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.ts,
                str(event.predicted),
                event.confidence,
                event.weight_g,
                str(event.bin),
                int(event.gated),
                event.image_path,
                str(event.ground_truth) if event.ground_truth else None,
                int(event.mech_ok),
                event.model_version,
                event.jurisdiction,
            ),
        )
        self._connection.commit()
        return int(cursor.lastrowid)

    def recent(self, limit: int = 100, since: int | None = None) -> list[sqlite3.Row]:
        if since is None:
            return self._connection.execute(
                "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return self._connection.execute(
            "SELECT * FROM events WHERE ts >= ? ORDER BY ts DESC LIMIT ?", (since, limit)
        ).fetchall()

    def counts_by_class(self, since: int | None = None) -> dict[str, int]:
        rows = self._connection.execute(
            "SELECT predicted, COUNT(*) AS n FROM events WHERE ts >= ? GROUP BY predicted",
            (since or 0,),
        ).fetchall()
        return {row["predicted"]: row["n"] for row in rows}

    def counts_by_bin(self, since: int | None = None) -> dict[str, int]:
        rows = self._connection.execute(
            "SELECT bin, COUNT(*) AS n FROM events WHERE ts >= ? GROUP BY bin",
            (since or 0,),
        ).fetchall()
        return {row["bin"]: row["n"] for row in rows}

    def gate_rate(self, since: int | None = None) -> float | None:
        row = self._connection.execute(
            "SELECT COUNT(*) AS total, SUM(gated) AS gated FROM events WHERE ts >= ?",
            (since or 0,),
        ).fetchone()
        if not row["total"]:
            return None
        return (row["gated"] or 0) / row["total"]

    def jam_count(self, since: int | None = None) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS n FROM events WHERE ts >= ? AND mech_ok = 0",
            (since or 0,),
        ).fetchone()
        return int(row["n"])

    def contamination_rate(
        self, jurisdiction: Jurisdiction, since: int | None = None
    ) -> float | None:
        rows = self._connection.execute(
            """
            SELECT ground_truth FROM events
            WHERE ts >= ? AND bin = ? AND ground_truth IS NOT NULL
            """,
            (since or 0, str(Bin.RECYCLE)),
        ).fetchall()

        if not rows:
            return None

        contaminated = sum(
            1 for row in rows if jurisdiction.bin_for(WasteClass(row["ground_truth"])) is Bin.TRASH
        )
        return contaminated / len(rows)

    def unlabelled(self, limit: int = 100) -> list[sqlite3.Row]:
        return self._connection.execute(
            """
            SELECT * FROM events
            WHERE ground_truth IS NULL AND image_path IS NOT NULL
            ORDER BY ts DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()

    def set_ground_truth(self, event_id: int, waste_class: WasteClass) -> None:
        self._connection.execute(
            "UPDATE events SET ground_truth = ? WHERE id = ?", (str(waste_class), event_id)
        )
        self._connection.commit()
