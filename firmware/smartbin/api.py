from __future__ import annotations

import time

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import Jurisdiction
from .storage import EventStore

DAY_MS = 24 * 60 * 60 * 1000


def create_app(store: EventStore, jurisdiction: Jurisdiction, model_version: str) -> FastAPI:
    app = FastAPI(title="SmartBin", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://.*",
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    started_at = time.time()

    def window(days: int) -> int:
        return int(time.time() * 1000) - days * DAY_MS

    @app.get("/health")
    def health() -> dict:
        latest = store.recent(limit=1)
        return {
            "uptime_s": round(time.time() - started_at, 1),
            "jurisdiction": jurisdiction.name,
            "confidence_threshold": jurisdiction.confidence_threshold,
            "model_version": model_version,
            "last_event_ts": latest[0]["ts"] if latest else None,
        }

    @app.get("/events")
    def events(since: int | None = None, limit: int = Query(100, le=1000)) -> list[dict]:
        return [dict(row) for row in store.recent(limit=limit, since=since)]

    @app.get("/stats/counts")
    def counts(days: int = Query(7, ge=1, le=365)) -> dict:
        since = window(days)
        return {
            "by_class": store.counts_by_class(since),
            "by_bin": store.counts_by_bin(since),
        }

    @app.get("/stats/contamination")
    def contamination(days: int = Query(7, ge=1, le=365)) -> dict:
        since = window(days)
        return {
            "days": days,
            "rate": store.contamination_rate(jurisdiction, since),
        }

    @app.get("/stats/confidence")
    def confidence(days: int = Query(7, ge=1, le=365)) -> dict:
        since = window(days)
        rows = store.recent(limit=1000, since=since)
        return {
            "gate_rate": store.gate_rate(since),
            "values": [row["confidence"] for row in rows],
        }

    @app.get("/stats/jams")
    def jams(days: int = Query(7, ge=1, le=365)) -> dict:
        since = window(days)
        return {"days": days, "count": store.jam_count(since)}

    return app
