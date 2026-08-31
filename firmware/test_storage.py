import storage


def test_totals_add_up(tmp_path):
    db = storage.open_db(tmp_path / "events.db")
    storage.log_event(db, "pet_bottle", 0.9, 20.0, "recycle", "rule")
    storage.log_event(db, "organics", 0.9, 50.0, "trash", "rule")
    storage.log_event(db, "soiled", 0.9, 60.0, "trash", "rule")

    counts = storage.totals(db)
    assert counts["total"] == 3
    assert counts["recycle"] == 1
    assert counts["trash"] == 2


def test_diverted_fraction(tmp_path):
    db = storage.open_db(tmp_path / "events.db")
    storage.log_event(db, "pet_bottle", 0.9, 20.0, "recycle", "rule")
    storage.log_event(db, "organics", 0.9, 50.0, "trash", "rule")
    storage.log_event(db, "soiled", 0.9, 60.0, "trash", "rule")

    assert storage.diverted_fraction(db) == 0.3333


def test_diverted_fraction_is_zero_when_empty(tmp_path):
    db = storage.open_db(tmp_path / "events.db")
    assert storage.diverted_fraction(db) == 0.0


def test_recent_returns_newest_first(tmp_path):
    db = storage.open_db(tmp_path / "events.db")
    storage.log_event(db, "pet_bottle", 0.9, 20.0, "recycle", "rule")
    storage.log_event(db, "organics", 0.8, 50.0, "trash", "rule")

    events = storage.recent(db)
    assert len(events) == 2
    assert events[0]["label"] == "organics"


def test_recent_respects_the_limit(tmp_path):
    db = storage.open_db(tmp_path / "events.db")
    for _ in range(5):
        storage.log_event(db, "metal_can", 0.9, 15.0, "recycle", "rule")

    assert len(storage.recent(db, limit=3)) == 3
