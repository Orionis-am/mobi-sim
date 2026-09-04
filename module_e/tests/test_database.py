from __future__ import annotations

from module_e import database


class TestSeedServicesIfEmpty:
    def test_seeds_from_catalog_json(self) -> None:
        with database.SessionLocal() as db:
            database.Base.metadata.create_all(bind=database.engine)
            inserted = database.seed_services_if_empty(db)
            assert inserted == 20
            assert db.query(database.Service).count() == 20

    def test_idempotent_when_already_seeded(self) -> None:
        with database.SessionLocal() as db:
            database.Base.metadata.create_all(bind=database.engine)
            database.seed_services_if_empty(db)
            second_pass = database.seed_services_if_empty(db)
            assert second_pass == 0
            assert db.query(database.Service).count() == 20


class TestInitDb:
    def test_creates_tables_and_seeds(self) -> None:
        database.init_db()
        with database.SessionLocal() as db:
            assert db.query(database.Service).count() == 20
