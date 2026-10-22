import os
import threading
from typing import Any

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session


def jdbc_to_sqlalchemy_url(url: str) -> str:
    """Convert jdbc:postgresql://host:5432/db to postgresql+psycopg://host:5432/db."""
    if url.startswith("jdbc:postgresql://"):
        return "postgresql+psycopg://" + url[len("jdbc:postgresql://"):]
    return url


def _blank(s: str | None) -> bool:
    return not (s and s.strip())


def _fields(obj: Any) -> dict[str, Any]:
    """Non-null column attributes of an entity instance."""
    return {
        c.key: getattr(obj, c.key)
        for c in obj.__mapper__.column_attrs
        if getattr(obj, c.key) is not None
    }


class PostgresDbClient:
    _instance: "PostgresDbClient | None" = None
    _lock = threading.Lock()

    def __init__(self, url: str, user: str | None = None, password: str | None = None, engine: Any = None):
        if engine is None:
            kwargs = {}
            if user is not None:
                kwargs["username"] = user
            if password is not None:
                kwargs["password"] = password
            from sqlalchemy.engine import make_url

            engine = create_engine(make_url(jdbc_to_sqlalchemy_url(url)).set(**kwargs))
        self.engine = engine

    @classmethod
    def from_environment(cls) -> "PostgresDbClient":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    url = os.getenv("POSTGRES_URL")
                    user = os.getenv("POSTGRES_USER")
                    password = os.getenv("POSTGRES_PASSWORD")
                    if _blank(url):
                        raise RuntimeError("POSTGRES_URL is not set")
                    if _blank(user):
                        raise RuntimeError("POSTGRES_USER is not set")
                    if password is None:
                        raise RuntimeError("POSTGRES_PASSWORD is not set")
                    cls._instance = cls(url, user, password)
        return cls._instance

    @staticmethod
    def _check_table(table_name: str, entity_cls: type) -> None:
        table = entity_cls.__table__
        names = {table.name}
        if table.schema:
            names.add(f"{table.schema}.{table.name}")
        if table_name not in names:
            raise ValueError(f"Table name '{table_name}' does not match entity {entity_cls.__name__}")

    def _run(self, fn):
        with Session(self.engine) as session:
            try:
                result = fn(session)
                session.commit()
                return result
            except Exception:
                session.rollback()
                raise

    def insert(self, table_name: str, entity: Any) -> Any:
        self._check_table(table_name, type(entity))

        def op(s: Session):
            s.add(entity)
            s.flush()
            s.refresh(entity)
            s.expunge(entity)
            return entity

        return self._run(op)

    def select(self, table_name: str, entity_cls: type, conditions: Any) -> list:
        self._check_table(table_name, entity_cls)
        if conditions is not None and not isinstance(conditions, entity_cls):
            raise ValueError("conditions must be an instance of entity_cls")
        where = _fields(conditions) if conditions is not None else {}

        def op(s: Session):
            q = s.query(entity_cls)
            for k, v in where.items():
                q = q.filter(getattr(entity_cls, k) == v)
            rows = q.all()
            s.expunge_all()
            return rows

        return self._run(op)

    def update(self, table_name: str, values: Any, conditions: Any) -> int:
        cls = type(values)
        if type(conditions) is not cls:
            raise ValueError("values and conditions must be the same entity class")
        self._check_table(table_name, cls)
        new, where = _fields(values), _fields(conditions)
        if not new:
            raise ValueError("values must have at least one non-null field")
        if not where:
            raise ValueError("conditions must have at least one non-null field")

        def op(s: Session):
            q = s.query(cls)
            for k, v in where.items():
                q = q.filter(getattr(cls, k) == v)
            return q.update(new, synchronize_session=False)

        return self._run(op)

    def delete(self, table_name: str, conditions: Any) -> int:
        cls = type(conditions)
        self._check_table(table_name, cls)
        where = _fields(conditions)
        if not where:
            raise ValueError("conditions must have at least one non-null field")

        def op(s: Session):
            q = s.query(cls)
            for k, v in where.items():
                q = q.filter(getattr(cls, k) == v)
            return q.delete(synchronize_session=False)

        return self._run(op)

    def validate_schema(self, entity_classes: list[type]) -> None:
        insp = inspect(self.engine)
        for cls in entity_classes:
            t = cls.__table__
            if not insp.has_table(t.name, schema=t.schema):
                raise RuntimeError(f"Table '{t.fullname}' does not exist")
