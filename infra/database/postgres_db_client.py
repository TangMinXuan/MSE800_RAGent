import os
import threading
from typing import Any

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from util.string_util import is_blank


def jdbc_to_sqlalchemy_url(url: str) -> str:
    """Convert jdbc:postgresql://host:5432/db to postgresql+psycopg://host:5432/db."""
    if url.startswith("jdbc:postgresql://"):
        return "postgresql+psycopg://" + url[len("jdbc:postgresql://"):]
    return url


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
        with cls._lock:
            if cls._instance is None:
                # POSTGRES_URL (SQLAlchemy 格式) 优先；兼容 Java 的 POSTGRES_JDBC_URL
                url = os.getenv("POSTGRES_URL")
                url_name = "POSTGRES_URL"
                if is_blank(url) and not is_blank(os.getenv("POSTGRES_JDBC_URL")):
                    url, url_name = os.getenv("POSTGRES_JDBC_URL"), "POSTGRES_JDBC_URL"
                user = os.getenv("POSTGRES_USER")
                password = os.getenv("POSTGRES_PASSWORD")
                if is_blank(url):
                    raise RuntimeError(f"Missing required environment variable: {url_name}")
                if is_blank(user):
                    raise RuntimeError("Missing required environment variable: POSTGRES_USER")
                if password is None:
                    raise RuntimeError("Missing required environment variable: POSTGRES_PASSWORD")
                cls._instance = cls(url, user, password)
            return cls._instance

    @staticmethod
    def _check_table(table_name: str, entity_cls: type) -> None:
        table = getattr(entity_cls, "__table__", None)
        if table is None:
            raise ValueError(f"Class must be a mapped entity: {entity_cls.__name__}")
        expected = f"{table.schema}.{table.name}" if table.schema else table.name
        if table_name != table.name and table_name != expected:
            raise ValueError(f"Table name {table_name} does not match entity table {expected}")

    def _run(self, fn, action: str, entity_cls: type):
        with Session(self.engine) as session:
            try:
                result = fn(session)
                session.commit()
                return result
            except Exception as exception:
                session.rollback()
                raise RuntimeError(f"Database {action} failed for {entity_cls.__name__}") from exception

    def insert(self, table_name: str, entity: Any) -> int:
        self._check_table(table_name, type(entity))

        def op(s: Session):
            s.add(entity)
            return 1

        return self._run(op, "insert", type(entity))

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

        return self._run(op, "select", entity_cls)

    def update(self, table_name: str, values: Any, conditions: Any) -> int:
        cls = type(values)
        if type(conditions) is not cls:
            raise ValueError("Update values and conditions must use the same entity class")
        self._check_table(table_name, cls)
        new, where = _fields(values), _fields(conditions)
        if not new:
            raise ValueError("Update object has no non-null fields")
        if not where:
            raise ValueError("Update requires at least one non-null condition")

        def op(s: Session):
            q = s.query(cls)
            for k, v in where.items():
                q = q.filter(getattr(cls, k) == v)
            return q.update(new, synchronize_session=False)

        return self._run(op, "update", cls)

    def delete(self, table_name: str, conditions: Any) -> int:
        cls = type(conditions)
        self._check_table(table_name, cls)
        where = _fields(conditions)
        if not where:
            raise ValueError("Delete requires at least one non-null condition")

        def op(s: Session):
            q = s.query(cls)
            for k, v in where.items():
                q = q.filter(getattr(cls, k) == v)
            return q.delete(synchronize_session=False)

        return self._run(op, "delete", cls)

    def validate_schema(self, entity_classes: list[type]) -> None:
        insp = inspect(self.engine)
        for cls in entity_classes:
            t = cls.__table__
            if not insp.has_table(t.name, schema=t.schema):
                raise RuntimeError(f"Table '{t.fullname}' does not exist")
