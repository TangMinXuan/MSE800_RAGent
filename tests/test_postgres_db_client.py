import pytest
from sqlalchemy import create_engine

from infra.database.postgres_db_client import PostgresDbClient, _fields, jdbc_to_sqlalchemy_url
from model import Base, ConversationHistory


def test_jdbc_url_conversion():
    assert jdbc_to_sqlalchemy_url("jdbc:postgresql://localhost:5432/ragent") == "postgresql+psycopg://localhost:5432/ragent"
    assert jdbc_to_sqlalchemy_url("postgresql+psycopg://localhost/ragent") == "postgresql+psycopg://localhost/ragent"


def test_check_table_validates_mapped_entity_name():
    PostgresDbClient._check_table("conversation_history", ConversationHistory)
    PostgresDbClient._check_table("public.conversation_history", ConversationHistory)
    with pytest.raises(ValueError, match="does not match entity table"):
        PostgresDbClient._check_table("wrong", ConversationHistory)
    with pytest.raises(ValueError, match="mapped entity"):
        PostgresDbClient._check_table("anything", object)


def test_fields_returns_only_non_null_mapped_columns():
    row = ConversationHistory(conversation_id="c1", conversation_title=None, history="[]")
    values = _fields(row)
    assert values["conversation_id"] == "c1"
    assert values["history"] == "[]"
    assert "conversation_title" not in values


def test_update_requires_same_entity_class_and_conditions():
    client = PostgresDbClient("sqlite://", engine=create_engine("sqlite://"))
    with pytest.raises(ValueError, match="same entity class"):
        client.update("conversation_history", ConversationHistory(conversation_title="x"), object())
    with pytest.raises(ValueError, match="at least one non-null condition"):
        client.update("conversation_history", ConversationHistory(conversation_title="x"), ConversationHistory())
    with pytest.raises(ValueError, match="no non-null fields"):
        client.update("conversation_history", ConversationHistory(), ConversationHistory(conversation_id="c1"))


def test_delete_requires_nonempty_condition():
    client = PostgresDbClient("sqlite://", engine=create_engine("sqlite://"))
    with pytest.raises(ValueError, match="at least one non-null condition"):
        client.delete("conversation_history", ConversationHistory())


def test_from_environment_requires_configuration(monkeypatch):
    PostgresDbClient._instance = None
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.delenv("POSTGRES_JDBC_URL", raising=False)
    with pytest.raises(RuntimeError, match="POSTGRES_URL"):
        PostgresDbClient.from_environment()
    monkeypatch.setenv("POSTGRES_URL", "postgresql+psycopg://localhost/db")
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    with pytest.raises(RuntimeError, match="POSTGRES_USER"):
        PostgresDbClient.from_environment()
    PostgresDbClient._instance = None
