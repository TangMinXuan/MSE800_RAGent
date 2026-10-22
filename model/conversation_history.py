from sqlalchemy import BigInteger, Integer, Sequence, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ConversationHistory(Base):
    __tablename__ = "conversation_history"
    __table_args__ = {"schema": "public"}

    id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        Sequence("conversation_history_id_seq", schema="public", increment=1),
        primary_key=True,
    )
    uid: Mapped[str | None] = mapped_column(String)
    conversation_id: Mapped[str | None] = mapped_column(String)
    conversation_title: Mapped[str | None] = mapped_column(String)
    history: Mapped[str | None] = mapped_column(String)

    def __init__(
        self,
        id: int | None = None,
        uid: str | None = None,
        conversation_id: str | None = None,
        conversation_title: str | None = None,
        history: str | None = None,
    ):
        self.id = id
        self.uid = uid
        self.conversation_id = conversation_id
        self.conversation_title = conversation_title
        self.history = history
