"""Session-scoped conversational memory powered by LangChain."""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage


class ConversationMemory:
    """Small memory wrapper with load/save helpers for chat sessions."""

    def __init__(self) -> None:
        self._history = InMemoryChatMessageHistory()

    def load_memory_variables(self, _: dict) -> dict[str, str]:
        lines: list[str] = []
        for msg in self._history.messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return {"history": "\n".join(lines)}

    def save_context(self, inputs: dict[str, str], outputs: dict[str, str]) -> None:
        question = inputs.get("question", "")
        answer = outputs.get("answer", "")
        if question:
            self._history.add_message(HumanMessage(content=question))
        if answer:
            self._history.add_message(AIMessage(content=answer))


_MEMORY_STORE: dict[str, ConversationMemory] = {}


def get_conversation_memory(session_id: str) -> ConversationMemory:
    """Return or create conversation memory for a given session."""
    if session_id not in _MEMORY_STORE:
        _MEMORY_STORE[session_id] = ConversationMemory()
    return _MEMORY_STORE[session_id]


def clear_conversation_memory(session_id: str) -> None:
    """Clear one session memory if present."""
    _MEMORY_STORE.pop(session_id, None)


def clear_all_conversation_memory() -> None:
    """Clear all in-memory chat histories."""
    _MEMORY_STORE.clear()
