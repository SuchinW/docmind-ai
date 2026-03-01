"""Conversation history management for DocMind AI."""

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class ChatHistory:
    """Simple conversation history manager."""

    def __init__(self, max_messages: int = 20):
        """Initialize chat history.

        Args:
            max_messages: Maximum number of message pairs to keep.
        """
        self.messages: list[BaseMessage] = []
        self.max_messages = max_messages * 2  # pairs

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.messages.append(HumanMessage(content=content))
        self._trim()

    def add_ai_message(self, content: str) -> None:
        """Add an AI message to history."""
        self.messages.append(AIMessage(content=content))
        self._trim()

    def get_messages(self) -> list[BaseMessage]:
        """Get all messages in history."""
        return list(self.messages)

    def clear(self) -> None:
        """Clear all history."""
        self.messages.clear()

    def _trim(self) -> None:
        """Trim history to max_messages."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]
