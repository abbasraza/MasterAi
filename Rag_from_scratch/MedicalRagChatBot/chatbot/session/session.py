from langchain_community.chat_message_histories import ChatMessageHistory


class SessionManager:

    def __init__(self, max_messages: int = 4):
        self._store: dict[str, ChatMessageHistory] = {}
        self.max_messages = max_messages

    def get(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    def trim(self, session_id: str):
        history = self.get(session_id)
        if len(history.messages) > self.max_messages:
            history.messages = history.messages[-self.max_messages:]