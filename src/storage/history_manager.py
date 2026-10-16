import os
import json
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import message_to_dict, messages_from_dict, BaseMessage
from src.core.config import settings

class EnterpriseChatHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.storage_dir = settings.CHAT_HISTORY_DIR
        os.makedirs(self.storage_dir, exist_ok=True)
        self.file_path = os.path.join(self.storage_dir, f"{self.session_id}.json")

    def add_messages(self, messages: List[BaseMessage]) -> None:
        all_msgs = list(self.messages)
        all_msgs.extend(messages)
        serialized = [message_to_dict(m) for m in all_msgs]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

    @property
    def messages(self) -> List[BaseMessage]:
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return messages_from_dict(data)
        except Exception:
            return []

    def clear(self) -> None:
        if os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([], f)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return EnterpriseChatHistory(session_id)
