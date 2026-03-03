"""
history.py
----------
基于本地文件的对话历史存储（从京东客服项目移植）。
每个session_id对应一个JSON文件，支持多用户隔离。
"""
import json
import os
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import CHAT_HISTORY_DIR


def get_history(session_id: str) -> "FileChatMessageHistory":
    """工厂函数，供 RunnableWithMessageHistory 调用。"""
    return FileChatMessageHistory(session_id, CHAT_HISTORY_DIR)


class FileChatMessageHistory(BaseChatMessageHistory):
    """将对话历史序列化为JSON文件存储在本地。"""

    def __init__(self, session_id: str, storage_path: str):
        self.session_id  = session_id
        self.file_path   = os.path.join(storage_path, f"{session_id}.json")
        os.makedirs(storage_path, exist_ok=True)

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return messages_from_dict(json.load(f))
        except FileNotFoundError:
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages) + list(messages)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([message_to_dict(m) for m in all_messages], f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
