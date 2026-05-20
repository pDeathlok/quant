from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from enum import Enum


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Notifier:
    def __init__(self):
        self.handlers: List["NotificationHandler"] = []

    def add_handler(self, handler: "NotificationHandler"):
        self.handlers.append(handler)

    def send(self, message: str, level: AlertLevel = AlertLevel.INFO, context: Optional[Dict] = None):
        for handler in self.handlers:
            handler.send(message, level, context)


class NotificationHandler(ABC):
    @abstractmethod
    def send(self, message: str, level: AlertLevel, context: Optional[Dict]):
        pass


class LogHandler(NotificationHandler):
    def send(self, message: str, level: AlertLevel, context: Optional[Dict]):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level.value.upper()}] {message}")


class FileHandler(NotificationHandler):
    def __init__(self, filepath: str = "./logs/alerts.log"):
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filepath = filepath

    def send(self, message: str, level: AlertLevel, context: Optional[Dict]):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [{level.value.upper()}] {message}\n")
