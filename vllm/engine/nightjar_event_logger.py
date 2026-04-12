# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional


class NightjarEventLogger:
    """Lightweight JSONL logger for Nightjar experiment events."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._path: Optional[Path] = None
        if path:
            self.configure(path)

    @property
    def path(self) -> Optional[str]:
        return str(self._path) if self._path is not None else None

    def configure(self, path: Optional[str]) -> None:
        with self._lock:
            self._path = Path(path).expanduser() if path else None
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._path is None:
            return
        record = {
            "timestamp": time.time(),
            "event_type": event_type,
            **payload,
        }
        with self._lock:
            if self._path is None:
                return
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
