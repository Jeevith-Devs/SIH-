"""
Thread-safe event dispatch system for decoupling simulation/tracking backends from UI.
"""

from typing import Callable, Dict, List, Any
import threading


class EventBus:
    """Lightweight publish-subscribe bus for telemetry and status notifications."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[..., None]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_name: str, handler: Callable[..., None]):
        with self._lock:
            if event_name not in self._subscribers:
                self._subscribers[event_name] = []
            if handler not in self._subscribers[event_name]:
                self._subscribers[event_name].append(handler)

    def unsubscribe(self, event_name: str, handler: Callable[..., None]):
        with self._lock:
            if event_name in self._subscribers and handler in self._subscribers[event_name]:
                self._subscribers[event_name].remove(handler)

    def emit(self, event_name: str, *args, **kwargs):
        handlers = []
        with self._lock:
            if event_name in self._subscribers:
                handlers = list(self._subscribers[event_name])
        for handler in handlers:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                print(f"[EventBus Error] in handler for {event_name}: {e}")

    def clear(self):
        with self._lock:
            self._subscribers.clear()


# Global event bus instance
global_event_bus = EventBus()
