from typing import TypeVar, Generic, Type
from pydantic import BaseModel
from contextlib import contextmanager
import chainlit as cl

T = TypeVar("T", bound=BaseModel)

class SessionManager(Generic[T]):
    def __init__(self, model_cls: Type[T]) -> None:
        stored_data = cl.user_session.get("state") or {}
        self._model: T = model_cls(**stored_data)
        self.sync()

    def sync(self) -> None:
        """Sync current model state to session"""
        cl.user_session.set("state", self._model.model_dump())

    @contextmanager
    def batch_update(self):
        """Context manager for batching multiple updates"""
        try:
            yield self._model
        finally:
            self.sync()

    def update(self, **kwargs) -> None:
        """Update multiple attributes on the underlying model"""
        with self.batch_update() as model:
            for key, value in kwargs.items():
                setattr(model, key, value)

    @property
    def model(self) -> T:
        """Access to the underlying model instance"""
        return self._model
