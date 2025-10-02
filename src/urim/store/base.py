from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class Store(ABC):
    @abstractmethod
    def full(self) -> bool: ...

    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def put(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def remove(self, key: str) -> None: ...
