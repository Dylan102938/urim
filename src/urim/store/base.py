from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    pass


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class Store(Generic[K, V], ABC):
    @abstractmethod
    def full(self) -> bool: ...

    @abstractmethod
    def get(self, key: K) -> V | None: ...

    @abstractmethod
    def put(self, key: K, value: V) -> None: ...

    @abstractmethod
    def remove(self, key: K) -> None: ...
