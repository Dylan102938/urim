from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from datetime import datetime

    from urim.dataset import Dataset

FineTuneJobId = str


class FineTuneJobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FineTuneJob:
    id: FineTuneJobId
    created_at: "datetime"
    updated_at: "datetime"
    status: FineTuneJobStatus
    service_identifier: str | None

    model_ids: list[str]

    def serialize(self) -> str:
        import orjson

        return orjson.dumps(self.__dict__).decode("utf-8")

    @classmethod
    def deserialize(cls, serialized: str) -> Self:
        import orjson

        return cls(**orjson.loads(serialized))


class FineTuneService(ABC):
    @abstractmethod
    async def create_job(
        self,
        model: str,
        *,
        train_ds: "Dataset",
        learning_rate: float,
        batch_size: int,
        n_epochs: int,
        **kwargs: Any,
    ) -> FineTuneJob: ...

    @abstractmethod
    async def get_job(self, job_id: FineTuneJobId) -> FineTuneJob: ...

    @abstractmethod
    async def delete_job(self, job_id: FineTuneJobId) -> FineTuneJob: ...

    @abstractmethod
    async def list_jobs(self) -> Sequence[FineTuneJob]: ...
