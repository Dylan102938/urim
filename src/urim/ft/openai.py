from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from urim.ft.service import (
    FineTuneJob,
    FineTuneJobId,
    FineTuneJobStatus,
    FineTuneService,
)

if TYPE_CHECKING:
    from openai.types.fine_tuning import FineTuningJob

    from urim.dataset import Dataset

_STATUS_MAP: dict[str, FineTuneJobStatus] = {
    "validating_files": FineTuneJobStatus.PENDING,
    "queued": FineTuneJobStatus.PENDING,
    "running": FineTuneJobStatus.RUNNING,
    "succeeded": FineTuneJobStatus.COMPLETED,
    "failed": FineTuneJobStatus.FAILED,
    "cancelled": FineTuneJobStatus.FAILED,
}


@dataclass
class OpenAIFineTuneJob(FineTuneJob):
    id: FineTuneJobId
    created_at: datetime
    updated_at: datetime
    status: FineTuneJobStatus
    service_identifier: str

    model_ids: list[str] = field(default_factory=list)

    raw: Any = field(default_factory=dict)


class OpenAIFineTuneService(FineTuneService):
    def __init__(self, *, service_key: str) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=service_key)
        self.api_key = service_key

    async def create_job(
        self,
        model: str,
        *,
        train_ds: Dataset,
        learning_rate: float = 1.0,
        batch_size: int = 4,
        n_epochs: int = 1,
        **kwargs: Any,
    ) -> OpenAIFineTuneJob:
        training_file_id = await self._upload_training_file(train_ds)
        job = await self._client.fine_tuning.jobs.create(
            model=model,
            training_file=training_file_id,
            hyperparameters={
                "batch_size": batch_size,
                "learning_rate_multiplier": learning_rate,
                "n_epochs": n_epochs,
            },
            **kwargs,
        )
        return await self._convert_job(job)

    async def get_job(self, job_id: FineTuneJobId) -> OpenAIFineTuneJob:
        job = await self._client.fine_tuning.jobs.retrieve(job_id)
        return await self._convert_job(job)

    async def delete_job(self, job_id: FineTuneJobId) -> OpenAIFineTuneJob:
        job = await self._client.fine_tuning.jobs.cancel(job_id)
        return await self._convert_job(job)

    async def list_jobs(self) -> list[OpenAIFineTuneJob]:
        response = await self._client.fine_tuning.jobs.list()
        jobs = response.data

        converted = await asyncio.gather(
            *(self._convert_job(job) for job in jobs),
            return_exceptions=True,
        )

        return [job for job in converted if not isinstance(job, BaseException)]

    async def _upload_training_file(self, dataset: Dataset) -> str:
        import io

        buffer = io.BytesIO()
        dataset.df().to_json(buffer)
        buffer.seek(0)

        upload = await self._client.files.create(
            file=("training.jsonl", buffer),
            purpose="fine-tune",
        )

        return upload.id

    async def _convert_job(self, job: FineTuningJob) -> OpenAIFineTuneJob:
        model_ids: set[str] = set()
        if job.status == "succeeded" and job.fine_tuned_model is not None:
            checkpoints = await self._get_checkpoints(job.id)
            model_ids.update(checkpoints)
            model_ids.add(job.fine_tuned_model)

        return OpenAIFineTuneJob(
            id=job.id,
            created_at=datetime.fromtimestamp(job.created_at, tz=timezone.utc),
            updated_at=datetime.fromtimestamp(job.created_at, tz=timezone.utc),
            status=_STATUS_MAP.get(job.status, FineTuneJobStatus.PENDING),
            service_identifier=self.api_key,
            model_ids=list(model_ids),
            raw=job,
        )

    async def _get_checkpoints(self, job_id: FineTuneJobId) -> list[str]:
        checkpoints = self._client.fine_tuning.jobs.checkpoints.list(
            fine_tuning_job_id=job_id,
        )

        models: list[str] = []
        async for checkpoint in checkpoints:
            models.append(checkpoint.fine_tuned_model_checkpoint)

        return models
