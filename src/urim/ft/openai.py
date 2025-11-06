from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from urim.ft.service import (
    FineTuneJob,
    FineTuneJobId,
    FineTuneJobStatus,
    FineTuneService,
)
from urim.logging import get_logger

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

logger = get_logger("ft.openai")


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
        logger.debug(
            "Initialized OpenAIFineTuneService with API key ending %s.",
            service_key[-4:] if len(service_key) >= 4 else "***",
        )

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
        logger.debug(
            "Creating fine-tune job for model=%s with lr=%s, batch_size=%s, epochs=%s.",
            model,
            learning_rate,
            batch_size,
            n_epochs,
        )
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
        logger.debug(
            "OpenAI accepted fine-tune job request for model=%s; job id=%s.",
            model,
            job.id,
        )
        return await self._convert_job(job)

    async def get_job(self, job_id: FineTuneJobId) -> OpenAIFineTuneJob:
        job = await self._client.fine_tuning.jobs.retrieve(job_id)
        logger.debug("Fetched fine-tune job %s from OpenAI.", job_id)
        return await self._convert_job(job)

    async def delete_job(self, job_id: FineTuneJobId) -> OpenAIFineTuneJob:
        job = await self._client.fine_tuning.jobs.cancel(job_id)
        logger.info("Requested cancellation for fine-tune job %s.", job_id)
        return await self._convert_job(job)

    async def list_jobs(self) -> list[OpenAIFineTuneJob]:
        response = await self._client.fine_tuning.jobs.list()
        jobs = response.data

        converted = await asyncio.gather(
            *(self._convert_job(job) for job in jobs),
            return_exceptions=True,
        )

        result = [job for job in converted if not isinstance(job, BaseException)]
        logger.debug("Retrieved %d fine-tune job(s) from OpenAI.", len(result))
        return result

    async def _upload_training_file(self, dataset: Dataset) -> str:
        import io

        buffer = io.BytesIO()
        df = dataset.df()
        logger.debug("Uploading training dataset with %d row(s) for fine-tuning.", len(df))
        df.to_json(buffer, orient="records", lines=True)
        buffer.seek(0)

        upload = await self._client.files.create(
            file=("training.jsonl", buffer),
            purpose="fine-tune",
        )

        logger.debug("Uploaded training file to OpenAI; file id=%s.", upload.id)
        return upload.id

    async def _convert_job(self, job: FineTuningJob) -> OpenAIFineTuneJob:
        model_ids: set[str] = set()
        if job.status == "succeeded" and job.fine_tuned_model is not None:
            checkpoints = await self._get_checkpoints(job.id)
            model_ids.update(checkpoints)
            model_ids.add(job.fine_tuned_model)

        logger.debug(
            "Converted OpenAI job %s with status=%s and %d model id(s).",
            job.id,
            job.status,
            len(model_ids),
        )

        return OpenAIFineTuneJob(
            id=job.id,
            created_at=datetime.fromtimestamp(job.created_at, tz=UTC),
            updated_at=datetime.fromtimestamp(job.created_at, tz=UTC),
            status=_STATUS_MAP.get(job.status, FineTuneJobStatus.PENDING),
            service_identifier=self.api_key,
            model_ids=list(model_ids),
        )

    async def _get_checkpoints(self, job_id: FineTuneJobId) -> list[str]:
        checkpoints = self._client.fine_tuning.jobs.checkpoints.list(
            fine_tuning_job_id=job_id,
        )

        models: list[str] = []
        async for checkpoint in checkpoints:
            models.append(checkpoint.fine_tuned_model_checkpoint)

        logger.debug(
            "Discovered %d checkpoint model(s) for fine-tune job %s.",
            len(models),
            job_id,
        )
        return models
