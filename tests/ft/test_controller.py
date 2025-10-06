from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import openai
import pytest
import pytest_asyncio
from httpx import Request, Response

from urim.dataset import Dataset
from urim.ft.controller import FineTuneController, FineTuneRequest
from urim.ft.openai import OpenAIFineTuneJob
from urim.ft.service import FineTuneJobStatus, FineTuneService
from urim.store.disk_store import DiskStore

DiskStoreKey = TypeVar("DiskStoreKey")
DiskStoreValue = TypeVar("DiskStoreValue")


class StubFineTuneService(FineTuneService):
    def __init__(
        self,
        *,
        service_key: str,
        max_concurrent: int = 2,
        time_per_job: float = 0.5,
    ) -> None:
        self.api_key = service_key
        self.jobs: list[OpenAIFineTuneJob] = []
        self.max_concurrent = max_concurrent
        self.time_per_job = time_per_job
        self._job_queue: asyncio.Queue[str] = asyncio.Queue()
        self._job_loop: asyncio.Task = asyncio.create_task(self._process_job_queue())
        self._stop_job_loop = asyncio.Event()

    async def stop(self) -> None:
        self._stop_job_loop.set()

    async def create_job(
        self,
        model: str,
        train_ds: Dataset,
        **hyperparams: Any,
    ) -> OpenAIFineTuneJob:
        running_jobs = [job for job in self.jobs if job.status == FineTuneJobStatus.RUNNING]
        if len(running_jobs) >= self.max_concurrent:
            raise openai.RateLimitError(
                "rate limited",
                response=Response(status_code=429, request=Request(method="POST", url="test.com")),
                body={},
            )

        job = OpenAIFineTuneJob(
            id=f"stub-job-{len(self.jobs)}",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            status=FineTuneJobStatus.RUNNING,
            service_identifier=self.api_key,
            raw=None,
        )
        self.jobs.append(job)

        await self._job_queue.put(job.id)

        return job

    async def get_job(self, job_id: str) -> OpenAIFineTuneJob:
        for job in self.jobs:
            if job.id == job_id:
                return job

        raise ValueError(f"Job {job_id} not found")

    async def delete_job(self, job_id: str) -> OpenAIFineTuneJob:
        for job in self.jobs:
            if job.id == job_id:
                self.jobs.remove(job)
                return job

        raise ValueError(f"Job {job_id} not found")

    async def list_jobs(self) -> list[OpenAIFineTuneJob]:
        return list(self.jobs)

    async def _process_job_queue(self) -> None:
        while not self._stop_job_loop.is_set():
            job = await self._job_queue.get()

            await asyncio.sleep(self.time_per_job)

            for real_job in self.jobs:
                if real_job.id == job:
                    real_job.status = FineTuneJobStatus.COMPLETED
                    break


@pytest_asyncio.fixture(autouse=True)
async def stub_service(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[dict[str, StubFineTuneService]]:
    STUB_FT_SERVICE_REGISTRY: dict[str, StubFineTuneService] = {
        "primary-key": StubFineTuneService(service_key="primary-key"),
        "secondary-key": StubFineTuneService(service_key="secondary-key"),
    }

    monkeypatch.setattr(
        "urim.ft.controller.OpenAIFineTuneService",
        lambda service_key: STUB_FT_SERVICE_REGISTRY[service_key],
    )

    yield STUB_FT_SERVICE_REGISTRY

    for service in STUB_FT_SERVICE_REGISTRY.values():
        await service.stop()


def make_request() -> FineTuneRequest:
    import pandas as pd

    return FineTuneRequest(
        model_name="gpt-ft",
        train_ds=Dataset(pd.DataFrame({"text": ["hello", "world"]})),
        learning_rate=0.1,
        batch_size=2,
        n_epochs=1,
        salt=str(uuid.uuid4()),
    )


async def test_basic_single_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "urim.env.collect_openai_keys",
        lambda: ["primary-key"],
    )

    controller = FineTuneController(
        max_concurrent=1,
        poll_status_interval=0.1,
        retry_submission_interval=0.1,
        cache_dir=tmp_path / "ft",
    )
    await controller.start()

    future = await controller.submit(make_request())
    job = await asyncio.wait_for(future, timeout=1.5)
    assert job.service_identifier == "primary-key"
    assert job.status == FineTuneJobStatus.COMPLETED
    assert job.id == "stub-job-0"

    await controller.stop()


async def test_rotate_service_when_rate_limited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "urim.env.collect_openai_keys",
        lambda: ["primary-key", "secondary-key"],
    )

    controller = FineTuneController(
        max_concurrent=3,
        poll_status_interval=0.1,
        cache_dir=tmp_path / "ft",
    )
    await controller.start()

    future1 = await controller.submit(make_request())
    future2 = await controller.submit(make_request())
    future3 = await controller.submit(make_request())

    job1 = await asyncio.wait_for(future1, timeout=1.5)
    job2 = await asyncio.wait_for(future2, timeout=1.5)
    job3 = await asyncio.wait_for(future3, timeout=1.5)

    assert job1.id == "stub-job-0"
    assert job1.service_identifier == "primary-key"
    assert job2.id == "stub-job-1"
    assert job2.service_identifier == "primary-key"
    assert job3.id == "stub-job-0"
    assert job3.service_identifier == "secondary-key"


async def test_submit_job_no_capacity(
    monkeypatch: pytest.MonkeyPatch, stub_service: dict[str, StubFineTuneService], tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "urim.env.collect_openai_keys",
        lambda: ["primary-key"],
    )

    controller = FineTuneController(
        max_concurrent=1,
        poll_status_interval=0.1,
        retry_submission_interval=0.1,
        cache_dir=tmp_path / "ft",
    )
    await controller.start()

    future1 = await controller.submit(make_request())
    future2 = await controller.submit(make_request())
    future3 = await controller.submit(make_request())

    async def wait_for_primary_jobs() -> list[str]:
        while True:
            job_ids = [job.id for job in stub_service["primary-key"].jobs]
            if "stub-job-0" in job_ids and "stub-job-1" in job_ids and len(job_ids) == 2:
                return job_ids

            await asyncio.sleep(0.05)

    await asyncio.sleep(0.05)
    await asyncio.wait_for(wait_for_primary_jobs(), timeout=1.5)

    assert len(stub_service["primary-key"].jobs) == 2
    assert stub_service["primary-key"].jobs[0].id == "stub-job-0"
    assert stub_service["primary-key"].jobs[1].id == "stub-job-1"

    job1 = await asyncio.wait_for(future1, timeout=1.5)
    job2 = await asyncio.wait_for(future2, timeout=1.5)
    job3 = await asyncio.wait_for(future3, timeout=1.5)

    assert job1.id == "stub-job-0"
    assert job2.id == "stub-job-1"
    assert job3.id == "stub-job-2"

    await controller.stop()


async def test_restart_recovers_inflight_jobs(
    monkeypatch: pytest.MonkeyPatch,
    stub_service: dict[str, StubFineTuneService],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "urim.env.collect_openai_keys",
        lambda: ["primary-key"],
    )

    service = stub_service["primary-key"]
    service.time_per_job = 0.2
    controller = FineTuneController(
        max_concurrent=2,
        poll_status_interval=0.05,
        retry_submission_interval=0.05,
        cache_dir=tmp_path / "ft",
    )
    await controller.start()

    request = make_request()
    request2 = make_request()
    request3 = make_request()
    future1 = await controller.submit(request)
    future2 = await controller.submit(request2)
    future3 = await controller.submit(request3)

    async def wait_for_inflight() -> None:
        while not (request in controller._inflight and request2 in controller._inflight):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait_for_inflight(), timeout=1.0)
    await controller.stop()

    assert not future1.done()
    assert not future2.done()
    assert not future3.done()

    async def wait_for_job_completion() -> None:
        while any(job.status != FineTuneJobStatus.COMPLETED for job in service.jobs):
            await asyncio.sleep(0.02)

    await asyncio.wait_for(wait_for_job_completion(), timeout=1.5)

    await controller.start()

    assert request in controller._inflight
    assert request2 in controller._inflight
    assert request3 not in controller._inflight
    assert request3 in controller._futures

    future1 = await controller.submit(request)
    future2 = await controller.submit(request2)
    future3 = await controller.submit(request3)

    job1 = await asyncio.wait_for(future1, timeout=1.5)
    job2 = await asyncio.wait_for(future2, timeout=1.5)
    job3 = await asyncio.wait_for(future3, timeout=1.5)

    assert job1.status == FineTuneJobStatus.COMPLETED
    assert job1.id == "stub-job-0"
    assert job1.service_identifier == "primary-key"

    assert job2.status == FineTuneJobStatus.COMPLETED
    assert job2.id == "stub-job-1"
    assert job2.service_identifier == "primary-key"

    assert job3.status == FineTuneJobStatus.COMPLETED
    assert job3.id == "stub-job-2"
    assert job3.service_identifier == "primary-key"

    await controller.stop()

    store: DiskStore[str, str] = DiskStore(tmp_path / "ft" / "inflight.jsonl")
    assert store._page.empty
    assert not (tmp_path / "ft" / "datasets" / f"{hash(request.train_ds)}.jsonl").exists()
