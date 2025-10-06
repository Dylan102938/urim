from __future__ import annotations

import asyncio
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Self

from urim.ft.openai import OpenAIFineTuneJob, OpenAIFineTuneService

if TYPE_CHECKING:
    from urim.dataset import Dataset
    from urim.model import ModelRef

RequestQueue = asyncio.PriorityQueue[tuple[float, int, "FineTuneRequest" | Literal["terminate"]]]


@dataclass(frozen=True, eq=True)
class FineTuneRequest:
    model_name: str
    train_ds: Dataset
    learning_rate: float
    batch_size: int
    n_epochs: int
    salt: str = field(default="")
    hyperparams: tuple[tuple[str, Hashable], ...] = field(default_factory=tuple)

    def serialize(self, cache_dataset: bool = True) -> str:
        import orjson

        from urim.env import storage_subdir

        dsid = hash(self.train_ds)
        if cache_dataset:
            self.train_ds.to_json(storage_subdir("ft", "datasets") / f"{dsid}.jsonl")

        return orjson.dumps(
            {
                "model_name": self.model_name,
                "train_ds": dsid,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
                "salt": self.salt,
                "hyperparams": dict(self.hyperparams),
            },
            option=orjson.OPT_SORT_KEYS,
        ).decode("utf-8")

    @classmethod
    def deserialize(cls, serialized: str) -> Self:
        import orjson

        from urim.dataset import Dataset
        from urim.env import storage_subdir

        obj = orjson.loads(serialized)
        hyperparams: dict[str, Hashable] = obj["hyperparams"]
        ds_path = storage_subdir("ft", "datasets") / f"{obj['train_ds']}.jsonl"

        return cls(
            model_name=obj["model_name"],
            train_ds=Dataset(ds_path),
            learning_rate=obj["learning_rate"],
            batch_size=obj["batch_size"],
            n_epochs=obj["n_epochs"],
            salt=obj["salt"],
            hyperparams=tuple((k, v) for k, v in hyperparams.items()),
        )


class FineTuneController:
    def __init__(
        self,
        *,
        max_concurrent: int = 2,
        poll_status_interval: float = 60.0,
        retry_submission_interval: float = 40.0,
    ):
        from itertools import count

        from urim.env import storage_subdir
        from urim.store.disk_store import DiskStore

        self.max_concurrent = max_concurrent
        self.poll_status_interval = poll_status_interval
        self.retry_submission_interval = retry_submission_interval
        self._ready = asyncio.Event()

        self._queue: RequestQueue = asyncio.PriorityQueue()
        self._retry_queue: RequestQueue = asyncio.PriorityQueue()
        self._queue_order = count()

        self._submission_loop: asyncio.Task | None = None
        self._retry_submission_loop: asyncio.Task | None = None
        self._status_poller: asyncio.Task | None = None
        self._stop_poller = asyncio.Event()

        self._futures: dict[FineTuneRequest, asyncio.Future[ModelRef]] = {}
        self._inflight: dict[FineTuneRequest, OpenAIFineTuneJob] = {}
        self._inflight_store: DiskStore[str, str] = DiskStore(
            storage_subdir("ft") / "inflight.jsonl"
        )

    async def start(self) -> None:
        if self._submission_loop and self._retry_submission_loop and self._status_poller:
            return

        self._status_poller = asyncio.create_task(self._poll_status())
        self._submission_loop = asyncio.create_task(self._handle_queue(self._queue))
        self._retry_submission_loop = asyncio.create_task(
            self._handle_queue(self._retry_queue, self.retry_submission_interval)
        )

        for _, row in self._inflight_store._page.iterrows():
            try:
                request = FineTuneRequest.deserialize(row.name)
                job = OpenAIFineTuneJob.deserialize(row["value"])
            except Exception as e:
                self._inflight_store.remove(row.name)
                await asyncio.to_thread(self._inflight_store.flush)
                raise Exception(
                    f"Job {row.name} is corrupted. You will need to re-submit the ft request."
                ) from e

            self._inflight[request] = job
            self._futures[request] = asyncio.get_event_loop().create_future()

        self._ready.set()

    async def stop(self) -> None:
        self._stop_poller.set()
        await self._retry_queue.put((-1, next(self._queue_order), "terminate"))
        await self._queue.put((-1, next(self._queue_order), "terminate"))
        await asyncio.gather(
            *[
                task
                for task in (
                    self._submission_loop,
                    self._retry_submission_loop,
                    self._status_poller,
                )
                if task is not None
            ],
            return_exceptions=True,
        )
        self._submission_loop = None
        self._retry_submission_loop = None
        self._status_poller = None
        self._ready.clear()
        self._stop_poller.clear()

    async def submit(
        self, request: FineTuneRequest, priority: float = 0.0
    ) -> asyncio.Future[ModelRef]:
        assert self._ready.is_set(), "Controller is not ready"

        if request in self._futures:
            return self._futures[request]

        fut: asyncio.Future[ModelRef] = asyncio.get_event_loop().create_future()
        self._futures[request] = fut
        await self._queue.put((priority, next(self._queue_order), request))

        return fut

    async def _handle_queue(self, queue: RequestQueue, sleep_before: float = 0.0) -> None:
        await self._ready.wait()

        while True:
            _, _, request = await queue.get()
            try:
                if request == "terminate":
                    break
                if request in self._inflight:
                    continue
                if request not in self._futures:
                    print(f"Request {request} does not have a corresponding future.")
                    continue
                if sleep_before > 1e-8:
                    await asyncio.sleep(sleep_before)

                job = await self._attempt_ft(request)
                if job is None:
                    await self._retry_queue.put((0, next(self._queue_order), request))
                else:
                    self._inflight[request] = job
                    self._inflight_store.put(
                        request.serialize(cache_dataset=True),
                        job.serialize(),
                    )
                    await asyncio.to_thread(self._inflight_store.flush)
            finally:
                queue.task_done()

    async def _attempt_ft(self, request: FineTuneRequest) -> OpenAIFineTuneJob | None:
        from openai import RateLimitError

        from urim.env import collect_openai_keys

        assert request not in self._inflight, f"Request {request} already inflight."
        assert request in self._futures, f"Request {request} does not have a corresponding future."

        keys = collect_openai_keys()
        job: OpenAIFineTuneJob | None = None
        for key in keys:
            service = OpenAIFineTuneService(service_key=key)
            try:
                return await service.create_job(
                    request.model_name,
                    train_ds=request.train_ds,
                    learning_rate=request.learning_rate,
                    batch_size=request.batch_size,
                    n_epochs=request.n_epochs,
                    **dict(request.hyperparams),
                )
            except RateLimitError:
                print("Rate limit error", request)
                continue
            except Exception as e:
                print(f"Error submitting job: {e}")
                raise e

        return job

    async def _poll_request(self, request: FineTuneRequest) -> None:
        from urim.ft.service import FineTuneJobStatus
        from urim.model import ModelRef, get_ft_store

        stale_job = self._inflight[request]
        ft_service_key = stale_job.service_identifier
        ft_service = OpenAIFineTuneService(service_key=ft_service_key)
        job = await ft_service.get_job(stale_job.id)

        self._inflight[request] = job
        self._inflight_store.put(
            request.serialize(cache_dataset=False),
            job.serialize(),
        )
        await asyncio.to_thread(self._inflight_store.flush)

        if job.status in (FineTuneJobStatus.COMPLETED, FineTuneJobStatus.FAILED):
            future = self._futures.get(request)
            if future is None or future.done():
                return

            if job.status == FineTuneJobStatus.COMPLETED:
                model_ref = ModelRef(
                    slug=next(mid for mid in job.model_ids if "ckpt-step-" not in mid),
                    checkpoints=list(sorted(job.model_ids)),
                )
                model_store = get_ft_store()
                model_store.put(
                    request.serialize(cache_dataset=False),
                    model_ref.serialize(),
                )
                await asyncio.to_thread(model_store.flush)
                future.set_result(model_ref)
            else:
                future.set_exception(RuntimeError(f"Fine-tune job {job.id} failed"))

    async def _poll_status(self) -> None:
        from urim.env import storage_subdir

        await self._ready.wait()

        while not self._stop_poller.is_set():
            await asyncio.gather(
                *(self._poll_request(request) for request in self._inflight.keys()),
                return_exceptions=True,
            )

            ### Cleanup stale futures ###

            stale = [request for request, fut in self._futures.items() if fut.done()]
            for request in stale:
                self._futures.pop(request)
                self._inflight.pop(request)
                self._inflight_store.remove(request.serialize(cache_dataset=False))
                await asyncio.to_thread(self._inflight_store.flush)

                ds_path = storage_subdir("ft", "datasets") / f"{hash(request.train_ds)}.jsonl"
                if ds_path.exists():
                    ds_path.unlink()

            if not self._stop_poller.is_set():
                await asyncio.sleep(self.poll_status_interval)
