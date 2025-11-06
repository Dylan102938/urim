from __future__ import annotations

import asyncio
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from typing_extensions import Self

from urim.ft.openai import OpenAIFineTuneJob, OpenAIFineTuneService
from urim.logging import get_logger

if TYPE_CHECKING:
    from urim.dataset import Dataset
    from urim.model import ModelRef

RequestQueue = asyncio.PriorityQueue[tuple[float, int, "FineTuneRequest" | Literal["terminate"]]]


logger = get_logger("ft.controller")


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
                "model_name": str(self.model_name),
                "train_ds": dsid,
                "learning_rate": float(self.learning_rate),
                "batch_size": int(self.batch_size),
                "n_epochs": int(self.n_epochs),
                "salt": str(self.salt),
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

    def __repr__(self) -> str:  # noqa: D401
        hyperparams = ", ".join(f"{k}={v}" for k, v in dict(self.hyperparams).items())
        dataset_hash = hash(self.train_ds)
        base = (
            f"FineTuneRequest(model={self.model_name}, ds_hash={dataset_hash}, "
            f"lr={self.learning_rate}, batch={self.batch_size}, epochs={self.n_epochs}"
        )
        if self.salt:
            base += f", salt={self.salt}"
        if hyperparams:
            base += f", hyperparams={{{hyperparams}}}"
        return base + ")"


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
            logger.warning("start called but controller is already running.")
            return

        logger.info(
            "Starting FineTuneController with max_concurrent=%s, poll_interval=%ss.",
            self.max_concurrent,
            self.poll_status_interval,
        )

        self._status_poller = asyncio.create_task(self._poll_status(), name="ft-status-poller")
        self._submission_loop = asyncio.create_task(
            self._handle_queue(self._queue, queue_name="primary"),
            name="primary-loop",
        )
        self._retry_submission_loop = asyncio.create_task(
            self._handle_queue(
                self._retry_queue,
                self.retry_submission_interval,
                queue_name="retry",
            ),
            name="retry-loop",
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
            logger.debug(
                "Recovered inflight job %s for request %s from persistent cache.",
                job.id,
                request,
            )

        self._ready.set()
        logger.info("FineTuneController is ready. %d inflight job(s) found.", len(self._inflight))

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

        logger.info("FineTuneController stopped.")

    async def submit(
        self, request: FineTuneRequest, priority: float = 0.0
    ) -> asyncio.Future[ModelRef]:
        if not self._ready.is_set():
            raise RuntimeError("Controller is not ready")

        if request in self._futures:
            logger.debug("Reusing existing future for request %s.", request)
            return self._futures[request]

        fut: asyncio.Future[ModelRef] = asyncio.get_event_loop().create_future()
        self._futures[request] = fut
        await self._queue.put((priority, next(self._queue_order), request))

        logger.debug(
            "Queued fine-tune request %s with priority %.2f. queue_size=%d",
            request,
            priority,
            self._queue.qsize(),
        )

        return fut

    async def _handle_queue(
        self,
        queue: RequestQueue,
        sleep_before: float = 0.0,
        *,
        queue_name: str,
    ) -> None:
        await self._ready.wait()

        while True:
            _, _, request = await queue.get()
            try:
                if request == "terminate":
                    logger.debug("Received termination sentinel on %s queue.", queue_name)
                    break
                if request in self._inflight:
                    logger.debug(
                        "Request %s already inflight; skipping submission on %s queue.",
                        request,
                        queue_name,
                    )
                    continue
                if request not in self._futures:
                    logger.warning(
                        "Request %s retrieved from %s queue without a tracked future; dropping.",
                        request,
                        queue_name,
                    )
                    continue
                if sleep_before > 1e-8:
                    logger.debug(
                        "Sleeping %.2fs before retrying request %s from %s queue.",
                        sleep_before,
                        request,
                        queue_name,
                    )
                    await asyncio.sleep(sleep_before)

                logger.debug(
                    "Attempting fine-tune submission for %s via %s queue.", request, queue_name
                )

                job = await self._attempt_ft(request)
                if job is None:
                    logger.debug(
                        "Submission for %s did not succeed; enqueueing on retry queue.",
                        request,
                    )
                    await self._retry_queue.put((0, next(self._queue_order), request))
                else:
                    self._inflight[request] = job
                    self._inflight_store.put(
                        request.serialize(cache_dataset=True),
                        job.serialize(),
                    )
                    await asyncio.to_thread(self._inflight_store.flush)
                    logger.info(
                        "Submitted fine-tune request %s; OpenAI job id=%s.",
                        request,
                        job.id,
                    )
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
                logger.debug(
                    "Submitting fine-tune request %s using OpenAI key ending with %s.",
                    request,
                    key[-4:] if len(key) >= 4 else "***",
                )
                job = await service.create_job(
                    request.model_name,
                    train_ds=request.train_ds,
                    learning_rate=request.learning_rate,
                    batch_size=request.batch_size,
                    n_epochs=request.n_epochs,
                    **dict(request.hyperparams),
                )
                logger.debug(
                    "Fine-tune request %s accepted by OpenAI job id=%s.",
                    request,
                    job.id,
                )
                return job
            except RateLimitError:
                logger.warning(
                    "Rate limit encountered when submitting request %s with current OpenAI key.",
                    request,
                )
                continue
            except Exception as e:
                logger.exception("Unexpected error while submitting request %s: %s", request, e)
                raise

        return job

    async def _poll_request(self, request: FineTuneRequest) -> ModelRef | None:
        from urim.ft.service import FineTuneJobStatus
        from urim.model import ModelRef, get_ft_store

        try:
            stale_job = self._inflight[request]
            ft_service_key = stale_job.service_identifier
            ft_service = OpenAIFineTuneService(service_key=ft_service_key)
            job = await ft_service.get_job(stale_job.id)
        except Exception as e:
            logger.exception("Unexpected error while polling request %s: %s", request, e)
            return None

        logger.debug(
            "Polled job %s for request %s. Previous status=%s, new status=%s.",
            job.id,
            request,
            stale_job.status,
            job.status,
        )

        self._inflight[request] = job
        self._inflight_store.put(
            request.serialize(cache_dataset=False),
            job.serialize(),
        )
        await asyncio.to_thread(self._inflight_store.flush)

        if job.status in (FineTuneJobStatus.COMPLETED, FineTuneJobStatus.FAILED):
            future = self._futures.get(request)
            if future is None or future.done():
                return None

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

                logger.info(
                    "Fine-tune job %s completed for request %s. Model slug=%s.",
                    job.id,
                    request,
                    model_ref.slug,
                )

                return model_ref
            else:
                logger.error("Fine-tune job %s failed for request %s.", job.id, request)
                raise RuntimeError(f"Fine-tune job {job.id} failed")

        return None

    async def _poll_status(self) -> None:
        from urim.env import storage_subdir

        await self._ready.wait()

        while not self._stop_poller.is_set():
            logger.debug("Polling status for %d inflight request(s).", len(self._inflight))
            inflight = list(self._inflight.keys())
            refs_or_errors = await asyncio.gather(
                *(self._poll_request(request) for request in inflight),
                return_exceptions=True,
            )

            ### Cleanup stale futures ###

            stale = [
                request
                for request, model_ref in zip(inflight, refs_or_errors, strict=False)
                if model_ref is not None
            ]
            for request, model_ref in zip(inflight, refs_or_errors, strict=False):
                if model_ref is None:
                    continue

                self._inflight_store.remove(request.serialize(cache_dataset=False))
                await asyncio.to_thread(self._inflight_store.flush)

                dataset_hash = hash(request.train_ds)
                ds_path = storage_subdir("ft", "datasets") / f"{dataset_hash}.jsonl"
                if ds_path.exists():
                    shared_inflight = any(
                        hash(other_request.train_ds) == dataset_hash
                        for other_request in self._inflight
                        if other_request not in stale
                    )
                    if shared_inflight:
                        logger.debug(
                            "Keeping cached dataset for request %s; inflight job still uses"
                            " hash %s.",
                            request,
                            dataset_hash,
                        )
                    else:
                        ds_path.unlink()
                        logger.debug(
                            "Removed cached dataset for request %s at %s.",
                            request,
                            ds_path,
                        )

                logger.debug(
                    "Cleaned up completed request %s and removed it from persistence.",
                    request,
                )

                self._inflight.pop(request)
                future = self._futures.pop(request)
                if isinstance(model_ref, BaseException):
                    future.set_exception(model_ref)
                else:
                    future.set_result(model_ref)

            if not self._stop_poller.is_set():
                logger.debug("Sleeping %.2fs before next status poll.", self.poll_status_interval)
                await asyncio.sleep(self.poll_status_interval)
