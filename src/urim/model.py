from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from urim.store.disk_store import DiskStore

if TYPE_CHECKING:
    from urim.dataset import Dataset
    from urim.ft.controller import FineTuneController

_TAGS_STORE: DiskStore[str, dict[str, Any]] | None = None
_TAGS_STORE_LOCK = threading.RLock()
_FT_STORE: DiskStore[str, dict[str, Any]] | None = None
_FT_STORE_LOCK = threading.RLock()
_FT_CONTROLLER: FineTuneController | None = None
_FT_CONTROLLER_LOCK = asyncio.Lock()


@dataclass(frozen=True)
class ModelRef:
    slug: str
    checkpoints: list[str] = field(default_factory=list)

    def serialize(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "checkpoints": list(self.checkpoints),
        }

    @classmethod
    def deserialize(cls, payload: dict[str, Any]) -> Self:
        return cls(
            slug=payload["slug"],
            checkpoints=payload["checkpoints"],
        )


async def model(
    name: str,
    *,
    train_ds: Dataset | None = None,
    batch_size: int = 1,
    n_epochs: int = 1,
    learning_rate: float = 1.0,
    salt: str = "",
    **ft_kwargs: Any,
) -> ModelRef:
    from urim.ft.controller import FineTuneRequest

    ### return base model or direct finetuned model ###

    if train_ds is None:
        tag_store = get_tag_store()
        cached_ref_payload = tag_store.get(name)
        return (
            ModelRef.deserialize(cached_ref_payload)
            if cached_ref_payload is not None
            else ModelRef(slug=name)
        )

    ### return previously finetuned model by descriptor ###

    ft_store = get_ft_store()
    request = FineTuneRequest(
        model_name=name,
        train_ds=train_ds,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        salt=salt,
        hyperparams=tuple(sorted(ft_kwargs.items(), key=lambda item: item[0])),
    )
    desc_key = request.serialize(cache_dataset=False)
    cached_ref_payload = ft_store.get(desc_key)
    if cached_ref_payload is not None:
        return ModelRef.deserialize(cached_ref_payload)

    ### run finetuning job and return model ref ###

    controller = await get_controller()
    future = await controller.submit(request)
    model_ref = await future

    return model_ref


def get_tag_store() -> DiskStore[str, dict[str, Any]]:
    from urim.env import storage_subdir
    from urim.store.disk_store import DiskStore

    global _TAGS_STORE

    with _TAGS_STORE_LOCK:
        store_path = storage_subdir("models") / "tags.jsonl"
        if _TAGS_STORE is None or _TAGS_STORE.store_path != store_path:
            _TAGS_STORE = DiskStore(store_path)

    return _TAGS_STORE


def get_ft_store() -> DiskStore[str, dict[str, Any]]:
    from urim.env import storage_subdir
    from urim.store.disk_store import DiskStore

    global _FT_STORE

    with _FT_STORE_LOCK:
        store_path = storage_subdir("models") / "ft.jsonl"
        if _FT_STORE is None or _FT_STORE.store_path != store_path:
            _FT_STORE = DiskStore(store_path)

    return _FT_STORE


async def get_controller() -> FineTuneController:
    from urim.ft.controller import FineTuneController

    global _FT_CONTROLLER

    async with _FT_CONTROLLER_LOCK:
        if _FT_CONTROLLER is None:
            _FT_CONTROLLER = FineTuneController()
            await _FT_CONTROLLER.start()

    return _FT_CONTROLLER
