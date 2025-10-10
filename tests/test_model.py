from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from urim.dataset import Dataset
from urim.env import set_storage_root, storage_file
from urim.ft.controller import FineTuneRequest
from urim.model import ModelRef, get_ft_store, model
from urim.store.disk_store import DiskStore


@pytest.fixture(autouse=True)
def configure_storage(tmp_path: Path) -> None:
    set_storage_root(tmp_path)


async def test_model_returns_base_model_when_no_tag() -> None:
    ref = await model("gpt-4.1")
    assert ref.slug == "gpt-4.1"
    assert ref.checkpoints == []


async def test_model_resolves_tag_from_disk() -> None:
    tag_store: DiskStore[str, dict[str, Any]] = DiskStore(storage_file("models", "tags.jsonl"))
    expected = {"slug": "ft-model", "checkpoints": ["ft-model-ckpt"]}
    tag_store.put("alias", expected)
    await asyncio.to_thread(tag_store.flush)

    ref = await model("alias")

    assert ref == ModelRef(slug="ft-model", checkpoints=["ft-model-ckpt"])


async def test_model_resolve_with_descriptor() -> None:
    ft_store: DiskStore[str, dict[str, Any]] = DiskStore(storage_file("models", "ft.jsonl"))
    expected = {"slug": "ft_model", "checkpoints": ["ft_model_ckpt"]}
    ds = Dataset("databricks/databricks-dolly-15k", split="train")
    request = FineTuneRequest(
        model_name="model",
        train_ds=ds,
        learning_rate=0.5,
        batch_size=8,
        n_epochs=2,
    )

    ft_store.put(request.serialize(cache_dataset=False), expected)
    await asyncio.to_thread(ft_store.flush)

    ref = await model(
        "model",
        train_ds=ds.sample(frac=100),
        learning_rate=0.5,
        batch_size=8,
        n_epochs=2,
    )
    assert ref == ModelRef(slug="ft_model", checkpoints=["ft_model_ckpt"])


async def test_model_submits_finetune(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = Dataset("databricks/databricks-dolly-15k", split="train")

    class StubController:
        def __init__(self) -> None:
            self.requests: list = []

        async def submit(self, request: FineTuneRequest) -> asyncio.Future[ModelRef]:
            self.requests.append(request)
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            model_ref = ModelRef(
                slug="ft-alias",
                checkpoints=[
                    "ft-alias-ckpt",
                ],
            )
            fut.set_result(model_ref)

            model_store = get_ft_store()
            model_store.put(
                request.serialize(cache_dataset=False),
                model_ref.serialize(),
            )
            await asyncio.to_thread(model_store.flush)

            return fut

        async def start(self) -> None:
            return

    stub_controller = StubController()

    def fake_get_controller() -> StubController:
        return stub_controller

    monkeypatch.setattr("urim.ft.controller.FineTuneController", fake_get_controller)

    ref = await model(
        "alias",
        train_ds=ds,
        batch_size=8,
        n_epochs=2,
        learning_rate=0.2,
        extra="value",
    )

    assert ref == ModelRef(slug="ft-alias", checkpoints=["ft-alias-ckpt"])

    ref_cached = await model(
        "alias",
        train_ds=ds.sample(frac=1),
        batch_size=8,
        n_epochs=2,
        learning_rate=0.2,
        extra="value",
    )

    assert ref_cached == ref
    assert len(stub_controller.requests) == 1
