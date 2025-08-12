from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def _reload_modules_for_tmp_home() -> None:
    # Ensure modules that capture URIM_HOME at import time are reloaded
    import urim.env as env_mod

    importlib.reload(env_mod)


def _write_dummy_ds(path: Path, rows: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(rows):
            f.write(json.dumps({"id": i}) + "\n")


def test_urim_dataset_graph_persistence_and_prune(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("URIM_HOME", tmp_path.as_posix())
    _reload_modules_for_tmp_home()

    from urim.env import URIM_HOME, UrimDatasetGraph  # re-import after reload

    # Prepare dataset files for root and children
    root_id = "root"
    child_id = "child"
    _write_dummy_ds(URIM_HOME / "datasets" / f"{root_id}.jsonl")
    _write_dummy_ds(URIM_HOME / "datasets" / f"{child_id}.jsonl")

    g = UrimDatasetGraph()
    g.set_working_dataset(root_id)
    g.add_child(root_id, child_id, command="sample 1")
    assert g.get_node(root_id) is not None
    assert g.parent(child_id) == root_id
    assert child_id in g.children(root_id)

    # Persist and reload
    g.save()
    g2 = UrimDatasetGraph.from_file()
    assert g2.working_dataset == root_id
    assert g2.path_from_root(child_id) == [root_id, child_id]

    # Prune child -> stays at parent
    g2.prune_from_node(child_id)
    assert g2.working_dataset == root_id
    assert not (URIM_HOME / "datasets" / f"{child_id}.jsonl").exists()

    # Prune root -> working dataset becomes None and file removed
    g2.prune_from_node(root_id)
    assert g2.working_dataset is None
    assert not (URIM_HOME / "datasets" / f"{root_id}.jsonl").exists()
