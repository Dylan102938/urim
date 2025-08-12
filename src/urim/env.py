import os
from pathlib import Path
from typing import Self

from pydantic import BaseModel

URIM_HOME = Path(os.environ.get("URIM_HOME", os.path.expanduser("~/.urim")))
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
CUSTOM_BASE_URL = os.environ.get("CUSTOM_BASE_URL")


class UrimDatasetGraphNode(BaseModel):
    command: str | None = None
    parent: str | None = None
    children: list[str] = []


class UrimDatasetGraph(BaseModel):
    working_dataset: str | None = None
    graph: dict[str, UrimDatasetGraphNode] = {}
    _path_from_root_cache: dict[str, list[str]] = {}

    def get_node(self, node_id: str) -> UrimDatasetGraphNode | None:
        return self.graph.get(node_id)

    def parent(self, node_id: str) -> str | None:
        node = self.get_node(node_id)
        return node.parent if node else None

    def children(self, node_id: str) -> list[str]:
        node = self.get_node(node_id)
        return node.children if node else []

    def add_child(
        self,
        parent_id: str,
        child_id: str,
        command: str | None = None,
        save: bool = True,
    ) -> None:
        parent_node = self.get_node(parent_id)
        assert parent_node is not None

        parent_node.children.append(child_id)
        self.graph[child_id] = UrimDatasetGraphNode(parent=parent_id, command=command)

        if save:
            self.save()

    def set_working_dataset(self, dataset_id: str | None, save: bool = True) -> None:
        self.working_dataset = dataset_id
        if dataset_id is not None and dataset_id not in self.graph:
            self.graph[dataset_id] = UrimDatasetGraphNode()

        if save:
            self.save()

    def add_child_and_set_wd(
        self, parent: str, child: str, command: str | None = None, save: bool = True
    ) -> None:
        self.add_child(parent, child, command, save=False)
        self.set_working_dataset(child, save=False)

        if save:
            self.save()

    def save(self) -> None:
        with open(URIM_HOME / "dataset_graph.json", "w") as f:
            f.write(self.model_dump_json(indent=2))

    def path_from_root(self, node_id: str) -> list[str]:
        if node_id in self._path_from_root_cache:
            return self._path_from_root_cache[node_id]

        node = self.get_node(node_id)
        if node is None:
            return []

        parent_id = node.parent
        if parent_id is None:
            self._path_from_root_cache[node_id] = [node_id]
        else:
            self._path_from_root_cache[node_id] = [
                *self.path_from_root(parent_id),
                node_id,
            ]

        return self._path_from_root_cache[node_id]

    def prune_from_node(
        self, node_id: str, is_base_node: bool = True, save: bool = True
    ) -> None:
        node = self.get_node(node_id)
        assert node is not None
        for child_id in list(node.children):
            if self.get_node(child_id) is not None:
                self.prune_from_node(child_id, is_base_node=False, save=False)

        if is_base_node:
            self.set_working_dataset(node.parent, save=False)
            if self.working_dataset is not None:
                curr_node = self.get_node(self.working_dataset)
                assert curr_node is not None
                if node_id in curr_node.children:
                    curr_node.children.pop(curr_node.children.index(node_id))

        self.graph.pop(node_id, None)

        ds_path = URIM_HOME / "datasets" / f"{node_id}.jsonl"
        if ds_path.exists():
            os.remove(ds_path)

        if save:
            self.save()

    @classmethod
    def from_file(cls) -> Self:
        path = URIM_HOME / "dataset_graph.json"
        if path.exists():
            return cls.model_validate_json(open(path).read())
        else:
            with open(path, "w") as f:
                f.write(cls().model_dump_json(indent=2))

            return cls()
