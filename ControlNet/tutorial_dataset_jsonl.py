import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    JSONL format per line:
    {"source": "...", "target": "...", "prompt": "..."}

    ControlNet convention:
    - hint <- source image (control map), normalized to [0, 1]
    - jpg  <- target image (training target), normalized to [-1, 1]
    """

    def __init__(
        self,
        jsonl_path: str = "../training_data_qwen.jsonl",
        project_root: str = "..",
        image_size: int = 512,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        raw_jsonl = Path(jsonl_path)
        if raw_jsonl.is_absolute():
            self.jsonl_path = raw_jsonl
        else:
            self.jsonl_path = (self.project_root / raw_jsonl).resolve()
        self.image_size = image_size

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL not found: {self.jsonl_path}")

        self.data: list[dict[str, str]] = []
        with self.jsonl_path.open("rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.data.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | str]:
        item = self.data[idx]

        source_path = self._resolve_path(item["source"])
        target_path = self._resolve_path(item["target"])
        prompt = item["prompt"]

        source = cv2.imread(str(source_path))
        target = cv2.imread(str(target_path))

        if source is None:
            raise FileNotFoundError(f"Could not read source image: {source_path}")
        if target is None:
            raise FileNotFoundError(f"Could not read target image: {target_path}")

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        source = cv2.resize(
            source,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        target = cv2.resize(
            target,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
