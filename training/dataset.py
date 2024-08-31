import typing
import logging
import pathlib
import dataclasses
import collections
import random

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

Transform = typing.Callable[[np.ndarray], np.ndarray | torch.Tensor]

logger = logging.getLogger(__name__)


class FileCache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._cache = {}
        self.insert_queue = collections.deque()

    def __contains__(self, key):
        return key in self._cache

    def __getitem__(self, key):
        return self._cache[key]

    def __setitem__(self, key, value):
        if len(self._cache) >= self.max_size:
            oldest = self.insert_queue.popleft()
            del self._cache[oldest]
        self._cache[key] = value
        self.insert_queue.append(key)


class D3Dataset(Dataset):
    def __init__(
        self,
        batch_files: typing.Sequence[pathlib.Path],
        transforms: tuple[Transform] = tuple(),
        load_device: torch.device = torch.device("cpu"),
        file_batch_size: int = 128,
        cache_size: int = 100,
        preload: bool = False,
    ):
        self.batch_files = batch_files
        for file in self.batch_files:
            assert file.exists(), f"File {file} does not exist."
        self.transforms = transforms
        self.load_device = load_device
        self.file_batch_size = file_batch_size
        self.cache = FileCache(cache_size)
        if preload:
            logger.info("Preloading dataset")
            for file_idx in range(len(self.batch_files)):
                if file_idx >= cache_size:
                    break
                self.cache[file_idx] = self.load_file(file_idx)

    def load_file(self, file_idx: int) -> dict[str, np.ndarray]:
        batch_file = self.batch_files[file_idx]
        logger.info(f"Loading file {batch_file}")
        batch = np.load(batch_file)
        b_events = batch["polarity_data"].astype(np.float32)
        b_images = batch["frame_data"].astype(np.float32) / 255.0
        return {"polarity_data": b_events, "frame_data": b_images}

    def __len__(self) -> int:
        return len(self.batch_files) * self.file_batch_size

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        logger.debug(f"Getting item {idx}")
        file_idx = idx // self.file_batch_size
        if file_idx not in self.cache:
            batch = self.load_file(file_idx)
            self.cache[file_idx] = batch
        batch = self.cache[file_idx]

        idx = idx % self.file_batch_size
        logger.debug(f"Getting item {idx} from file {file_idx}")
        events = batch["polarity_data"][idx]
        image = batch["frame_data"][idx][np.newaxis, ...]

        for transform in self.transforms + (lambda x: torch.from_numpy(x),):
            events = transform(events)  # type: ignore
            image = transform(image)  # type: ignore

        return events.to(self.load_device), image.to(self.load_device)  # type: ignore


@dataclasses.dataclass
class Loaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader

    @classmethod
    def from_path(
        cls,
        path: pathlib.Path,
        transforms: tuple[Transform] = tuple(),
        percentage: tuple[float, float, float] = (0.7, 0.15, 0.15),
        device: torch.device = torch.device("cpu"),
        batch_size: int = 32,
        num_workers: int = 1,
        cache_size: int = 100,
        preload: bool = False,
        drop_percentage: float = 0.0,
    ):
        batch_files = list(path.glob("*.npz"))
        if drop_percentage > 0:
            batch_files = random.sample(
                batch_files, int(len(batch_files) * (1 - drop_percentage))
            )
        random.shuffle(batch_files)
        total = len(batch_files)
        train_end = int(total * percentage[0])
        val_end = int(total * (percentage[0] + percentage[1]))

        train = D3Dataset(
            batch_files[:train_end],
            transforms,
            load_device=device,
            cache_size=cache_size,
            preload=preload,
        )
        val = D3Dataset(
            batch_files[train_end:val_end],
            transforms,
            load_device=device,
            cache_size=cache_size,
            preload=preload,
        )
        test = D3Dataset(
            batch_files[val_end:],
            transforms,
            load_device=device,
            cache_size=cache_size,
            preload=preload,
        )

        return cls(
            DataLoader(
                train, batch_size=batch_size, shuffle=True, num_workers=num_workers
            ),
            DataLoader(
                val, batch_size=batch_size, shuffle=False, num_workers=num_workers
            ),
            DataLoader(
                test, batch_size=batch_size, shuffle=False, num_workers=num_workers
            ),
        )
