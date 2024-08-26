import typing
import pathlib
import numpy as np

from torch import device
from torch.utils.data import Dataset

Transform = typing.Callable[[np.ndarray], np.ndarray]


class D3Dataset(Dataset):
    def __init__(
        self,
        batch_files: typing.Iterable[pathlib.Path],
        transforms: tuple[Transform] = tuple(),
        load_device: device = device("cpu"),
    ):
        self.batch_files = batch_files
        for file in self.batch_files:
            assert file.exists(), f"File {file} does not exist."
        self.transforms = transforms
        self.load_device = load_device

    def __len__(self) -> int:
        return len(self.batch_files)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        batch_file = self.batch_files[idx]
        batch = np.load(batch_file)
        events = batch["polarity_data"].astype(np.float32)
        image = batch["frame_data"].astype(np.float32) / 255.0 - 0.5

        for transform in self.transforms:
            events = transform(events)
            image = transform(image)

        return events, image
