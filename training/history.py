import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from .vis import show_sample

HISTORY_FILE = "history.json"
IMG_DIR = "images"


class History:
    def __init__(
        self,
        model_name: str,
        output_dir: pathlib.Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        runs = [int(name.name.split("_")[0]) for name in output_dir.glob("*")]
        latest_run = 0
        if runs:
            latest_run = max(runs)
        self.run_idx = latest_run + 1
        self.model_name = model_name
        self._output_dir = output_dir
        self.history = []

    @property
    def full_dir(self) -> pathlib.Path:
        return self._output_dir / f"{self.run_idx:>04}_{self.model_name}"

    def add_entry(
        self, epoch: int, train_loss: float, val_loss: float, time: float
    ) -> None:
        self.history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time": time,
            }
        )

    def save(self) -> None:
        with open(self.full_dir / HISTORY_FILE, "w") as f:
            json.dump(self.history, f)

    def add_result_image(
        self,
        image: np.ndarray,
        events: np.ndarray,
        result: np.ndarray,
        epoch: int,
        idx: int,
    ):
        image_dir = self.full_dir / IMG_DIR
        image_dir.mkdir(parents=True, exist_ok=True)
        show_sample(image, events, result)
        plt.savefig(image_dir / f"epoch_E{epoch:>04}_i{idx}.png")
        plt.close()
