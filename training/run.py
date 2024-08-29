import torch
import torch.utils.data
import time

import tqdm


class Runner:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        criterion: torch.nn.Module,
    ) -> None:
        self.history = []
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
    ) -> None:
        for epoch in range(epochs):
            start_time = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss = self._val_epoch(val_loader)
            self.history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )
            print(
                f"Epoch {epoch} | Train Loss {train_loss} | Val Loss {val_loss} | Time {time.time() - start_time}"
            )

    def _train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for events, images in tqdm.tqdm(
            train_loader, desc="Training", total=len(train_loader)
        ):
            self.optimizer.zero_grad()
            output = self.model(events)[0]
            loss = self.criterion(output, images)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def _val_epoch(self, val_loader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for events, images in tqdm.tqdm(
                val_loader, desc="Validation", total=len(val_loader)
            ):
                output = self.model(events)[0]
                loss = self.criterion(output, images)
                total_loss += loss.item()
        return total_loss / len(val_loader)
