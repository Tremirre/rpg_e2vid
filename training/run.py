import time
import typing

import torch
import torch.utils.data
import tqdm

from .history import History


class Optimizer(typing.Protocol):
    def zero_grad(self) -> None: ...

    def step(self) -> None: ...


class Runner:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        criterion: torch.nn.Module,
        history: History,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = history

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
            self.history.add_entry(
                epoch, train_loss, val_loss, time.time() - start_time
            )
            print(
                f"Epoch {epoch} | Train Loss {train_loss} | Val Loss {val_loss} | Time {time.time() - start_time}"
            )
            self._on_epoch_end(epoch, val_loader)

    def _train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0
        pbar = tqdm.tqdm(train_loader, desc="Training", total=len(train_loader))
        for i, (events, images) in enumerate(pbar):
            self.optimizer.zero_grad()
            output = self.model(events)[0]
            loss = self.criterion(output, images)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"train_loss": total_loss / (i + 1)})
        return total_loss / len(train_loader)

    def _val_epoch(self, val_loader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            pbar = tqdm.tqdm(val_loader, desc="Validation", total=len(val_loader))
            for i, (events, images) in enumerate(pbar):
                output = self.model(events)[0]
                loss = self.criterion(output, images)
                total_loss += loss.item()
                pbar.set_postfix({"val_loss": total_loss / (i + 1)})
        return total_loss / len(val_loader)

    def _on_epoch_end(
        self, epoch: int, val_loader: torch.utils.data.DataLoader
    ) -> None:
        example = next(iter(val_loader))
        events, images = example
        output = self.model(events[:3])[0]
        for i in range(len(output)):
            self.history.add_result_image(
                images[i][0].cpu().numpy(),
                events[i].cpu().numpy(),
                output[i][0].cpu().detach().numpy(),
                epoch,
                i,
            )
        self.history.save()
