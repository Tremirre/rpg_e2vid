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
        criterions: tuple[torch.nn.Module, ...],
        history: History,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        assert len(criterions) > 0, "At least one criterion is required"
        self.criterions = criterions
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

            loss = 0
            for criterion in self.criterions:
                loss += criterion(output, images)
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
                loss = 0
                for criterion in self.criterions:
                    loss += criterion(output, images)
                total_loss += loss.item()
                pbar.set_postfix({"val_loss": total_loss / (i + 1)})
        return total_loss / len(val_loader)

    def _on_epoch_end(
        self, epoch: int, val_loader: torch.utils.data.DataLoader
    ) -> None:
        num_examples = len(val_loader.dataset)
        first = 0
        middle = num_examples // 2 + 1
        last = num_examples - 1
        a_events, a_images = val_loader.dataset[first]
        b_events, b_images = val_loader.dataset[middle]
        c_events, c_images = val_loader.dataset[last]
        events = torch.stack((a_events, b_events, c_events))
        images = torch.stack((a_images, b_images, c_images))
        output = self.model(events)[0]
        for i in range(len(output)):
            self.history.add_result_image(
                images[i][0].cpu().numpy(),
                events[i].cpu().numpy(),
                output[i][0].cpu().detach().numpy(),
                epoch,
                i,
            )
        self.history.save()
