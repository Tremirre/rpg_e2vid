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
        loss_config: dict[str, tuple[torch.nn.Module, float]],
        history: History,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_config = loss_config
        self.history = history

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
    ) -> None:
        for epoch in range(epochs):
            start_time = time.time()
            train_losses = self._train_epoch(train_loader)
            val_losses = self._val_epoch(val_loader)
            self.history.add_entry(
                epoch, train_losses, val_losses, time.time() - start_time
            )
            print(f"\n=== Epoch {epoch} ===\n")
            print(f"Time: {time.time() - start_time:.2f}")
            for name in self.loss_config:
                print(
                    f"{name} - train: {train_losses[name]:.4f} | val: {val_losses[name]:.4f}"
                )
            print()
            self._on_epoch_end(epoch, val_loader)

    def _train_epoch(
        self, train_loader: torch.utils.data.DataLoader
    ) -> dict[str, float]:
        self.model.train()
        total_epoch_loss = 0
        individiual_losses = {name: 0 for name in self.loss_config}
        pbar = tqdm.tqdm(train_loader, desc="Training", total=len(train_loader))
        for i, (events, images) in enumerate(pbar):
            self.optimizer.zero_grad()
            output = self.model(events)[0]
            losses = {
                name: criterion(output, images) * weight
                for name, (criterion, weight) in self.loss_config.items()
            }
            total_loss = sum(losses.values())
            total_loss.backward()
            self.optimizer.step()
            total_epoch_loss += total_loss.item()
            for name, loss in losses.items():
                individiual_losses[name] += loss.item()
            pbar.set_postfix({"train_total_loss": total_epoch_loss / (i + 1)})
        individiual_losses = {
            name: loss / len(train_loader) for name, loss in individiual_losses.items()
        }
        return individiual_losses

    def _val_epoch(self, val_loader: torch.utils.data.DataLoader) -> dict[str, float]:
        self.model.eval()
        total_epoch_loss = 0
        individiual_losses = {name: 0 for name in self.loss_config}
        with torch.no_grad():
            pbar = tqdm.tqdm(val_loader, desc="Validation", total=len(val_loader))
            for i, (events, images) in enumerate(pbar):
                output = self.model(events)[0]
                losses = {
                    name: criterion(output, images) * weight
                    for name, (criterion, weight) in self.loss_config.items()
                }
                total_loss = sum(losses.values())
                total_epoch_loss += total_loss.item()
                for name, loss in losses.items():
                    individiual_losses[name] += loss.item()
                pbar.set_postfix({"val_total_loss": total_epoch_loss / (i + 1)})
        individiual_losses = {
            name: loss / len(val_loader) for name, loss in individiual_losses.items()
        }
        return individiual_losses

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
