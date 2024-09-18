import torch
import piqa


class MAELoss(torch.nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(output - target))


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((output - target) ** 2)


class SSIMLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(SSIMLoss, self).__init__()
        self.ssim = piqa.SSIM(*args, **kwargs)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1 - self.ssim(output, target)


class MSSSIMLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(MSSSIMLoss, self).__init__()
        self.msssim = piqa.MS_SSIM(*args, **kwargs)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1 - self.msssim(output, target)
