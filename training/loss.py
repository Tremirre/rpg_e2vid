import torch
import piqa


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