"""A shared image generator trained through the unchanged LGM objective."""
import torch
from torch import nn
from torch.nn import functional as F

from config import DistillCfg
from data.dataloaders import BaseRealDataset
from my_utils.log_utils import log_images

from .base import BaseDistilledDataset


class ImageGenerator(nn.Module):
    def __init__(self, latent_dim: int = 128, channels: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8)),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, channels, 3, padding=1),
        )

    def forward(self, noise):
        return self.network(noise)


class GeneratorDataset(BaseDistilledDataset):
    def __init__(self, train_dataset: BaseRealDataset, cfg: DistillCfg, device="cuda"):
        if cfg.distill_opt != "adam":
            raise ValueError("generator_pixel requires --distill_opt=adam")
        if cfg.init_mode != "noise":
            raise ValueError("generator_pixel requires --init_mode=noise")
        if cfg.generator_latent_dim < 1 or cfg.ipc < 1 or cfg.syn_res < 1:
            raise ValueError("latent dimension, IPC and syn_res must be positive")
        super().__init__(device=device)
        self.cfg = cfg
        self.train_dataset = train_dataset
        # All existing real datasets and the PixelDataset use RGB images.
        self.generator = ImageGenerator(cfg.generator_latent_dim, channels=3).to(device)
        self.noise = torch.randn(
            train_dataset.num_classes * cfg.ipc, cfg.generator_latent_dim,
            device=device, requires_grad=False,
        )
        self.syn_labels = torch.arange(
            train_dataset.num_classes, device=device, dtype=torch.long
        ).repeat_interleave(cfg.ipc)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=cfg.lr)

    def get_data(self):
        images = self.generator(self.noise)
        if images.shape[-2:] != (self.cfg.syn_res, self.cfg.syn_res):
            images = F.interpolate(
                images, size=(self.cfg.syn_res, self.cfg.syn_res),
                mode="bilinear", align_corners=False,
            )
        if self.cfg.decorrelate_color:
            images = self.linear_decorrelate_color(images)
        return torch.sigmoid(2 * images), self.syn_labels

    @torch.no_grad()
    def log_images(self, step=None):
        images, _ = self.get_data()
        log_images(syn_images=images.detach().clone(), step=step)

    def _representation(self):
        return {
            "version": 1, "mode": "generator_pixel",
            "latent_dim": self.cfg.generator_latent_dim,
            "channels": 3, "syn_res": self.cfg.syn_res,
            "num_classes": self.train_dataset.num_classes, "ipc": self.cfg.ipc,
            "decorrelate_color": self.cfg.decorrelate_color,
        }

    def get_save_dict(self):
        return {
            "representation": self._representation(),
            "generator": self.generator.state_dict(),
            "noise": self.noise.detach().clone(),
            "labels": self.syn_labels.detach().clone(),
            "opt_state": self.optimizer.state_dict(),
        }

    def load_from_dict(self, load_dict):
        if load_dict["representation"] != self._representation():
            raise ValueError("Generator checkpoint configuration does not match")
        noise = load_dict["noise"].to(self.noise.device)
        labels = load_dict["labels"].to(self.syn_labels.device)
        if noise.shape != self.noise.shape or not torch.isfinite(noise).all():
            raise ValueError("Invalid fixed noise in Generator checkpoint")
        if not torch.equal(labels, self.syn_labels):
            raise ValueError("Generator checkpoint labels do not match")
        self.generator.load_state_dict(load_dict["generator"])
        with torch.no_grad():
            self.noise.copy_(noise)
        self.optimizer.load_state_dict(load_dict["opt_state"])
