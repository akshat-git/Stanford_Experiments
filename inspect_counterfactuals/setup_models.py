import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler

# ────────────────────────── Model A ──────────────────────────
class PlainAutoencoder(nn.Module):
    """Regular generative model: simple conv autoencoder."""
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.ReLU(),     # 128 → 64
            nn.Conv2d(64,128,4,2,1), nn.ReLU(),       # 64  → 32
            nn.Conv2d(128,256,4,2,1), nn.ReLU())      # 32  → 16
        self.fc_mu = nn.Linear(256*16*16, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 256*16*16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(),  # 16→32
            nn.ConvTranspose2d(128,64,4,2,1),  nn.ReLU(),  # 32→64
            nn.ConvTranspose2d(64,1,4,2,1),   nn.Sigmoid())# 64→128

    def forward(self, x):
        z = self.encoder(x).view(x.size(0), -1)
        z = self.fc_mu(z)
        x_hat = self.decoder_fc(z).view(x.size(0),256,16,16)
        return self.decoder(x_hat)

# ────────────────────────── Model B ──────────────────────────
class CondAutoencoder(torch.nn.Module):
    def __init__(self, cond_dim=2):  # default to 2 for PE + mortality
        super().__init__()
        image_dim = 128 * 128  # 16384
        total_input = image_dim + cond_dim  # e.g., 16386
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(total_input, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(256 + cond_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, image_dim),
            torch.nn.Sigmoid(),
        )
        self.image_dim = image_dim
        self.cond_dim = cond_dim

    def forward(self, x, cond):
        x = x.view(x.size(0), -1)  # Flatten image: (B, 16384)
        x = torch.cat([x, cond], dim=1)  # (B, 16386)
        z = self.encoder(x)
        z = torch.cat([z, cond], dim=1)  # (B, 256 + cond_dim)
        out = self.decoder(z)
        return out.view(-1, 1, 128, 128)

# ────────────────────────── Model C (CoLa‑diff) ─────────────────────
class CoLaDiffUNet(nn.Module):
    """
    Latent‑diffusion model (CoLa‑diff style) for 128×128 grayscale images.
    Uses HuggingFace diffusers UNet2DConditionModel + DDPM scheduler.
    """
    def __init__(self, cond_dim=2, img_size=128):
        super().__init__()
        self.unet = UNet2DConditionModel(
            sample_size=img_size,
            in_channels=1,
            out_channels=1,
            class_embed_type=None,
            encoder_hid_dim=cond_dim,
            block_out_channels=(64,128,256),
            down_block_types=("DownBlock2D","AttnDownBlock2D","DownBlock2D"),
            up_block_types  =("UpBlock2D","AttnUpBlock2D","UpBlock2D"),
            layers_per_block=2,
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)

    def forward(self, x, cond):
        """
        Denoising step: x is clean image; we add noise and predict noise.
        """
        b = x.size(0)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps,
                                  (b,), device=x.device).long()
        noisy = self.scheduler.add_noise(x, noise, timesteps)
        pred_noise = self.unet(noisy, timesteps, encoder_hidden_states=cond).sample
        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def generate(self, cond, num_steps=50):
        """
        Reverse DDPM sampling to generate a counterfactual image
        conditioned on `cond` (shape B×cond_dim).
        """
        b = cond.size(0)
        img = torch.randn(b,1,128,128, device=cond.device)
        self.scheduler.set_timesteps(num_steps)
        for t in self.scheduler.timesteps:
            noise_pred = self.unet(img, t, encoder_hidden_states=cond).sample
            img = self.scheduler.step(noise_pred, t, img).prev_sample
        return img.clamp(0,1)
