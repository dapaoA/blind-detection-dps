import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

from guided_diffusion.unet import UNetModel

class VAE(nn.Module):
    def __init__(self, unet_model, latent_dim):
        super(VAE, self).__init__()
        self.encoder = unet_model
        self.fc_mu = nn.Linear(unet_model.out_channels, latent_dim)
        self.fc_logvar = nn.Linear(unet_model.out_channels, latent_dim)
        self.decoder = unet_model
        
    def encode(self, x):
        h = self.encoder(x, timesteps=torch.zeros(x.size(0), device=x.device))
        h = h.mean(dim=[2, 3])  # Global average pooling
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        z = z.view(z.size(0), -1, 1, 1).expand(-1, -1, self.decoder.image_size, self.decoder.image_size)
        return self.decoder(z, timesteps=torch.zeros(z.size(0), device=z.device))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VAETrainer(pl.LightningModule):
    def __init__(self, image_size, in_channels, model_channels, latent_dim):
        super(VAETrainer, self).__init__()
        unet = UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=in_channels,
            num_res_blocks=2,
            attention_resolutions=(16,),
            dropout=0.1,
            channel_mult=(1, 2, 4),
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=4,
            num_head_channels=32,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
        )
        self.vae = VAE(unet, latent_dim)
        
    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, mu, logvar = self.vae(x)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_div
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

# Data preparation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model training
model = VAETrainer(image_size=64, in_channels=1, model_channels=64, latent_dim=20)
trainer = pl.Trainer(max_epochs=10, gpus=1 if torch.cuda.is_available() else 0)
trainer.fit(model, dataloader)