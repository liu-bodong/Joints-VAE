import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# from chamfer_distance import ChamferDistance # TODO: NOT WORKING!!!!!!
# from pytorch3d.loss import chamfer_distance

# UPPER_ARM_LEN = 0.33
# FOREARM_LEN = 0.28


class Encoder(nn.Module):
    def __init__(self, input_dim=4, global_feat_dim=1024, latent_dim=32):
        super().__init__()
        self.feat_mlp = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, global_feat_dim), nn.ReLU()
        )
        # VAE head
        self.fc_mu = nn.Linear(global_feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(global_feat_dim, latent_dim)
        
    def forward(self, x):
        # x shape: [B, M, 4]
        x = self.feat_mlp(x)       # [B, M, 1024]
        x = torch.max(x, 1)[0]     # [B, 1024] (Global max pooling)
        mu = self.fc_mu(x)         # [B, D_z]
        logvar = self.fc_logvar(x) # [B, D_z]
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=32, num_task_points=256, output_dim=3):
        super().__init__()
        self.num_task_points = num_task_points
        
        # create a 2D grid scaffold
        grid = torch.meshgrid(
            torch.linspace(-0.5, 0.5, int(np.sqrt(num_task_points))),
            torch.linspace(-0.5, 0.5, int(np.sqrt(num_task_points)))
        )
        grid = torch.stack(grid, dim=-1).view(-1, 2) # [K, 2]
        self.register_buffer('grid', grid) # store as non-parameter buffer

        # folding MLP
        self.folding_mlp = nn.Sequential(
            nn.Linear(latent_dim + 2, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, output_dim) # output 3d coords
        )

    def forward(self, z):
        # z shape: [B, D_z]
        batch_size = z.shape[0]
        
        # Replicate z for each grid point
        z_replicated = z.unsqueeze(1).repeat(1, self.num_task_points, 1) # [B, K, D_z]
        
        # Replicate grid for each batch item
        grid_replicated = self.grid.unsqueeze(0).repeat(batch_size, 1, 1) # [B, K, 2]

        x = torch.cat([z_replicated, grid_replicated], dim=2) # [B, K, D_z + 2]
        
        output_cloud = self.folding_mlp(x) # [B, K, 3]
        return output_cloud
    
class JointLimitDecoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=256, output_dim=8):
        super().__init__()
        # map from latent z to the 8 limit parameters
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) # Output 8 values
        )

    def forward(self, z):
        # z shape: [B, D_z]
        limits_recon = self.decoder_mlp(z) # [B, 8]
        return limits_recon

class fROM_VAE_task(nn.Module):
    def __init__(self, input_dim=4, latent_dim=32, num_task_points=256):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, num_task_points=num_task_points)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, joint_cloud):
        # joint_cloud shape: [B, M, 4]
        mu, logvar = self.encoder(joint_cloud)
        z = self.reparameterize(mu, logvar)
        task_cloud_recon = self.decoder(z) # [B, K, 3]
        return task_cloud_recon, mu, logvar

class fROM_VAE_joints(nn.Module):
    def __init__(self, joint_dim=4, latent_dim=32, global_feat_dim=1024, output_limits_dim=8):
        super().__init__()
        self.encoder = Encoder(
            input_dim=joint_dim, 
            global_feat_dim=global_feat_dim, 
            latent_dim=latent_dim
        )
        self.decoder = JointLimitDecoder(
            latent_dim=latent_dim, 
            hidden_dim=global_feat_dim // 4,
            output_dim=output_limits_dim
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, joint_cloud):
        # joint_cloud shape: [B, M, 4]
        mu, logvar = self.encoder(joint_cloud)
        z = self.reparameterize(mu, logvar)
        limits_recon = self.decoder(z) # [B, 8]
        return limits_recon, mu, logvar
