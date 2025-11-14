import torch
import torch.nn as nn
import torch.nn.functional as F

class PointEncoder(nn.Module):
    """
    PointNet-style encoder for a D-dimensional point cloud.
    Takes [B, N * K, D] -> [B, global_feature_dim]
    """
    def __init__(self, input_dims=4, global_feature_dim=1024):
        super(PointEncoder, self).__init__()
        self.global_feature_dim = global_feature_dim
        
        # one data is one entire point cloud of K points in R^K
        self.mlp = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.global_feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        point_features = self.mlp(x) # x: [B, K, 4] -> [B, K, 1024]
        global_feature, _ = torch.max(point_features, dim=1) # [B, 1024]
        return global_feature


class PointDecoder(nn.Module):
    """
    MLP-based decoder.
    Takes [B, latent_dim] -> [B, K, D]
    """
    def __init__(self, latent_dim, num_points_k, output_dims=4):
        super(PointDecoder, self).__init__()
        self.num_points_k = num_points_k
        self.output_dims = output_dims
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            # The final layer outputs all K*4 point coordinates at once
            nn.Linear(1024, self.num_points_k * self.output_dims)
        )

    def forward(self, z):
        # z shape: [B, latent_dim]
        
        # 1. Pass latent vector through the MLP
        flat_point_cloud = self.mlp(z) # Output: [B, K * 4]
        
        # 2. Reshape to the final point cloud format
        # Output shape: [B, K, 4]
        recon_cloud = flat_point_cloud.view(-1, self.num_points_k, self.output_dims)
        
        return recon_cloud


class PointVAE(nn.Module):
    """
    point cloud -> Encoder -> latent (mu, logvar) -> reparameterization (z) -> Decoder -> reconstructed point cloud
    """
    def __init__(self, latent_dim, num_points_k, global_feature_dim=1024):
        super(PointVAE, self).__init__()
        
        self.encoder = PointEncoder(input_dims=4, global_feature_dim=global_feature_dim)
        self.decoder = PointDecoder(latent_dim=latent_dim, num_points_k=num_points_k)
        
        # VAE-specific layers: map global feature to mu and logvar
        self.fc_mu = nn.Linear(global_feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(global_feature_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        """
        The reparameterization trick (z = mu + epsilon * std).
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x):
        # x shape: [B, K, 4]
        
        # 1. Encode the cloud to a global feature
        
        global_feat = self.encoder(x) # [B, global_feature_dim]
        
        # 2. Get latent space parameters
        
        mu = self.fc_mu(global_feat) # mu, logvar shapes: [B, latent_dim]
        logvar = self.fc_logvar(global_feat)
        
        # 3. Sample from the latent distribution
        z = self.reparameterize(mu, logvar) # z shape: [B, latent_dim]
        
        # 4. Decode the latent vector back into a point cloud
        
        recon_cloud = self.decoder(z) # recon_cloud shape: [B, K, 4]
        
        return recon_cloud, mu, logvar


# --- Example Usage ---
if __name__ == "__main__":
    
    # --- Define model hyperparameters ---
    LATENT_DIM = 32         # Size of the latent vector z
    NUM_POINTS_K = 4096     # Number of points per user (must match data)
    BATCH_SIZE = 8          # Number of point clouds/profiles/users in a batch
    
    # --- Create a dummy input batch ---
    # This simulates one batch from DataLoader
    # (B, K, 4) in rads
    dummy_input_cloud = torch.rand(BATCH_SIZE, NUM_POINTS_K, 4)
    print(f"Input batch shape: {dummy_input_cloud.shape}")
    
    # --- Instantiate the VAE ---
    vae = PointVAE(latent_dim=LATENT_DIM, num_points_k=NUM_POINTS_K)
    print("\nVAE Model Instantiated:")
    print(vae)

    # --- Run a forward pass ---
    recon_cloud, mu, logvar = vae(dummy_input_cloud)
    
    # --- Check output shapes ---
    print("\n--- Forward Pass Check ---")
    print(f"Reconstructed cloud shape: {recon_cloud.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"LogVar shape: {logvar.shape}")
    
    # Verify shapes
    assert recon_cloud.shape == (BATCH_SIZE, NUM_POINTS_K, 4)
    assert mu.shape == (BATCH_SIZE, LATENT_DIM)
    assert logvar.shape == (BATCH_SIZE, LATENT_DIM)
    
    print("\nAll shapes are correct!")