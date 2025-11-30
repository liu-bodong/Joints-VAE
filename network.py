import torch
import torch.nn as nn
import torch.nn.functional as F

# --- FoldingVAE

class FoldingNetEncoder(nn.Module):
    """
    FoldingNet-style encoder for a D-dimensional point cloud.
    Takes [B, K, n] -> mu, logvar of shape [B, z_dim]
    
    For latent vector, first half is mu, second half is logvar.
    Each has shape [B, z_dim]
    
    n -> 64 -> 1024 -> 1024 ->512
    """
    def __init__(self, input_dim=4, z_dim=16):
        super(FoldingNetEncoder, self).__init__()

        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.BatchNorm1d(1024)
        )
        
        
        
        # maps global feature to mu and logvar
        self.fc4 = nn.Linear(1024, z_dim * 2)

    def forward(self, x):
        B, K, _ = x.shape # x shape: [B, K, 4]
        
        # change x shape from [B, K, 4] to [B, 4, K] for BatchNorm1d
        x = x.transpose(2, 1)
        
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x) # [B, 1024, K]
        
        # transpose back to [B, K, z_dim]
        x = x.transpose(2, 1)
        
        global_feat, _ = torch.max(x, dim=1) # [B, z_dim]
        
        # mu + logvar
        combined = self.fc4(global_feat) # [B, z_dim*2]
        
        mu, logvar = torch.chunk(combined, chunks=2, dim=1) # each [B, z_dim]
        
        return mu, logvar



class FoldingNetDecoderV2(nn.Module):
    def __init__(self, z_dim=16, output_dim=4):
        super(FoldingNetDecoderV2, self).__init__()
        self.z_dim = z_dim
        
        # Global Shape Network ---
        # 4 Scales and 4 Centers.
        self.affine_head = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.SiLU(),
            nn.Linear(64, output_dim * 2) # Outputs [Scale(4), Center(4)]
        )
        
        # Detail Network ---
        # This adds the curvature and cuts holes.
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + 4, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, z, num_points):
        B = z.shape[0]
        
        primitive = torch.rand(B, num_points, 4, device=z.device) * 2 - 1
        
        # Get Scale and Center from Z
        affine_params = self.affine_head(z) # [B, 8]
        scale = affine_params[:, :4].unsqueeze(1)  # [B, 1, 4]
        center = affine_params[:, 4:].unsqueeze(1) # [B, 1, 4]
        
        base_shape = primitive * torch.exp(scale) + center
        
        z_expanded = z.unsqueeze(1).expand(B, num_points, -1)
        cat_input = torch.cat([base_shape, z_expanded], dim=2)
        
        residual = self.mlp(cat_input)
        
        return base_shape + residual # [B, K, n]


class FoldingNetDecoderV1(nn.Module):
    """
    FoldingNet decoder. But the paper is 3d
    Takes [B, z_dim] -> [B, K, n]
    
    """
    def __init__(self, z_dim=256, num_points_k=1024, output_dim=4, primitive_type='4d'):
        super(FoldingNetDecoderV1, self).__init__()
        self.num_points_k = num_points_k
        self.z_dim = z_dim
        
        # Original 3d "paper" primitive for building
        self.register_buffer('primitive', self._build_primitive(num_points_k))
        
        # First fold
        self.fold1 = nn.Sequential(
            nn.Linear(z_dim + 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4) # output intermediate 4d coords
        )
        
        # Second fold
        self.fold2 = nn.Sequential(
            nn.Linear(z_dim + 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim) # output final n-dim coords (3d))
        )
        
        # self.fold3 = nn.Sequential(
        #     nn.Linear(z_dim + 4, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, output_dim) # output final n-dim coords (3d))
        # )
        
    def _build_primitive(self, k):
        """
        Generates points on/in a random unit 3D sphere.
        """
        # 4D Hypercube
        
        # hypercube by uniform sampling
        points = torch.rand(k, 4) * 2 - 1  
        
        # hypercube by fixed grid
        # x = torch.linspace(-1, 1, int(k**0.25))
        # grid_x, grid_y, grid_z, grid_w = torch.meshgrid(x, x, x, x, indexing='ij')
        # points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1), grid_w.reshape(-1)], dim=1)
        
        # 4d Hypersphere
        # points = torch.randn(k, 4)
        # points = F.normalize(points, p=2, dim=1)  # normalize to unit hypersphere
        
        # 3d Sphere
        # points = torch.randn(k, 3)
        # points = F.normalize(points, p=2, dim=1)  # normalize to unit sphere
        
        # 3d grid
        # points = torch.rand(k, 3) * 2 - 1
        
        return points

    def forward(self, z):
        # z: [B, z_dim]
        B = z.shape[0] 
        K = self.num_points_k
        
        # 1. replicate latent vector z for every point 
        z_expanded = z.unsqueeze(1).expand(B, K, -1) # [B, K, z_dim]
        
        # 2. Replicate primitive for batch
        # [B, K, 4]
        # for 4d primitive
        primitive_expanded = self.primitive.unsqueeze(0).expand(B, K, -1)
        
        # --- FOLD 1
        # cat [Grid, z] -> [B, K, 3 + z]
        cat1 = torch.cat([primitive_expanded, z_expanded], dim=2)
        folding1_out = self.fold1(cat1)
        
        # --- FOLD 2
        # cat [Fold1_Output, z] -> [B, K, 3 + z]
        cat2 = torch.cat([folding1_out, z_expanded], dim=2)
        folding2_out = self.fold2(cat2)
        
        
        # cat3 = torch.cat([folding2_out, z_expanded], dim=2)
        # folding3_out = self.fold3(cat3)  # [B, K, n]
        
        return folding2_out #, folding2_out
    
    
class FoldingNetV1(nn.Module):
    def __init__(self, latent_dim=16, num_points_k=1024):
        super(FoldingNetV1, self).__init__()
        self.num_points_k = num_points_k
        self.encoder = FoldingNetEncoder(z_dim=latent_dim)
        self.decoder = FoldingNetDecoderV1(z_dim=latent_dim, num_points_k=num_points_k, output_dim=4)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_cloud_1= self.decoder(z)
        
        return recon_cloud_1, mu, logvar
    

class FoldingNetV2(nn.Module):
    def __init__(self, latent_dim=16, num_points_k=1024):
        super(FoldingNetV2, self).__init__()
        self.num_points_k = num_points_k
        self.encoder = FoldingNetEncoder(z_dim=latent_dim)
        self.decoder = FoldingNetDecoderV2(z_dim=latent_dim, output_dim=4)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_cloud = self.decoder(z, self.num_points_k)
        
        # print(recon_cloud.shape, x.shape)s
        return recon_cloud, mu, logvar









# --- OLD ---
# --- PointNet Encoder + MLP Decoder for Point Cloud VAE ---
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
    LATENT_DIM = 16         # Size of the latent vector z
    NUM_POINTS_K = 4096     # Number of points per user (must match data)
    BATCH_SIZE = 256          # Number of point clouds/profiles/users in a batch
    
    # --- Create a dummy input batch ---
    # This simulates one batch from DataLoader
    # (B, K, 4) in rads
    dummy_input_cloud = torch.rand(BATCH_SIZE, NUM_POINTS_K, 4)
    print(f"Input batch shape: {dummy_input_cloud.shape}")
    
    # --- Instantiate the VAE ---
    vae = FoldingNetV1(latent_dim=LATENT_DIM, num_points_k=NUM_POINTS_K).to(torch.device("cpu"))
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
    
    total_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters in the VAE: {total_params}")
    
    total_memory = sum(p.element_size() * p.nelement() for p in vae.parameters())
    print(f"Total memory usage for parameters: {total_memory / (1024 **2):.2f} MB")