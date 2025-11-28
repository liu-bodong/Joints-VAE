import torch
import torch.nn as nn
from chamferdist import ChamferDistance
# from pytorch3d.loss import chamfer_distance


def vae_loss_function(recon_cloud, input_cloud, mu, logvar, beta=1.0):
    """
    TODO: write this later
    """
    chamfer_dist = ChamferDistance()

    recon_loss = chamfer_dist(input_cloud, recon_cloud, bidirectional=True, point_reduction='mean', batch_reduction='mean')
    recon_loss = recon_loss.mean()
    
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_loss.mean()
    
    total_loss = recon_loss + (beta * kl_loss)
    
    return total_loss, recon_loss, kl_loss 


def flow_matching_loss(model, x_1, kl_weight=0.001):
    """
    x_1: Real joint data batch [B, K, 4]
    """
    device = x_1.device
    B, K, D = x_1.shape
    
    # --- 1. VAE Encoding ---
    # Get the concise profile 'z'
    z, mu, logvar = model(x_1)
    
    # --- 2. Flow Matching Setup ---
    # Sample Noise (Source)
    x_0 = torch.randn_like(x_1)
    
    # Sample Time t [0, 1]
    t = torch.rand(B, device=device)
    
    # Linear Interpolation (Conditional Flow Matching path)
    # x_t = t * x_1 + (1 - t) * x_0
    t_view = t.view(B, 1, 1)
    x_t = t_view * x_1 + (1 - t_view) * x_0
    
    # Target Velocity (Points straight from Noise to Data)
    target_v = x_1 - x_0
    
    # --- 3. Prediction ---
    # Predict velocity at x_t, conditioned on profile z
    pred_v = model.decoder(x_t, t, z)
    
    # --- 4. Loss ---
    # Simple MSE! (Much faster and stable than Chamfer)
    fm_loss = F.mse_loss(pred_v, target_v)
    
    # Add VAE regularization
    # (Use the mean-reduction trick we discussed)
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = fm_loss + (kl_weight * kld_loss)
    
    return total_loss, fm_loss, kld_loss