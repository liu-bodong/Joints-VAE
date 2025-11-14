import torch
import torch.nn as nn
import kaolin.metrics.pointcloud as kaolin_pc
# from pytorch3d.loss import chamfer_distance


# --- KAOLIN ---
def vae_loss_function(a, b, mu, logvar, beta=1.0):
    """
    Calculates the combined VAE loss (using Kaolin).
    """
    
    recon_loss = kaolin_pc.chamfer_distance(a, b)
    recon_loss = recon_loss.mean()
        
    kl_loss  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_loss.mean()
    
    total_loss = recon_loss + (beta * kl_loss)
    
    return total_loss, recon_loss, kl_loss


# --- CUSTOM chamfer distance ---
def custom_reconstruction_loss(input_cloud, recon_cloud):
    """
    Calculates the Chamfer Distance in pure PyTorch.
    
    Args:
        input_cloud (torch.Tensor): The original cloud, shape [B, K, 4]
        recon_cloud (torch.Tensor): The reconstructed cloud, shape [B, K, 4]
        
    Returns:
        torch.Tensor: A single loss value.
    """
    
    # 1. Calculate the pairwise distance matrix
    # This is the O(K^2) step and is memory-intensive!
    # dist_matrix shape: [B, K, K]
    # dist_matrix[b, i, j] = distance from point i in input to point j in recon
    dist_matrix = torch.cdist(input_cloud, recon_cloud, p=2.0) # p=2.0 for L2 distance
    
    # 2. Find nearest neighbors (A -> B)
    # For each point in the input, find its closest point in the recon
    # .values -> just get the distances, not the indices
    # min_dists_a_to_b shape: [B, K]
    min_dists_a_to_b, _ = torch.min(dist_matrix, dim=2)
    
    # 3. Find nearest neighbors (B -> A)
    # For each point in the recon, find its closest point in the input
    # min_dists_b_to_a shape: [B, K]
    min_dists_b_to_a, _ = torch.min(dist_matrix, dim=1)

    # 4. Sum the distances and take the mean
    # We take the mean across the batch and the points
    loss_a_to_b = min_dists_a_to_b.mean()
    loss_b_to_a = min_dists_b_to_a.mean()
    
    # The final Chamfer distance is the sum of both terms
    chamfer_loss = loss_a_to_b + loss_b_to_a
    
    return chamfer_loss

def vae_loss_function(recon_cloud, input_cloud, mu, logvar, beta=1.0):
    
    # Use our new pure-pytorch function!
    recon_loss = custom_reconstruction_loss(input_cloud, recon_cloud)
    
    kld_loss = kl_loss = kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_loss.mean()
    
    total_loss = recon_loss + (beta * kld_loss)
    
    return total_loss, recon_loss, kld_loss




# --- BELOW is pytorch3d version ---

# def reconstruction_loss(input_cloud, recon_cloud):
#     """
#     Calculates the Chamfer Distance between two point clouds.
    
#     Args:
#         input_cloud (torch.Tensor): The original cloud, shape [B, K, 4]
#         recon_cloud (torch.Tensor): The reconstructed cloud, shape [B, K, 4]
        
#     Returns:
#         torch.Tensor: A single loss value.
#     """
#     # The pytorch3d chamfer_distance function returns the loss
#     # and (optionally) the nearest-neighbor indices. We just need the loss.
#     # It handles batches automatically.
    
#     # NOTE: It's crucial that both tensors are on the same device (e.g., 'cuda')
#     chamfer_loss, _ = chamfer_distance(input_cloud, recon_cloud)
    
#     return chamfer_loss




# def vae_loss_function(recon_cloud, input_cloud, mu, logvar, beta=1.0):
#     """
#     Calculates the combined VAE loss.
#     loss = Reconstruction_Loss + beta * KL_Divergence_Loss
    
#     'beta' is a hyperparameter to balance the two losses.
#     """
    
#     # 1. Reconstruction Loss (how well we rebuilt the cloud)
#     recon_loss = reconstruction_loss(input_cloud, recon_cloud)
    
#     # 2. KL Divergence (how "regularized" the latent space is)
#     kld_loss = kl_divergence_loss(mu, logvar)
    
#     # 3. Total Loss
#     total_loss = recon_loss + (beta * kld_loss)
    
#     return total_loss, recon_loss, kld_loss