
import torch

def chamfer_loss(cloud1, cloud2):
    
    """
    Calculates the Chamfer Distance between two point clouds in pure PyTorch.
    
    Args:
        cloud1 (torch.Tensor): (B, N, 3) tensor of the first point cloud.
        cloud2 (torch.Tensor): (B, M, 3) tensor of the second point cloud.

    Returns:
        torch.Tensor: The Chamfer loss.
    """
    
    cloud1 = cloud1.float()
    cloud2 = cloud2.float()

    # Calculate the pairwise distance matrix
    # p1_dists will be [B, N, M], where p1_dists[i, j, k] is the distance
    # between the j-th point in the i-th batch of cloud1 and the
    # k-th point in the i-th batch of cloud2.
    p1_dists = torch.cdist(cloud1, cloud2)

    # For each point in cloud1, find the minimum distance to any point in cloud2
    dist_c1_to_c2 = p1_dists.min(dim=2)[0] #[B, N]
    
    # For each point in cloud2, find the minimum distance to any point in cloud1
    # dist_c2_to_c1 will be (B, M)
    dist_c2_to_c1 = p1_dists.min(dim=1)[0]

    # avg the distances
    # loss_c1 is the mean dist from cloud1 to cloud2
    # loss_c2 is the mean dist from cloud2 to cloud1
    loss_c1 = dist_c1_to_c2.mean(dim=1)
    loss_c2 = dist_c2_to_c1.mean(dim=1)

    # The final Chamfer loss is the sum of these two, mean over the batch
    loss = (loss_c1 + loss_c2).mean()
    
    return loss