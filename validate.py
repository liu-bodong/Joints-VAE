import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from sklearn.manifold import TSNE


import network
from loss_functions import vae_loss_function
import visualization


def validate(model, val_loader, kl_weight, device):
    """
    Runs evaluation on the validation dataset.

    Args:
        model (_type_): _description_
        val_loader (_type_): _description_
        kl_weight (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.eval()
    total_val_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    progress_bar = tqdm.tqdm(val_loader, desc="Validation", leave=False)
            
    with torch.no_grad():
        for (input_cloud,) in progress_bar:
            input_cloud = input_cloud.to(device)  # [batch_size, num_points_per_user, 4]
            
            recon_cloud, mu, logvar = model(input_cloud)
            
            total_loss, recon_loss, kl_loss = vae_loss_function(
                    recon_cloud,
                    input_cloud,
                    mu,
                    logvar,
                    kl_weight
                )

            total_val_loss += total_loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
    num_batches = len(val_loader)
    avg_total = total_val_loss / num_batches
    avg_recon = total_recon_loss / num_batches
    avg_kl = total_kl_loss / num_batches
    
    progress_bar.set_postfix({
        "Val Total Loss": avg_total,
        "Val Recon Loss": avg_recon,
        "Val KL Loss": avg_kl
    })
    
    return avg_total, avg_recon, avg_kl

 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    LATENT_DIM = 32
    NUM_POINTS_K = 1024
    MODEL_PATH = './models/folding_vae_N4096_K1024_D32_3dball.pth'
    DATA_PATH = './data/sirs_dense_group1979_n1024.npy'
    KL_WEIGHT = 0.0


    model = network.FoldingNetV1(latent_dim=LATENT_DIM, num_points_k=NUM_POINTS_K).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    joint_clouds = np.load(DATA_PATH)  # [num_users, num_points_per_user, 4]
    dataset = TensorDataset(torch.tensor(joint_clouds).float())
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    validate(model, val_loader, KL_WEIGHT, device)
