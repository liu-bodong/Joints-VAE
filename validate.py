import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from sklearn.manifold import TSNE


import kinematics
import FK_data_generator
import network
from loss_functions import vae_loss_function
import visualization


def validate(model, val_loader, kl_weight, device):
    avg_total_loss = 0.0
    avg_recon_loss = 0.0
    avg_kl_loss = 0.0
            
    with torch.no_grad():  
        for (x,) in val_loader:
            x = x.to(device)  # shape: (batch_size, num_points_per_user, 4)
            recon_x, mu, logvar = model(x)
            
            loss, recon_loss, kl_loss = vae_loss_function(
                    recon_x,
                    x,
                    mu,
                    logvar,
                    kl_weight
                )

            epoch_total_loss = loss.mean().item()
            epoch_recon_loss = recon_loss.mean().item()
            epoch_kl_loss = kl_loss.mean().item()
            
            avg_total_loss += epoch_total_loss
            avg_recon_loss += epoch_recon_loss
            avg_kl_loss += epoch_kl_loss
            

    num_batches = len(val_loader)
    print(f"Validation Total Loss: {avg_total_loss / num_batches}, \
            Recon Loss: {avg_recon_loss / num_batches}, \
            KL Loss: {avg_kl_loss / num_batches}")   
    

    return avg_total_loss / num_batches, avg_recon_loss / num_batches, avg_kl_loss / num_batches


    # visualization.plot_N_joint_pairplots(joint_clouds[:10], marker_size=3)
    # with torch.no_grad():
    #     joint_clouds = torch.tensor(joint_clouds, dtype=torch.float32)
    #     recon_val_joints, _, _ = model(joint_clouds.to(device))
    # visualization.plot_N_joint_pairplots(recon_val_joints[:10].cpu().numpy(), marker_size=3)  
        

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    LATENT_DIM = 32
    NUM_POINTS_K = 1024
    MODEL_PATH = './models/folding_vae_N4096_K1024_D32_3dball.pth'
    DATA_PATH = './data/sirs_dense_group1979_n1024.npy'
    KL_WEIGHT = 0.0


    model = network.FoldingNetVAE(latent_dim=LATENT_DIM, num_points_k=NUM_POINTS_K).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    joint_clouds = np.load(DATA_PATH)  # [num_users, num_points_per_user, 4]
    dataset = TensorDataset(torch.tensor(joint_clouds).float())
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    validate(model, val_loader, KL_WEIGHT, device)
