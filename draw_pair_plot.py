import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from sklearn.manifold import TSNE
import sys

import kinematics
import FK_data_generator
import network
from loss_functions import vae_loss_function
import visualization


def draw(latent_dim, k_num_points, model_path, val_data, count=1):   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    model = network.FoldingNetV1(latent_dim=latent_dim, num_points_k=k_num_points).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    visualization.plot_N_joint_pairplots(val_data[0:count], marker_size=3)
    with torch.no_grad():
        recon_cloud_1, _, _ = model(torch.tensor(val_data[0:count], dtype=torch.float32).to(device))
    visualization.plot_N_joint_pairplots(recon_cloud_1.cpu().numpy(), marker_size=3)
    # visualization.plot_N_joint_pairplots(recon_cloud_2.cpu().numpy(), marker_size=3)
    
    
if __name__ == "__main__":
    MODEL_PATH = './runs/4d_cube_N2_K4096_D8_KL0/4d_cube_N2_K4096_D8_KL0.pth'
    DATA_PATH = 'data\sirs_dense_N10_K4096.npy'
    LATENT_DIM = 8
    K_NUM_POINTS = 4096
    COUNT = 1
    
    args = sys.argv[1:]
    model_path = sys.argv[1] if len(args) > 0 else MODEL_PATH
    data_path = sys.argv[2] if len(args) > 1 else DATA_PATH
    latent_dim = int(sys.argv[3]) if len(args) > 2 else LATENT_DIM
    k_num_points = int(sys.argv[4]) if len(args) > 3 else K_NUM_POINTS
    count = int(sys.argv[5]) if len(args) > 4 else COUNT
    
    draw(latent_dim, k_num_points, model_path, data_path, count)