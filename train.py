import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import tqdm

import network


N_USERS = 10
M_JOINT_POINTS = 256  # Number of points to sample joint-space
K_TASK_POINTS = 256   # Number of points to generate in task-space
LATENT_DIM = 32
LEARNING_RATE = 1e-4
EPOCHS = 200
BATCH_SIZE = 4
BETA_KL = 0.001 # Weight for KL loss

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# read in user joint limits from CSV
joint_limits_df = pd.read_csv("joint_limits.csv")
joint_limits = joint_limits_df.iloc[:, 1:].values

# read in user joint configs from CSV
joint_configs_df = pd.read_csv("dataset_elbow_bent.csv")
joint_configs = joint_configs_df.iloc[:, 1:].values

# 3. Create DataLoader
dataset = TensorDataset(joint_configs, joint_limits)
# BATCH_SIZE here refers to batch of patients
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# joint_clouds shape: [N, M, 4], joint_limits shape: [N, 8]

# 4. Initialize Model
# model = network.fROM_VAE_task(joint_dim=4, latent_dim=LATENT_DIM, num_joint_points=M_JOINT_POINTS, num_task_points=K_TASK_POINTS).to(device)

model = network.fROM_VAE_joints(joint_dim=4, latent_dim=LATENT_DIM, global_feat_dim=1024, output_limits_dim=8).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
recon_loss_fn = nn.MSELoss(reduction='sum')
# chamfer_loss_fn = ChamferDistance() #

# tain
for epoch in range(EPOCHS):
    model.train()
    total_loss_epoch = 0
    total_recon_loss_epoch = 0
    total_kl_loss_epoch = 0
    
    for p_joint_batch, p_limits_batch in dataloader:
        p_joint_batch = p_joint_batch.to(device)     # [B, M, 4]
        p_limits_batch = p_limits_batch.to(device) # [B, 8]
        
        # Forward pass
        limits_recon, mu, logvar = model(p_joint_batch) # [B, 8], [B, D_z], [B, D_z]
        
        # Calculate Loss
        # 1. Reconstruction Loss (MSE on the 8 limit values)
        recon_loss = recon_loss_fn(limits_recon, p_limits_batch)
        
        # 2. KL Loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 3. Total Loss
        # Normalize losses by batch size to make BETA_KL more independent of batch size
        loss = (recon_loss / BATCH_SIZE) + (BETA_KL * kl_loss / BATCH_SIZE)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss_epoch += loss.item() * BATCH_SIZE
        total_recon_loss_epoch += recon_loss.item()
        total_kl_loss_epoch += kl_loss.item()

    # Print epoch stats
    avg_loss = total_loss_epoch / len(dataset)
    avg_recon = total_recon_loss_epoch / len(dataset)
    avg_kl = total_kl_loss_epoch / len(dataset)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, "
              f"Recon Loss (MSE): {avg_recon:.4f}, KL Loss: {avg_kl:.4f}")

# Training Loop
# print(f"Starting VAE training with {N_PATIENTS} patients...")
# for epoch in range(EPOCHS):
#     model.train()
#     total_loss_epoch = 0
#     total_recon_loss_epoch = 0
#     total_kl_loss_epoch = 0
    
#     for p_joint_batch, p_task_batch in dataloader:
#         p_joint_batch = p_joint_batch.to(device)
#         p_task_batch = p_task_batch.to(device)
        
#         # Forward pass
#         p_task_recon, mu, logvar = model(p_joint_batch)
        
#         # Calculate Loss
#         # 1. KL Loss
#         kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         kl_loss = kl_loss / (BATCH_SIZE * K_TASK_POINTS * 3) # Average over batch & points
        
#         # 2. Reconstruction Loss (Chamfer)
#         # pytorch-chamfer
#         dist1, dist2, _, _ = chamfer_loss_fn(p_task_recon, p_task_batch)
#         recon_loss = (torch.mean(dist1)) + (torch.mean(dist2))
#         # # pytorch3d
#         # recon_loss, _ = chamfer_distance(p_task_recon, p_task_batch)

#         # 3. Total Loss
#         loss = recon_loss + BETA_KL * kl_loss
        
#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         total_loss_epoch += loss.item()
#         total_recon_loss_epoch += recon_loss.item()
#         total_kl_loss_epoch += kl_loss.item()

#     # Print epoch stats
#     avg_loss = total_loss_epoch / len(dataloader)
#     avg_recon = total_recon_loss_epoch / len(dataloader)
#     avg_kl = total_kl_loss_epoch / len(dataloader)
    
#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}, "
            #   f"Recon Loss (Chamfer): {avg_recon:.6f}, KL Loss: {avg_kl:.6f}")

print("Training finished.")

# --- Step 3: Evaluation Example ---
# model.eval()
# with torch.no_grad():
#     # Get the fROM for the first patient in the dataset
#     joint_cloud_sample = joint_clouds[0].unsqueeze(0).to(device) # [1, M, 4]
    
#     # Generate the task-space reconstruction
#     task_cloud_recon, _, _ = model(joint_cloud_sample) # [1, K, 3]
    
#     print(f"\nGenerated task cloud shape: {task_cloud_recon.shape}")
#     # You would now visualize this point cloud (task_cloud_recon)
#     # using a library like Open3D or matplotlib.