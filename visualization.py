import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from sklearn.manifold import TSNE

import kinematics

def plot_N_task_point_cloud(point_clouds, limits, rows, marker_size):
    """
    Plots N task space point clouds in 3D scatter plots.

    Args:
        point_clouds (torch.Tensor | np.ndarray): shape (num_user, num_points, 3).
        N (int): Number of point clouds to plot.
        limits (dict): Dictionary with keys 'x', 'y', 'z' containing (min, max) tuples for axis limits.
    """
    N = point_clouds.shape[0]

    fig = plt.figure(figsize=(15, 8))

    for i in range(N):
        ax = fig.add_subplot(rows, N // rows, i + 1, projection='3d')
        pc = point_clouds[i]
        if pc is torch.Tensor:
            pc = point_clouds.cpu().numpy() 
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='seagreen', s=marker_size)
        ax.set_xlim(limits['x'])
        ax.set_ylim(limits['y'])
        ax.set_zlim(limits['z'])
        ax.set_title(f"Point Cloud {i+1}")

    plt.tight_layout()
    plt.show()
    
    
def plot_N_pair_comparison_task_point_cloud(pc_1, pc_2, limits, marker_size):
    N = pc_1.shape[0]
    fig = plt.figure(figsize=(6, 3 * N))
    
    for i in range(N):
        ax = fig.add_subplot(N, 2, 2 * i + 1, projection='3d')
        
        pc1 = pc_1[i]
        if pc1 is torch.Tensor:
            pc1 = pc_1.cpu().numpy() 
        ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c='royalblue', s=marker_size)
        ax.set_xlim(limits['x'])
        ax.set_ylim(limits['y'])
        ax.set_zlim(limits['z'])
        ax.set_title(f"Point Cloud 1 - Sample {i+1}")

        ax = fig.add_subplot(N, 2, 2*i + 2, projection='3d')
        pc2 = pc_2[i]
        if pc2 is torch.Tensor:
            pc2 = pc_2.cpu().numpy() 
        ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], c='brown', s=marker_size)
        ax.set_xlim(limits['x'])
        ax.set_ylim(limits['y'])
        ax.set_zlim(limits['z'])
        ax.set_title(f"Point Cloud 2 - Sample {i+1}")
    
    plt.tight_layout()
    plt.show()




def plot_N_task_2d_slices(point_clouds, limits, marker_size=1):
    """
    Plots N task space point clouds in 2D slices (XY, XZ, YZ).

    Args:
        point_clouds (torch.Tensor | np.ndarray): shape (num_user, num_points, 3).
        N (int): Number of point clouds to plot.
        limits (dict): Dictionary with keys 'x', 'y', 'z' containing (min, max) tuples for axis limits.
    """
    N = point_clouds.shape[0]

    fig, axes = plt.subplots(N, 3, figsize=(10, 3*N))
    
    x_min, x_max = limits['x']
    y_min, y_max = limits['y']
    z_min, z_max = limits['z']

    for i in range(N):
        pc = point_clouds[i]
        if pc is torch.Tensor:
            pc = point_clouds.cpu().numpy()
        
        ax_xy = axes[i, 0]
        ax_xz = axes[i, 1]
        ax_yz = axes[i, 2]
        
        ax_xy.scatter(pc[:, 0], pc[:, 1], c='royalblue', s=marker_size)
        ax_xy.set_xlim(x_min, x_max); ax_xy.set_ylim(y_min, y_max)
        ax_xy.set_xlabel("X"); ax_xy.set_ylabel("Y")
        ax_xy.set_title(f"Sample {i} (XY)")

        ax_xz.scatter(pc[:, 0], pc[:, 2], c='brown', s=marker_size)
        ax_xz.set_xlim(x_min, x_max); ax_xz.set_ylim(z_min, z_max)
        ax_xz.set_xlabel("X"); ax_xz.set_ylabel("Z")
        ax_xz.set_title(f"Sample {i} (XZ)")

        ax_yz.scatter(pc[:, 1], pc[:, 2], c='seagreen', s=marker_size)
        ax_yz.set_xlim(y_min, y_max); ax_yz.set_ylim(z_min, z_max)
        ax_yz.set_xlabel("Y"); ax_yz.set_ylabel("Z")
        ax_yz.set_title(f"Sample {i} (YZ)")
        
    plt.tight_layout()
    plt.show()
    
    
def plot_N_pair_comparison_task_2d_slices(point_clouds_1, point_clouds_2, limits, marker_size):
    N = point_clouds_1.shape[0]
    fig, axes = plt.subplots(N, 6, figsize=(20, 3 * N))
    
    x_min, x_max = limits['x']
    y_min, y_max = limits['y']
    z_min, z_max = limits['z']

    for i in range(N):
        pc1 = point_clouds_1[i]
        pc2 = point_clouds_2[i]

        ax_xy_1 = axes[i, 0]
        ax_xy_2 = axes[i, 3]
        
        ax_xz_1 = axes[i, 1]
        ax_xz_2 = axes[i, 4]
        
        ax_yz_1 = axes[i, 2]
        ax_yz_2 = axes[i, 5]

        ax_xy_1.scatter(pc1[:, 0], pc1[:, 1], c='royalblue', s=0.5)
        ax_xy_1.set_xlim(x_min, x_max); ax_xy_1.set_ylim(y_min, y_max)
        ax_xy_1.set_xlabel("X"); ax_xy_1.set_ylabel("Y")
        ax_xy_1.set_title(f"Sample {i} (XY)")
        
        ax_xy_2.scatter(pc2[:, 0], pc2[:, 1], c='royalblue', s=0.5)
        ax_xy_2.set_xlim(x_min, x_max); ax_xy_2.set_ylim(y_min, y_max)
        ax_xy_2.set_xlabel("X"); ax_xy_2.set_ylabel("Y")
        ax_xy_2.set_title(f"Recon {i} (XY)")

        ax_xz_1.scatter(pc1[:, 0], pc1[:, 2], c='brown', s=0.5)
        ax_xz_1.set_xlim(x_min, x_max); ax_xz_1.set_ylim(z_min, z_max)
        ax_xz_1.set_xlabel("X"); ax_xz_1.set_ylabel("Z")
        ax_xz_1.set_title(f"Sample {i} (XZ)")
        
        ax_xz_2.scatter(pc2[:, 0], pc2[:, 2], c='brown', s=0.5)
        ax_xz_2.set_xlim(x_min, x_max); ax_xz_2.set_ylim(z_min, z_max)
        ax_xz_2.set_xlabel("X"); ax_xz_2.set_ylabel("Z")
        ax_xz_2.set_title(f"Recon {i} (XZ)")

        ax_yz_1.scatter(pc1[:, 1], pc1[:, 2], c='seagreen', s=0.5)
        ax_yz_1.set_xlim(y_min, y_max); ax_yz_1.set_ylim(z_min, z_max)
        ax_yz_1.set_xlabel("Y"); ax_yz_1.set_ylabel("Z")
        ax_yz_1.set_title(f"Sample {i} (YZ)")
        
        ax_yz_2.scatter(pc2[:, 1], pc2[:, 2], c='seagreen', s=0.5)
        ax_yz_2.set_xlim(y_min, y_max); ax_yz_2.set_ylim(z_min, z_max)
        ax_yz_2.set_xlabel("Y"); ax_yz_2.set_ylabel("Z")
        ax_yz_2.set_title(f"Recon {i} (YZ)")

    plt.tight_layout()
    plt.show()
    
    
def plot_joint_pairplot(joint_cloud, marker_size):
    """
    Plots pairplot of joint data.

    Args:
        joint_cloud (tensor or np.ndarray): shape (K, 4).
    """
    joint_names = ['Abduction (Ry)', 'Flexion (Rx)', 'Rotation (Rz)', 'Elbow (Rx)']
    
    if joint_cloud is torch.Tensor:
        joint_cloud = joint_cloud.cpu().numpy()
        
    df = pd.DataFrame(joint_cloud, columns=joint_names)
    sns.pairplot(df, plot_kws={'s': marker_size}, diag_kind='kde', diag_kws={'fill': True})
    plt.show()
    

def plot_N_joint_pairplots(joint_clouds, marker_size):
    """
    Plots N pairplots of joint data.

    Args:
        joint_clouds (tensor or np.ndarray): shape (N, K, 4).
    """
    N = joint_clouds.shape[0]
    joint_names = ['Abduction (Ry)', 'Flexion (Rx)', 'Rotation (Rz)', 'Elbow (Rx)']
    
    # Create a single pairplot containing all users and use a 'user' column as hue so each user has a different color
    if isinstance(joint_clouds, torch.Tensor):
        joint_clouds_np = joint_clouds.cpu().numpy()
    else:
        joint_clouds_np = np.asarray(joint_clouds)

    # joint_clouds_np shape: (N, K, 4)
    all_dfs = []
    for u in range(joint_clouds_np.shape[0]):
        df = pd.DataFrame(joint_clouds_np[u], columns=joint_names)
        df['profile'] = f'profile {u+1}'
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    sns.pairplot(combined_df, hue='profile', plot_kws={'s': marker_size}, diag_kind='kde', diag_kws={'fill': False}, palette='tab10')
    plt.suptitle("Joint Pairplot - All Profiles", y=1.02)
    plt.show()
    
    
    
    
if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "./data/joint_data_sanity_100_4096.npy"
    all_joints = np.load(data_path)  # [num_users, num_points_per_user, 4]
    print(all_joints.shape) 
    selected_joints = all_joints[:10]  # select first 10 users for visualization
    
    # --- conver to task space cloud
    point_clouds = kinematics.forward_kinematics_pytorch(
        torch.tensor(selected_joints, dtype=torch.float16).to(device).view(-1, 4))

    point_clouds = point_clouds.cpu().numpy().reshape(
        selected_joints.shape[0], selected_joints.shape[1], 3)

    # downsample for visualization
    point_clouds = point_clouds[:, :, :]
    
    limits = {'x': [-0.7, 0.7],
              'y': [-0.7, 0.7],
              'z': [-0.7, 0.7]}

    # plot_N_task_point_cloud(point_clouds, limits, rows=2)
    # plot_N_task_2d_slices(point_clouds, limits, marker_size=0.6)
    
    # print(point_clouds[:5].shape, point_clouds[5:10].shape)
    
    # plot_N_pair_comparison_task_point_cloud(point_clouds[:5], point_clouds[5:10], limits, marker_size=1)
    # plot_N_pair_comparison_task_2d_slices(point_clouds[:5], point_clouds[5:10], limits, marker_size=0.6)
    
    # joint_cloud = torch.tensor(selected_joints[1], dtype=torch.float32).view(-1, 4)
    # plot_joint_pairplot(joint_cloud, marker_size=2)
    
    joint_clouds = torch.tensor(selected_joints[:3, :1024, :], dtype=torch.float32)
    plot_N_joint_pairplots(joint_clouds, marker_size=3)