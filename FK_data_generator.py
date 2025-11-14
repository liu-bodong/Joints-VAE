
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

import kinematics

"""
Data generation utilities for arm joint configurations and task space points.
Pipeline:
1. Sample joint space configurations uniformly within joint limits.
2. Compute forward kinematics to get task space points.
3. Filter out dense clusters in task space to obtain uniform coverage.
4. Return filtered joint configurations and task space points.
"""

# TODO: some code use this, and some other code takes in args. They are not the same so it works. 
#       in future should be unified 
BASE_LIMITS_DEG = np.array([
    [-180.0, 20.0],  # Abd (Ry)
    [-60.0, 180.0],  # Flex (Rx)
    [-60.0, 90.0],   # Rot (Rz)
    [0.0, 150.0]     # Elbow (Rx)
])

# def sample_joint_space_config(user_joint_limits, num_samples=12):
#     """Generate uniform joint space configurations for all users based on their joint limits.
#     """
#     all_user_grids = []
#     for user_limits in all_user_joint_limits:
#         joint_ranges = [np.linspace(lim[0], lim[1], num_samples) for lim in user_limits]
#         joint_grid = np.array(list(itertools.product(*joint_ranges)))
#         all_user_grids.append(joint_grid)
#     return all_user_grids


def generate_random_user_limits():
    """
    Generates a single set of random joint limits for a "new user"
    by sampling a random sub-range from within the base healthy limits.
    
    The sampling would ensure that zero pose is always reachable.
    
    Returns:
        np.ndarray: A [4, 2] array of [min, max] limits for a new user.
    """
    limits_deg = np.zeros((4, 2))
    
    for i in range(4):
        healthy_min, healthy_max = BASE_LIMITS_DEG[i]
        
        # Sample two random points within the healthy range
        
        if i != 3:  
            val1 = np.random.uniform(healthy_min, -15)
        else: # Elbow joint must include 0
            val1 = 0.0
            
        if i == 0: # Abd joint has lower max limit
            val2 = np.random.uniform(5, healthy_max)
        else:
            val2 = np.random.uniform(15, healthy_max)
        
        # The new limits are the min and max of these two points
        limits_deg[i, 0] = min(val1, val2)
        limits_deg[i, 1] = max(val1, val2)
        
    return limits_deg


def sample_joint_space_randomly(user_limits_deg, num_samples):
    """
    Generates a fixed-size point cloud of joint configurations
        
    Args:
        user_limits (np.ndarray): Shape [4, 2] of [min, max]
        num_samples (int): How many 4D points to generate (e.g., 4096)
        
    Returns:
        np.ndarray: A [num_samples, 4] array of joint angles (in degrees).
    """
    # Get the min and span for each of the 4 joints
    user_mins = user_limits_deg[:, 0]              # Shape [4,]
    user_maxs = user_limits_deg[:, 1]              # Shape [4,]
    
    joint_cloud_deg = np.random.uniform(low=user_mins, high=user_maxs, size=(num_samples, 4))
    
    return joint_cloud_deg


def reject_dense_cluster(joint_poses, task_points, workspace_min, workspace_max, epsilon):
    """
    Filters a point cloud by keeping only one point per occupied
    voxel in the task space.

    Args:
        joint_poses (np.ndarray): The input joint poses [N^4, 4].
        task_points (np.ndarray): The corresponding task-space points [N^4, 3].
        workspace_min (np.ndarray): The minimum coordinates of the workspace in each dimension [3].
        workspace_max (np.ndarray): The maximum coordinates of the workspace in each dimension [3].
        epsilon (float): The side length of each voxel cube (in meters).

    Returns:
        (np.ndarray, np.ndarray):
            - filtered_joint_poses: The kept joint poses [M_filtered, 4].
            - filtered_task_points: The kept task-space points [M_filtered, 3].
    """
    
    workspace_span = workspace_max - workspace_min

    # 1. normalize points from world space [min, max] to [0, 1]
    normalized_points = (task_points - workspace_min) / workspace_span

    # 2. clip AFTER norm
    normalized_points = np.clip(normalized_points, 0.0, 1.0 - 1e-6) # 1e-6 to avoid index 32

    # 3. divide by eps to get the number of bins
    grid_size = (workspace_span / epsilon).astype(int) + 1

    indices = (normalized_points * grid_size).astype(int)

    df = pd.DataFrame(indices, columns=['vx', 'vy', 'vz'])
    df['original_index'] = np.arange(len(df))
    filtered_indices = df.groupby(['vx', 'vy', 'vz'])['original_index'].first().values

    return joint_poses[filtered_indices], task_points[filtered_indices]



def clip_to_k_points(joint_cloud, k):
    """
    Downsample a point cloud of size M to a fixed size K.
    
    We do the downsampling in the task space to ensure uniform converage.
    Based on the filtered points, we correspondingly select the joint configurations.
    
    Args:
        joint_cloud (np.ndarray): Shape [M, 4] (variable M)
        k (int): The target number of points (e.g., 4096)
        
    Returns:
        clipped_joint_cloud (np.ndarray): Shape [k, 4]
    """
    current_num_points = joint_cloud.shape[0]
    
    if current_num_points == k:
        return joint_cloud
    
    if current_num_points > k:
        # M > K: clip 
        indices = np.random.choice(current_num_points, k, replace=False)
        return joint_cloud[indices]
    
    if current_num_points < k:
        # M < K: Throw this case away, return None
        return None


def generate_data(num_users, num_points_per_user, epsilon, workspace_min, workspace_max, device):
    """Generate data for all users based on their joint limits.
    Args:
        all_user_joint_limits (np.ndarray): Joint limits for all users.
        num_samples_per_joint (int): Number of samples per joint.
        grid_size (int): Size of the voxel grid along each dimension.
    Returns:
        list: List of joint poses for all users.
    """
        
    # all_points = []
    all_joints = []
    
    # 2. Loop through each user
    for i in tqdm(range(num_users), desc=f"Generating {num_points_per_user} points for each of {num_users} users"):
        # user_joint_grid [N^4, 4]
        user_limits_deg = generate_random_user_limits()
        # print(user_limits)
        
        joint_pool_deg = sample_joint_space_randomly(user_limits_deg, num_points_per_user * 4)
        joint_pool_rad = np.deg2rad(joint_pool_deg)
        
        joint_poses_tensor_rad = torch.tensor(joint_pool_rad, dtype=torch.float32).to(device)
        
        # 3. Run FK
        with torch.no_grad():
            task_pool_tensor = kinematics.forward_kinematics_pytorch(joint_poses_tensor_rad) # [N^4, 3]
            task_pool_np = task_pool_tensor.cpu().numpy()
                
        # 4. filter out dense points   
        filtered_joints, filtered_points = reject_dense_cluster(
            joint_poses   = joint_pool_rad,
            task_points   = task_pool_np,
            workspace_min = workspace_min,
            workspace_max = workspace_max,
            epsilon       = epsilon
        )
        
        # normaliz
        # # sampled_k_joints = normalize_to_k_points(filtered_joints, num_points_per_user)
        # This normalization trick is done in joint space which causes task space distribution
        # point cloud to concentrate densely towards the center. DO NOT USE
        
        # 5 Clip to K points
        clipped_joints = clip_to_k_points(
            joint_cloud=filtered_joints,
            k=num_points_per_user
        )
        
        
        # all_points.append(torch.tensor(filtered_points))
        all_joints.append(clipped_joints) if clipped_joints is not None else None
        
        # print(f"Processed user {i+1}/{num_users}. ")

    return all_joints


def get_dataset(num_users, num_points_per_user, epsilon, workspace_min, workspace_max, device):
    all_joints = generate_data(
        num_users=num_users,
        num_points_per_user=num_points_per_user,
        epsilon=epsilon,
        workspace_min=workspace_min,
        workspace_max=workspace_max,
        device=device
    )
    
    all_joints = np.stack(all_joints, axis=0, dtype=np.float32)  # Shape [num_users, num_points_per_user, 4]
    all_joints_tensor = torch.tensor(all_joints, dtype=torch.float32)
    dataset = TensorDataset(all_joints_tensor)
    return dataset



# Main function for generating np data and saving to disk
if __name__ == "__main__":
    NUM_PROFILES = 1000
    POINT_CLOUD_SIZE = 2048 
    EPSILON = 0.02  # meters
    
    WORKSPACE_MIN = np.array([-0.7, -0.7, -0.7])
    WORKSPACE_MAX = np.array([ 0.7,  0.7,  0.7])
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    PATH = f"./data/joint_data_{NUM_PROFILES}_{POINT_CLOUD_SIZE}.npy"
    
    all_joints = generate_data(
        num_users=NUM_PROFILES,
        num_points_per_user=POINT_CLOUD_SIZE,
        epsilon=EPSILON,
        workspace_min=WORKSPACE_MIN,
        workspace_max=WORKSPACE_MAX,
        device=DEVICE
    )
    
    # for i in range(len(all_joints)):
    #     print(f"User {i+1} joint data shape: {all_joints[i].shape}")
    
    all_joints = np.stack(all_joints, axis=0, dtype=np.float32)  # Shape [num_users, num_points_per_user, 4]
    
    print("Data generation complete.")
    print("Shape: ", all_joints.shape)  # Should be [4096, 4] for each user
    
    
    # Save to disk 
    np.save(PATH, all_joints)
    print(f"Saved to {PATH}")