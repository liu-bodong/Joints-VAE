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

# --- Define arm constants based on your URDF ---
UPPER_ARM_LEN = 0.33
FOREARM_LEN = 0.28


def build_rotation_matrix(angle_batch, axis_vector, device):
    """
    Builds a batch of 4x4 homogeneous rotation matrices.

    Args:
        angle_batch (torch.Tensor): A 1D tensor of angles, shape [B].
        axis_vector (torch.Tensor): A 1D tensor for the axis, shape [3] (e.g., [1., 0., 0.]).
        device (torch.device): The device (e.g., 'cuda') to create tensors on.

    Returns:
        torch.Tensor: A batch of 4x4 rotation matrices, shape [B, 4, 4].
    """
    batch_size = angle_batch.shape[0]
    
    # Ensure axis is a unit vector
    axis = axis_vector.to(device)
    
    # Get trig components
    cos_a = torch.cos(angle_batch)
    sin_a = torch.sin(angle_batch)
    versin_a = 1.0 - cos_a # 1 - cos(a)
    
    # Rodrigues' rotation formula
    x, y, z = axis[0], axis[1], axis[2]
    rot_mat_3x3 = torch.zeros(batch_size, 3, 3, device=device)
    
    rot_mat_3x3[:, 0, 0] = cos_a + x*x*versin_a
    rot_mat_3x3[:, 0, 1] = x*y*versin_a - z*sin_a
    rot_mat_3x3[:, 0, 2] = x*z*versin_a + y*sin_a
    rot_mat_3x3[:, 1, 0] = y*x*versin_a + z*sin_a
    rot_mat_3x3[:, 1, 1] = cos_a + y*y*versin_a
    rot_mat_3x3[:, 1, 2] = y*z*versin_a - x*sin_a
    rot_mat_3x3[:, 2, 0] = z*x*versin_a - y*sin_a
    rot_mat_3x3[:, 2, 1] = z*y*versin_a + x*sin_a
    rot_mat_3x3[:, 2, 2] = cos_a + z*z*versin_a
    
    # Create 4x4 homogeneous matrices
    T_4x4 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    T_4x4[:, :3, :3] = rot_mat_3x3
    
    return T_4x4

def build_translation_matrix(device, dx=0.0, dy=0.0, dz=0.0):
    """
    Builds a single 4x4 homogeneous translation matrix.
    This is fixed, so it doesn't need a batch dimension (it will be broadcast).
    """
    T = torch.eye(4, device=device)
    T[0, 3] = dx
    T[1, 3] = dy
    T[2, 3] = dz
    return T.unsqueeze(0) # Shape [1, 4, 4] for broadcasting



def forward_kinematics_pytorch(joint_angles_batch):
    """
    Calculates the end-effector position for a batch of 4-DoF joint angle configurations.

    Args:
        joint_angles_batch (torch.Tensor): Shape [B, 4], where B is the batch size.
            Order: [shoulder_ry, shoulder_rx, shoulder_rz, elbow_ry]

    Returns:
        torch.Tensor: The batch of end-effector XYZ positions, shape [B, 3].
    """
    batch_size = joint_angles_batch.shape[0]
    device = joint_angles_batch.device

    # --- 1. Define fixed axes and link translations ---
    # These match the axes and origins in your 'simple_arm.urdf' file
    
    # Joint Axes
    axis_y = torch.tensor([0., 1., 0.], device=device)
    axis_x = torch.tensor([1., 0., 0.], device=device)
    axis_z = torch.tensor([0., 0., 1.], device=device)
    
    # Link Translations (from joint origins)
    # T_upper_arm corresponds to the <origin> tag of the 'elbow_ry' joint
    T_upper_arm_origin = build_translation_matrix(device, dz=UPPER_ARM_LEN)
    
    # T_forearm corresponds to the <origin> tag of the 'wrist_fixed' joint
    T_forearm_origin = build_translation_matrix(device, dz=FOREARM_LEN)

    # --- 2. Calculate transformation for each joint in the chain ---
    # Note: Matrix multiplication order is Parent-to-Child: T_parent @ T_joint_origin @ T_joint_rotation
    
    # T_0_to_1: Shoulder Ry (origin 0,0,0)
    T_0_1 = build_rotation_matrix(joint_angles_batch[:, 0], axis_y, device)
    
    # T_1_to_2: Shoulder Rx (origin 0,0,0)
    T_1_2 = build_rotation_matrix(joint_angles_batch[:, 1], axis_x, device)
    
    # T_2_to_3: Shoulder Rz (origin 0,0,0)
    T_2_3 = build_rotation_matrix(joint_angles_batch[:, 2], axis_z, device)
    
    # T_3_to_4: Elbow Ry (origin 0,0,0.33)
    T_3_4 = T_upper_arm_origin @ build_rotation_matrix(joint_angles_batch[:, 3], axis_y, device)
    
    # T_4_to_End: Fixed Wrist (origin 0,0,0.28)
    T_4_end = T_forearm_origin
    
    # --- 3. Combine transformations by chain multiplication ---
    # T_base_to_end = T_0->1 @ T_1->2 @ T_2->3 @ T_3->4 @ T_4->end
    # This gives the final transformation of the end-effector link relative to the base
    T_final = T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_end

    # --- 4. Extract the XYZ position ---
    # The position is in the last column of the 4x4 matrix
    end_effector_pos = T_final[:, :3, 3] # Shape: [B, 3]
    
    return end_effector_pos