import torch
import numpy as np

# --- Define arm constants ---
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
    
    # ensure axis is on device
    axis = axis_vector.to(device)
    # NOTE: Assumes axis_vector is a unit vector, which is true for [1,0,0], [0,1,0], [0,0,1]
    
    # get trig components
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
    return T.unsqueeze(0) # shape= [1, 4, 4] for broadcasting


def forward_kinematics_pytorch(joint_angles_batch):
    """
    Calculates the end-effector position for a batch of 4-DoF joint angle configurations.
    
    Coordinate System:
        +X: Lateral (Right)
        +Y: Forward (Chest)
        +Z: Upward
     
    Zero Pose (all angles = 0):
        Arm hangs from origin (0,0,0) straight down the -Z axis. The end-effector is at (0, 0, - TOTAL_ARM_LEN).

    Args:
        joint_angles_batch (torch.Tensor): Shape [B, 4], where B is the batch size.
            Angles must be in RADIANS.
            Order:
             [0] Shoulder Abduction (Ry)
             [1] Shoulder Flexion (Rx)
             [2] Shoulder Rotation (Rz)
             [3] Elbow Flexion (local Rx)

    Returns:
        torch.Tensor: The batch of end-effector XYZ positions, shape [B, 3].
    """
    batch_size = joint_angles_batch.shape[0]
    device = joint_angles_batch.device
    
    # print(f"Processing {batch_size} poses for FK on device {device}.")


    # 1. Define fixed axes and link translations
    
    # Joint Axes
    axis_x = torch.tensor([1., 0., 0.], device=device)
    axis_y = torch.tensor([0., 1., 0.], device=device)
    axis_z = torch.tensor([0., 0., 1.], device=device)
    
    # Link Translations (from joint origins)
    # Arm extends down the -Z axis from the origin
    
    # T_upper_arm translates from shoulder to elbow
    T_upper_arm_origin = build_translation_matrix(device, dz=-UPPER_ARM_LEN)
    
    # T_forearm translates from elbow to wrist (end-effector)
    T_forearm_origin = build_translation_matrix(device, dz=-FOREARM_LEN)


    # 2. Calculate transformation for each joint in the chain
    
    # T_0_to_1: Shoulder Abduction (Ry)
    T_0_1 = build_rotation_matrix(joint_angles_batch[:, 0], axis_y, device)
    
    # T_1_to_2: Shoulder Flexion (Rx)
    T_1_2 = build_rotation_matrix(joint_angles_batch[:, 1], axis_x, device)
    
    # T_2_to_3: Shoulder Rotation (Rz)
    T_2_3 = build_rotation_matrix(joint_angles_batch[:, 2], axis_z, device)
    
    # T_3_to_4: Elbow Flexion (local Rx)
    # This transformation is T_translation @ T_rotation
    T_3_4 = T_upper_arm_origin @ build_rotation_matrix(joint_angles_batch[:, 3], axis_x, device)
    
    # T_4_to_End: Fixed Wrist (translation only)
    T_4_end = T_forearm_origin
    
    
    # 3. Combine transformations by chain multiplication
    # T_base_to_end = T_0->1 @ T_1->2 @ T_2->3 @ T_3->4 @ T_4->end
    T_final = T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_end


    # 4. Extract the XYZ position
    # The position is in the last column of the 4x4 matrix
    end_effector_pos = T_final[:, :3, 3] # Shape: [B, 3]
    
    return end_effector_pos


# Run this script for sanity check
if __name__ == "__main__":
    # --- Define test poses in degrees ---
    # [Abd, Flex, Rot, Elbow]
    poses_deg = [
        [0, 0, 0, 0],       # 1. Zero pose: arm straight down
        [0, 0, 0, 90],      # 2. Elbow bent 90 degrees
        [0, 90, 0, 0],      # 3. Shoulder flexed 90 (arm straight forward)
        [-90, 0, 0, 0],     # 4. Shoulder abducted 90 (arm straight right)
        [-90, 0, 0, 90]     # 5. Abducted 90, elbow bent 90
    ]
    
    angles_batch_deg = torch.tensor(poses_deg, dtype=torch.float32)
    angles_batch_rad = torch.deg2rad(angles_batch_deg)
    positions = forward_kinematics_pytorch(angles_batch_rad)
    
    # --- Print Results ---
    print("Sanity check: 5 tests")
    print("Arm Lengths: Upper={:.2f}, Forearm={:.2f}, Total={:.2f}\n".format(
        UPPER_ARM_LEN, FOREARM_LEN, UPPER_ARM_LEN + FOREARM_LEN
    ))
    
    names = [
        "1. Zero Pose",
        "2. Elbow 90 deg",
        "3. Shoulder Flex 90 deg",
        "4. Shoulder Abd 90 deg",
        "5. Abd 90 + Elbow 90"
    ]

    print(f"{'Pose':<25} {'Input (deg)':<20} {'Output Position (X, Y, Z)':<40}")
    print("-" * 85)
    
    for i, name in enumerate(names):
        pos_str = f"({positions[i, 0]:.2f}, {positions[i, 1]:.2f}, {positions[i, 2]:.2f})"
        print(f"{name:<25} {str(poses_deg[i]):<20} {pos_str:<40}")

    # --- Expected Outputs Explained ---
    # 1. Zero Pose: (0, 0, -0.61)    -> Correct, straight down -Z
    # 2. Elbow 90:  (0, 0.28, -0.33) -> Correct, elbow at (0,0,-0.33), hand swung forward +Y
    # 3. Flex 90:   (0, 0.61, 0)     -> Correct, arm straight forward along +Y
    # 4. Abd 90:    (0.61, 0, 0)     -> Correct, arm straight right along +X
    # 5. Abd 90 + Elbow 90: (0.33, 0.28, 0) -> Correct, upper arm along +X, forearm along +Y