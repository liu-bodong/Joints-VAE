import os
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import ikpy.chain
import ikpy.utils.plot as plot_utils
import itertools
import csv
import torch

import kinematics

base_urdf_file = "simple_arm.urdf"

M_POSES = 2401

# joint limits       =  healthy      mild        moderate     severe
shoulder_abd_limits  = [[-180, 50], [-150, 40], [-110, 30], [-80, 30]]
shoulder_flex_limits = [[-90, 180], [-70, 140], [-50, 100], [-40, 70]]
shoulder_rot_limits  = [[-90,  90], [-70,  70], [-50,  50]]
elbow_flex_limits    = [[0,   150], [0,   110], [0,    70]]

def generate_user_fROM(user_chain, joint_limits_lower_rad, joint_limits_upper_rad, num_poses=1000):
    """
    Generates an fROM point cloud for a *specific user's chain*.
    """
    # (This is your 'generate_arm_poses' function, adapted)
    
    num_active_joints = 4
    active_joint_indices = list(range(1, 1 + num_active_joints)) # Indices 1, 2, 3, 4

    # Define the Target Workspace 
    ws_min = np.array([0.2, -0.4, -0.2]) 
    ws_max = np.array([0.8, 0.4, 0.5])   

    valid_poses = []
    attempts = 0
    max_attempts = num_poses * 50 
    
    # Initial pose for IK solver
    initial_position_full = np.zeros(len(user_chain.links))
    initial_position_full[4] = np.deg2rad(10)

    while len(valid_poses) < num_poses and attempts < max_attempts:
        attempts += 1
        target_point = np.random.uniform(low=ws_min, high=ws_max)

        try:
            ik_solution_full = user_chain.inverse_kinematics(
                target_position=target_point,
                initial_position=initial_position_full,
                orientation_mode=None
            )
            pose_angles = ik_solution_full[active_joint_indices]

            # Check if the solution respects THIS user'S joint limits
            within_limits = np.all(pose_angles >= joint_limits_lower_rad) and \
                            np.all(pose_angles <= joint_limits_upper_rad)

            if within_limits:
                # We only need to check limits, not filter for zero,
                # as the limits themselves will prevent a zero pose
                # unless the limits are [0,0,0,0].
                valid_poses.append(pose_angles)

        except ValueError:
            pass # Target unreachable

    if len(valid_poses) < num_poses:
        print(f"  Warning: Only generated {len(valid_poses)}/{num_poses} poses.")
        
    return np.array(valid_poses)

if __name__ == "__main__":

    # combination of all these limits for different users
    all_joint_options = [
        shoulder_abd_limits,
        shoulder_flex_limits,
        shoulder_rot_limits,
        elbow_flex_limits
    ]

    combinations_iterator = itertools.product(*all_joint_options)

    user_joint_limits_deg = np.array(list(combinations_iterator), dtype=np.float64)


    # add noise
    noise = np.random.normal(0, 7.0, user_joint_limits_deg.shape)
    user_joint_limits_deg += noise

    csv_filename = "joint_limits.csv"
    with open(csv_filename, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["user_index", "shoulder_abd(rx)_lower", "shoulder_abd(rx)_upper", "shoulder_flex(ry)_lower","shoulder_flex(ry)_upper", "shoulder_rot(rz)_lower","shoulder_rot(rz)_upper", "elbow_flex(ry)_lower", "elbow_flex(ry)_upper"])
        data = np.reshape(user_joint_limits_deg, (-1, 8))
        for index, joint_limit in enumerate(data):
            csv_writer.writerow([str(index), *joint_limit])
            
    base_urdf_file = "simple_arm.urdf"
    active_joint_indices = [1, 2, 3, 4] # The indices of 4 active joints
    all_user_fROMs = {} # A dictionary to store the results

    print("Dataset generation starts")

    for index, limits_deg in enumerate(user_joint_limits_deg):
        print(f"user {index}")
        start_time = time.time()
        
        chain = ikpy.chain.Chain.from_urdf_file(base_urdf_file)

        limits_rad_lower = np.deg2rad([limit[0] for limit in limits_deg])
        limits_rad_upper = np.deg2rad([limit[1] for limit in limits_deg])

        for i, joint_idx in enumerate(active_joint_indices):
            new_bounds = (limits_rad_lower[i], limits_rad_upper[i])
            chain.links[joint_idx].bounds = new_bounds
            # print(f"  Set joint {joint_idx} bounds to {new_bounds}")

        user_fROM_data = generate_user_fROM(
            chain, 
            limits_rad_lower, 
            limits_rad_upper, 
            num_poses=M_POSES
        )
        
        user_name = f"{index}"
        all_user_fROMs[user_name] = user_fROM_data
        
        end_time = time.time()
        print(f"  Finished {user_name}. Generated {user_fROM_data.shape[0]} poses in {end_time - start_time:.2f}s.")
    
    csv_filename = "joints_data_144_4096.csv"
    with open(csv_filename, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["user_name", "shoulder_abd(rx)", "shoulder_flex(ry)", "shoulder_rot(rz)", "elbow_flex(ry)"])
        for user_name, fROM_data in all_user_fROMs.items():
            for pose in fROM_data:
                pose_str = ','.join(map(str, pose))
                csv_writer.writerow([user_name, *pose])
