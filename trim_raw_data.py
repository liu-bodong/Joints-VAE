import numpy as np


N = 10  # number of users
K = 4096  # number of points per user
data_path = "data\sirs_dense_group4268_n4096.npy"
all_joints = np.load(data_path)  # shape: (num_users, num_points_per_user, 4)
all_joints = all_joints[:N, :K, :]
print(all_joints.shape)

save_path = f"./data/sirs_dense_N{N}_K{K}.npy"
np.save(save_path, all_joints)