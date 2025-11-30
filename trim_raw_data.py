import numpy as np


N = 1024  # number of users
K = 2048  # number of points per user
data_path = "data/4joints_N4096_K4096.npy"
data = np.load(data_path)  # shape: (num_users, num_points_per_user, 4)
data = data[:N, :K, :]
print(data.shape)

save_path = f"data/4joints_N{N}_K{K}.npy"
np.save(save_path, data)
