import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import tqdm
import wandb

import FK_data_generator
from network import PointVAE
from loss_functions import vae_loss_function


def train(num_epochs, learning_rate, batch_size, kl_weight, num_users, points_per_user, latent_dim, train_loader, device):
    save_path = f"./models/point_vae_N{num_users}_K{points_per_user}_D{latent_dim}.pth"

    # --- Setup Weights & Biases ---
    run = wandb.init(
        entity = "liubodong-cornell-university",
        project = "ROMA-VAE",
        name = f"PointVAE_profiles{num_users}_latent{latent_dim}_points{points_per_user}_kl{kl_weight}_lr{learning_rate}_bs{batch_size}",
        config = {
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "kl_weight": kl_weight,
            "num_users": num_users,
            "points_per_user": points_per_user,
            "latent_dim": latent_dim
        },
        mode = "online"
    )


    # --- Initialize Model ---
    model = PointVAE(
        latent_dim=32,
        num_points_k=points_per_user
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Main Training Loop ---
    print("Starting training...")
    for epoch in range(num_epochs):
        
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kld_loss = 0.0
        
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        # We get (batch_data,) because our dataset returns a list
        for (input_cloud,) in progress_bar:
            
            input_cloud = input_cloud.to(device)
            
            # --- Forward Pass ---
            recon_cloud, mu, logvar = model(input_cloud)
            # print(recon_cloud.shape, input_cloud.shape)
            
            # --- Calculate Loss ---
            total_loss, recon_loss, kl_loss = vae_loss_function(
                recon_cloud,
                input_cloud,
                mu,
                logvar,
                kl_weight
            )
            
            total_loss = total_loss.mean()
            recon_loss = recon_loss.mean()
            kl_loss = kl_loss.mean()
            
            # --- Backward Pass ---
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # --- Log batch losses ---
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kld_loss += kl_loss.item()
            
            progress_bar.set_postfix({
                "Batch Loss": total_loss.item(),
                "Recon Loss": recon_loss.item(),
                "KL Loss": kl_loss.item()
            })
        
        # --- End of Epoch: Print Averages ---
        avg_loss = epoch_loss / len(train_loader)
        avg_recon = epoch_recon_loss / len(train_loader)
        avg_kld = epoch_kld_loss / len(train_loader)
        
        # if (epoch + 1) % 100 == 0 or epoch == 0:
        #     print(f"Epoch {epoch+1}/{num_epochs} | "
        #         f"Total Loss: {avg_loss:.4f} | "
        #         f"Recon Loss: {avg_recon:.4f} | "
        #         f"KL Loss: {avg_kld:.4f}")
        
        run.log({
            "total_loss": avg_loss,
            "recon_loss": avg_recon,
            "kl_loss": avg_kld
        })

    print("Training complete.")
    
    run.finish()
    
    torch.save(model.state_dict(), save_path)
    return 1
    
    
if __name__ == "__main__":
    NUM_EPOCHS = 5000
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 320
    KL_WEIGHT = 1e-3
    
    NUM_USERS = 1000
    POINTS_PER_USER = 1024
    
    LATENT_DIM = [1, 2, 3, 4, 8, 16, 32, 64, 128]
    
    PATH = f"./models/point_vae_N{NUM_USERS}_K{POINTS_PER_USER}_D{LATENT_DIM}.pth"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")    
    
    # --- Load Data ---
    print("Loading dataset")
    all_joints = np.load("./data/joint_data_1000_1024.npy")  # shape: (num_users, num_points_per_user, 4)
    all_joints = all_joints[:NUM_USERS, :POINTS_PER_USER, :]
    all_joints_tensor = torch.tensor(all_joints, dtype=torch.float32)
    dataset = TensorDataset(all_joints_tensor)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Dataset loaded.")
    
    print(f"Configs: ")
    
    for dim in LATENT_DIM:
        train(
            NUM_EPOCHS,
            LEARNING_RATE,
            BATCH_SIZE,
            KL_WEIGHT,
            NUM_USERS,
            POINTS_PER_USER,
            dim,
            train_loader,
            DEVICE
        )
    
    