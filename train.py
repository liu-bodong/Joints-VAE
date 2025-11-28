import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import tqdm
import wandb
import os

from draw_pair_plot import draw
from network import FoldingNetV1, FoldingNetV2
from loss_functions import vae_loss_function
from validate import validate


def train(num_epochs, learning_rate, batch_size, kl_weight, num_users, points_per_user, latent_dim, train_loader, device, enable_wandb):
    print(kl_weight)
    

    if enable_wandb:
        # --- Setup Weights & Biases ---
        run = wandb.init(
            entity = "liubodong-cornell-university",
            project = "ROMA-VAE-Few-Sample-Overfit",
            name = f"FoldingNetVae_3fold_profiles{num_users}_latent{latent_dim}_points{points_per_user}_kl{kl_weight}_lr{learning_rate}_bs{batch_size}",
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
    model = FoldingNetV1(latent_dim=latent_dim, num_points_k=points_per_user).to(device)
    
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
        
        if enable_wandb:
            run.log({
                "total_loss": avg_loss,
                "recon_loss": avg_recon,
                "kl_loss": avg_kld
            })
        
        # TODO: write early stopping logic
        # TODO: also save checkpoints so we could manually stopit
        # if epoch >= 2000 and 

    print("Training complete.")
    
    if enable_wandb:
        run.finish()
    
    return model
    
    
if __name__ == "__main__":
   
    
    NUM_EPOCHS = 10000
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 256
    KL_WEIGHT = 0
    
    NUM_USERS = 1
    POINTS_PER_USER = 4096
    
    LATENT_DIM = 8
    primitive = "4d_cube_regular" 
    
    run_dir = f"./runs/3fold_{primitive}_N{NUM_USERS}_K{POINTS_PER_USER}_D{LATENT_DIM}_KL{KL_WEIGHT}"
    

    
    train_data_path = f"data\sirs_dense_N10_K4096.npy"
    val_data_path = "data\sirs_dense_N10_K4096.npy"
    
    model_path = f"/1fold_{primitive}_N{NUM_USERS}_K{POINTS_PER_USER}_D{LATENT_DIM}_KL{KL_WEIGHT}.pth"
    doc_path  = f"/doc.txt"
    save_path = run_dir + model_path
    print(f"Save path: {save_path}")
    doc_path = run_dir + doc_path
    print(f"Doc path: {doc_path}")
    
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")    
    
    # --- Load Data ---
    print("Loading dataset")
    train_data = np.load(train_data_path)  # shape: (num_users, num_points_per_user, 4)
    train_data = train_data[:NUM_USERS, :POINTS_PER_USER, :]
    dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_data = np.load(val_data_path)  # shape: (num_users, num_points_per_user, 4)
    val_data = val_data[:NUM_USERS, :POINTS_PER_USER, :]
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    
    print("Dataset loaded.")
    
    print(f"Configs: ")

    model = train(
                num_epochs=NUM_EPOCHS, 
                learning_rate=LEARNING_RATE, 
                batch_size=BATCH_SIZE, 
                kl_weight=KL_WEIGHT, 
                num_users=NUM_USERS, 
                points_per_user=POINTS_PER_USER, 
                latent_dim=LATENT_DIM, 
                train_loader=train_loader, 
                device=DEVICE,
                enable_wandb=True
        )
    
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    validate(model, val_loader, KL_WEIGHT, DEVICE)
    
    with open(doc_path, 'w') as f:
        f.write(f"Number of Epochs: {NUM_EPOCHS}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"KL Weight: {KL_WEIGHT}\n")
        f.write(f"Number of Users: {NUM_USERS}\n")
        f.write(f"Points per User: {POINTS_PER_USER}\n")
        f.write(f"Latent Dimension: {LATENT_DIM}\n")
        f.write(f"Training Data Path: {train_data_path}\n")
        f.write(f"Validation Data Path: {val_data_path}\n")
        f.write("Primitive type: " + primitive + "\n")
        
    draw(LATENT_DIM, POINTS_PER_USER, save_path, val_data_path, count=3)   
    
    
    
    