import os
import numpy as np
import tqdm
import wandb
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from draw_pair_plot import draw
from network import FoldingNetV1, FoldingNetV2
from loss_functions import vae_loss_function
from validate import validate


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, optimizer, scheduler, train_loader, val_loader, config, device, run, enable_wandb):
    
    num_epochs = config['num_epochs']
    kl_weight = config['kl_weight']
    run_dir = config['run_dir']
    
    best_val_loss = float('inf')
    
    try:
        for epoch in range(num_epochs):
            model.train()
            
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            
            progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            
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
                epoch_kl_loss += kl_loss.item()
                
                progress_bar.set_postfix({
                    "Batch Loss": total_loss.item(),
                    "Recon Loss": recon_loss.item(),
                    "KL Loss": kl_loss.item()
                })
            
            # --- End of Epoch: Averages ---
            avg_train_loss = epoch_loss / len(train_loader)
            avg_train_recon = epoch_recon_loss / len(train_loader)
            avg_train_kl = epoch_kl_loss / len(train_loader)
            
        # 1. Validation Step (Every 20 epochs to save time)
            if (epoch + 1) % 20 == 0:
                val_total, val_recon, val_kl = validate(model, val_loader, kl_weight, device)
                
                # Learning Rate Scheduler Step
                # Reduce LR if validation loss plateaus
                if scheduler:
                    scheduler.step(val_total)
                    current_lr = optimizer.param_groups[0]['lr']
                else:
                    current_lr = config['learning_rate']

                # WandB Logging
                if run:
                    run.log({
                        "epoch": epoch,
                        "train/total_loss": avg_train_loss,
                        "train/recon_loss": avg_train_recon,
                        "train/kl_loss": avg_train_kl,
                        "val/total_loss": val_total,
                        "val/recon_loss": val_recon,
                        "val/kl_loss": val_kl,
                        "learning_rate": current_lr
                    })
                
                # print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_total:.4f} | LR: {current_lr:.2e}")

                # 2. Checkpoint: Best Model
                if epoch > 10000 and val_total < best_val_loss - 1e-7:
                    best_val_loss = val_total
                    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            
            else:
                # Just log train stats if no validation this epoch
                if run:
                    run.log({
                        "epoch": epoch,
                        "train/total_loss": avg_train_loss,
                        "train/recon_loss": avg_train_recon,
                        "train/kl_loss": avg_train_kl
                    })

            # 3. Checkpointing: Periodic (10k, 12k, 15k)
            if (epoch + 1) == 10000 or (epoch + 1) == 12000 or (epoch + 1) == 15000:
                save_name = f"checkpoint_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), os.path.join(run_dir, save_name))

        print("Training complete.")
        
    except KeyboardInterrupt:
        print("\n\n"+ "-" * 30)
        print("Training interrupted.")
        print("Saving last model...")
    

    torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pth"))
    
    return
    
    
def main():
    # TODO: Change hyperparams
    config = {
        'num_epochs': 20000,
        'learning_rate': 1e-4,
        'batch_size': 128,
        'kl_weight': 0,
        'num_users': 1024,
        'points_per_user': 4096,
        'latent_dim': 16,
        'seed': 42,
        'run_dir': None
    }
    
    # TODO: Double check the paths
    base_run_dir = "./runs/"
    exp_name = f"N{config['num_users']}_K{config['points_per_user']}_D{config['latent_dim']}_KL{config['kl_weight']}"
    config['run_dir'] = os.path.join(base_run_dir, exp_name)
    os.makedirs(config['run_dir'], exist_ok=True)  
          
    data_path = f"data/4joints_N{config['num_users']}_K{config['points_per_user']}.npy"
    
    setup_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TODO: Double check the logging 
    run = wandb.init(
        entity="liubodong-cornell-university",
        project="ROMA-VAE",
        name=exp_name,
        config=config,
        mode="online"  # Change to "offline" if you want to log locally only
    )

    data = np.load(data_path)  # shape: (num_users, num_points_per_user, 4)
    val_size = int(config['num_users'] * 0.125)
    
    val_data = data[:val_size, :config['points_per_user'], :]
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    
    train_data = data[val_size:, :config['points_per_user'], :]
    dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    
    # TODO: Double check the these
    model = FoldingNetV1(latent_dim=config['latent_dim'], num_points_k=config['points_per_user']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    
    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        run=run,
        enable_wandb=True
    )
        
    doc_path = os.path.join(config['run_dir'], "desc.txt")
    with open(doc_path, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
            
    # Draw final plots (Using the BEST model, not necessarily the last one)
    # Reload best weights
    best_path = os.path.join(config['run_dir'], "best_model.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        print("Loaded Best Model for visualization.")
    else:
        best_path = os.path.join(config['run_dir'], "last_model.pth")
        model.load_state_dict(torch.load(best_path))
    
    # Generate visualization (assuming draw_pair_plot handles the logic)
    # Passing the path where images should be saved
    draw(config['latent_dim'], config['points_per_user'], best_path, val_data_path, count=5)

    if run:
        run.finish()
    

if __name__ == "__main__":
    main()