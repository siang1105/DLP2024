import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import iclevrDataset
from model.DDPM import conditionalDDPM
from torchvision import transforms
from diffusers import DDPMScheduler
from tqdm import tqdm

from torch.utils import tensorboard
    
def training_stage(args):
    train_loader = train_dataloader(args)

    model = conditionalDDPM().to(args.device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.total_timesteps, beta_schedule=args.beta_schedule)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    experiment_name = f"lr_{args.lr}_{args.num_epoch}"
    writer = tensorboard.SummaryWriter(log_dir=f"runs/{experiment_name}")

    if args.ckpt_path != None:
        load_checkpoint(model, optimizer, args.checkpoint)

    for epoch in range(args.num_epoch):
        loss = training_one_step(epoch, model, optimizer, train_loader, noise_scheduler, args.total_timesteps, args.device)
        writer.add_scalar('Loss/train', loss, epoch)

        if epoch % args.per_save == 0:
            save(model, optimizer, epoch, os.path.join(args.save_root, f"epoch={epoch}.pth"))

        print(f"Epoch [{epoch}/{args.num_epoch}], Training Loss: {loss:.4f}")

    save(model, optimizer, args.num_epoch, os.path.join(args.save_root, f"epoch={args.num_epoch}.pth"))

def training_one_step(epoch, model, optimizer, train_loader, noise_scheduler, total_timesteps, device):
    model.train()
    mse_loss = nn.MSELoss()
    train_losses = []
    
    progress_bar = tqdm(train_loader, desc=f'Epoch: {epoch}', leave=True)

    for i, (x, label) in enumerate(progress_bar):
        batch_size = x.shape[0]
        x, label = x.to(device), label.to(device)
        noise = torch.randn_like(x)
        
        timesteps = get_random_timesteps(batch_size, total_timesteps, device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
        output = model(noisy_x, timesteps, label)
        
        loss = mse_loss(output, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        progress_bar.set_postfix({'Loss': np.mean(train_losses)})
        
    return np.mean(train_losses)

def train_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = iclevrDataset(root='iclevr', mode='train', transform=transform)
        
    train_loader = DataLoader(dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              drop_last=True,
                              shuffle=False)  
    return train_loader

    
def save(model, optimizer, current_epoch, path):
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),  
        "last_epoch": current_epoch
    }, path)
    print(f"save ckpt to {path}")

def load_checkpoint(model, optimizer, path):
    model.load_state_dict(torch.load(path)['state_dict'])
    optimizer.load_state_dict(torch.load(path)['optimizer'])

def get_random_timesteps(batch_size, total_timesteps, device):
    return torch.randint(0, total_timesteps, (batch_size,)).long().to(device)

def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    training_stage(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    
    parser.add_argument('--batch_size',    type=int,    default=32)
    parser.add_argument('--lr',            type=float,  default=0.0001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=7)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=300,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    
    parser.add_argument('--ckpt_path',     type=str,    default=None, help="The path of your checkpoints") 
    parser.add_argument('--total_timesteps', type=int, default=1000)  
    parser.add_argument('--beta_schedule', type=str, default='squaredcos_cap_v2')
    
    args = parser.parse_args()
    
    main(args)
