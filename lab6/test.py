import os
import argparse
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import iclevrDataset
from model.DDPM import conditionalDDPM
from torchvision import transforms
from diffusers import DDPMScheduler
from tqdm import tqdm
from evaluator import evaluation_model

from torchvision.utils import make_grid, save_image
from torch.utils import tensorboard

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def inference(args, model, mode='test'):
    os.makedirs(os.path.join(args.save_root, mode), exist_ok=True)
    device = args.device
    timesteps = args.total_timesteps
    dataloader = test_dataloader(args, mode)
    noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps, beta_schedule=args.beta_schedule)
    evaluator = evaluation_model()

    results = []
    accuracy_list = []

    progress_bar = tqdm(dataloader)

    for idx, label in enumerate(progress_bar):
        label = label.to(device)
        img = torch.randn(1, 3, 64, 64).to(device)
        denoising_step = []

        for i, t in enumerate(noise_scheduler.timesteps):
            with torch.no_grad():
                residual = model(img, t, label)

            img = noise_scheduler.step(residual, t, img).prev_sample # 將img denoising
            if i % (timesteps // 10) == 0:
                denoising_step.append(img.squeeze(0))

        accuracy = evaluator.eval(img, label)
        accuracy_list.append(accuracy)

        progress_bar.set_postfix_str(f'image: {idx}, accuracy: {accuracy:.4f}')

        denoising_step.append(img.squeeze(0))
        denoising_step = torch.stack(denoising_step)

        row_image = make_grid((denoising_step + 1) / 2, nrow=denoising_step.shape[0], pad_value=0)
        save_image(row_image, f'result/{mode}/{mode}_{idx}.png') #將像素值從 [-1, 1] 範圍重新縮放到 [0, 1] 範圍

        results.append(img.squeeze(0))

    results = torch.stack(results)
    results = make_grid(results, nrow=8, pad_value=-1)
    save_image((results + 1) / 2, f'result/{mode}_result.png')

    return np.mean(accuracy_list)


def test_dataloader(args, mode="test"):
    dataset = iclevrDataset(root='iclevr', mode=mode)
        
    test_loader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                shuffle=False)  
    return test_loader

def load_checkpoint(args):
     if args.ckpt_path != None:
        model = conditionalDDPM().to(args.device)
        checkpoint = torch.load(args.ckpt_path, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'], strict=True) 
        model.eval()

        return model

def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    set_seed(40)

    model = load_checkpoint(args)
    test_accuracy = inference(args, model, mode='test')
    print(f'test accuracy: {test_accuracy}')

    new_test_accuracy = inference(args, model, mode='new_test')
    print(f'new test accuracy: {new_test_accuracy}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    

    parser.add_argument('--batch_size',    type=int,    default=1)
    parser.add_argument('--lr',            type=float,  default=0.0001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda:3", "cpu", "cuda:0"], default="cuda:0")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=7)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--save_root',     type=str, default="result", help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=300,     help="number of total epoch")
    
    parser.add_argument('--ckpt_path',     type=str, default="checkpoints/DL_lab6_313551098_張欀齡.pth", help="The path of your checkpoints") 
    parser.add_argument('--total_timesteps', type=int, default=1000)  
    parser.add_argument('--beta_schedule', type=str, default='squaredcos_cap_v2')
    
    
    args = parser.parse_args()
    
    main(args)
