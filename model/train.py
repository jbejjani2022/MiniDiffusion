import os
import gc
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as TF
from IPython.display import display
from torchmetrics import MeanMetric
from torchvision.utils import make_grid
from torch.cuda import amp
from tqdm import tqdm
import matplotlib.pyplot as plt

from unet import UNet
from configs import BaseConfig, TrainingConfig
from dataloader import get_dataloader, inverse_transform
from helpers import get, frames2vid, setup_log_directory
from diffusion import DenoiseDiffusion


def train_one_epoch(model, dd, loader, optimizer, scaler, loss_fn, epoch=800, 
                   base_config=BaseConfig(), training_config=TrainingConfig()):
    
    loss_record = MeanMetric()
    model.train()

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")
         
        for x0s, _ in loader:
            tq.update(1)
            
            ts = torch.randint(low=1, high=training_config.TIMESTEPS, size=(x0s.shape[0],), device=base_config.DEVICE)
            xts, gt_noise = dd.q_sample(x0s, ts)

            with amp.autocast():
                pred_noise = model(xts, ts)
                loss = loss_fn(gt_noise, pred_noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.detach().item()
            loss_record.update(loss_value)

            tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

        mean_loss = loss_record.compute().item()
    
        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")
    
    return mean_loss 


@torch.inference_mode()
def reverse_diffusion(model, dd, timesteps=1000, img_shape=(3, 64, 64), 
                      num_images=5, nrow=8, device=BaseConfig.DEVICE, **kwargs):

    x = torch.randn((num_images, *img_shape), device=device)
    model.eval()
    
    save_path = BaseConfig.working_dir + "/sample.png"

    if kwargs.get("generate_video", False):
        outs = []

    for time_step in tqdm(iterable=reversed(range(1, timesteps)), 
                          total=timesteps-1, dynamic_ncols=False, 
                          desc="Sampling :: ", position=0):

        ts = torch.ones(num_images, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

        predicted_noise = model(x, ts)

        beta_t                            = get(dd.beta, ts)
        one_by_sqrt_alpha_t               = get(dd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(dd.sqrt_one_minus_alpha_cumulative, ts) 

        x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )

        if kwargs.get("generate_video", False):
            x_inv = inverse_transform(x).type(torch.uint8)
            grid = make_grid(x_inv, nrow=nrow, pad_value=255.0).to("cpu")
            ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
            outs.append(ndarr)

    if kwargs.get("generate_video", False): # Generate and save video of the entire reverse process. 
        frames2vid(outs, kwargs['save_path'])
        display(Image.fromarray(outs[-1][:, :, ::-1])) # Display the image at the final timestep of the reverse process.
        return None

    else: # Display and save the image at the final timestep of the reverse process. 
        x = inverse_transform(x).type(torch.uint8)
        grid = make_grid(x, nrow=nrow, pad_value=255.0).to("cpu")
        pil_image = TF.functional.to_pil_image(grid)
        pil_image.save(kwargs['save_path'], format=save_path[-3:].upper())
        display(pil_image)
        return None


class ModelConfig:
    N_CH = 64
    BASE_CH_MULT = (1, 2, 4, 4)
    APPLY_ATTENTION = (False, False, True, True)
    N_BLOCKS = 2

eps_model = UNet(
    input_channels = TrainingConfig.IMG_SHAPE[0],
    n_channels = ModelConfig.N_CH,
    ch_mults = ModelConfig.BASE_CH_MULT,
    is_attn = ModelConfig.APPLY_ATTENTION,
    n_blocks = ModelConfig.N_BLOCKS
)

eps_model.to(BaseConfig.DEVICE)

dd = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=TrainingConfig.TIMESTEPS,
        device=BaseConfig.DEVICE
    )


if __name__ == '__main__':
    optimizer = torch.optim.AdamW(eps_model.parameters(), lr=TrainingConfig.LR)

    dataloader = get_dataloader(
        dataset_name  = BaseConfig.DATASET,
        batch_size    = TrainingConfig.BATCH_SIZE,
        device        = BaseConfig.DEVICE,
        pin_memory    = True,
        num_workers   = TrainingConfig.NUM_WORKERS,
    )

    loss_fn = nn.MSELoss()

    scaler = amp.GradScaler()
    
    total_epochs = TrainingConfig.NUM_EPOCHS + 1
    log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig())

    generate_video = False
    ext = ".mp4" if generate_video else ".png"
    
    # Training loop
    for epoch in range(1, total_epochs):
        torch.cuda.empty_cache()
        gc.collect()

        # Algorithm 1: Training
        train_one_epoch(eps_model, dd, dataloader, optimizer, scaler, loss_fn, epoch=epoch)

        if epoch % 5 == 0:
            save_path = os.path.join(log_dir, f"{epoch}{ext}")

            # Sample
            reverse_diffusion(eps_model, dd, timesteps=TrainingConfig.TIMESTEPS, num_images=32, generate_video=generate_video,
                save_path=save_path, img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE,
            )

            # clear_output()
            checkpoint_dict = {
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "model": eps_model.state_dict()
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "ckpt.tar"))
            del checkpoint_dict