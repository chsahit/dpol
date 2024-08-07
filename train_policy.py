# diffusion policy import
import numpy as np
import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm

import vis_encoder

# env import
from conditional_residual import ConditionalUnet1D
from global_vars import *
from image_dataset import ImageDataset
from push_t_statedataset import PushTStateDataset


def train_policy(num_demos: int, vision_based: bool):
    if vision_based:
        dataset_cls = ImageDataset
    else:
        dataset_cls = PushTStateDataset

    dataset = dataset_cls(
        dataset_path="buffer",
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        num_demos=num_demos,
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=1,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True,
    )
    action_dim = 2
    # ResNet18 has output dim of 512
    vision_feature_dim = 512
    # agent_pos is 2 dimensional
    lowdim_obs_dim = 2
    # observation feature has 514 dims in total per step

    if vision_based:
        obs_dim = vision_feature_dim + lowdim_obs_dim
    else:
        obs_dim = 8

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim, global_cond_dim=obs_dim * obs_horizon
    )
    vision_encoder = vis_encoder.get_resnet("resnet18")
    vision_encoder = vis_encoder.replace_bn_with_gn(vision_encoder)
    # the final arch has 2 parts
    if vision_based:
        nets = nn.ModuleDict(
            {"vision_encoder": vision_encoder, "noise_pred_net": noise_pred_net}
        )
    else:
        nets = noise_pred_net

    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule="squaredcos_cap_v2",
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type="epsilon",
    )

    # device transfer
    device = torch.device("cuda")
    _ = nets.to(device)

    num_epochs = 200

    ema = EMAModel(parameters=nets.parameters(), power=0.75)

    optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs,
    )

    with tqdm(range(num_epochs), desc="Epoch") as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                for nbatch in tepoch:
                    if vision_based:
                        nimage = nbatch["image"][:, :obs_horizon].to(device)
                        nagent_pos = nbatch["agent_pos"][:, :obs_horizon].to(device)
                        naction = nbatch["action"].to(device)
                        B = nagent_pos.shape[0]

                        # encoder vision features
                        image_features = nets["vision_encoder"](
                            nimage.flatten(end_dim=1)
                        )
                        image_features = image_features.reshape(*nimage.shape[:2], -1)
                        # (B,obs_horizon,D)

                        # concatenate vision feature and low-dim obs
                        obs_cond = torch.cat([image_features, nagent_pos], dim=-1)
                    else:
                        nobs = nbatch["obs"].to(device)
                        naction = nbatch["action"].to(device)
                        B = nobs.shape[0]

                        # observation as FiLM conditioning
                        # (B, obs_horizon, obs_dim)
                        obs_cond = nobs[:, :obs_horizon, :]
                    # (B, obs_horizon * obs_dim)
                    obs_cond = obs_cond.flatten(start_dim=1)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (B,),
                        device=device,
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond
                    )

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    # Weights of the EMA model
    # is used for inference
    ema_nets = nets
    ema.copy_to(ema_nets.parameters())

    return noise_scheduler, ema_nets, stats
