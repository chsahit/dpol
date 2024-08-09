import collections

# env import
import numpy as np
import torch
from skvideo.io import vwrite
from tqdm.auto import tqdm

from global_vars import *
from pih_env import PIHEnv
from push_t_statedataset import *
from train_policy import train_policy


def eval_policy(
    noise_scheduler, ema_nets, stats, vision_based: bool, seed: int = 100000
) -> bool:
    device = torch.device("cuda")
    max_steps = 400
    action_dim = 3
    num_diffusion_iters = 100
    env = PIHEnv(image_obs=vision_based)
    # use a seed >200 to avoid initial states seen in the training dataset
    env.seed(seed)
    obs, info = env.reset()
    obs = obs[2:]
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [env.render(mode="rgb_array")]
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc="Eval", disable=True) as pbar:
        while not done:
            B = 1
            if vision_based:
                images = np.stack([x["image"] for x in obs_deque])
                agent_poses = np.stack([x["agent_pos"] for x in obs_deque])

                # normalize observation
                nagent_poses = normalize_data(agent_poses, stats=stats["agent_pos"])
                # images are already normalized to [0,1]

                nimages = images
                # device transfer
                nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
                nagent_poses = torch.from_numpy(nagent_poses).to(
                    device, dtype=torch.float32
                )
            else:
                obs_seq = np.stack(obs_deque)
                nobs = normalize_data(obs_seq, stats=stats["obs"])
                nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

            # infer action
            with torch.no_grad():
                if vision_based:
                    image_features = ema_nets["vision_encoder"](nimages)
                    obs_features = torch.cat([image_features, nagent_poses], dim=-1)
                    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
                else:
                    obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn((B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    if vision_based:
                        f = ema_nets["noise_pred_net"]
                    else:
                        f = ema_nets
                    noise_pred = f(sample=naction, timestep=k, global_cond=obs_cond)

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=naction
                    ).prev_sample

            # unnormalize action
            naction = naction.detach().to("cpu").numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats["action"])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end, :]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, _, info = env.step(action[i])
                obs = obs[2:]
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                imgs.append(env.render(mode="rgb_array"))

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break
    vwrite("vis.mp4", imgs)
    if step_idx < max_steps - 1:
        return True
    else:
        return False


def experiments(num_demos: int, vision_based: bool, num_experiments: int):
    noise_scheduler, ema_nets, stats = train_policy(
        num_demos=num_demos, vision_based=vision_based, agent="block"
    )
    seed0 = 100000
    score = 0
    for i in tqdm(range(num_experiments), desc="Eval"):
        score += eval_policy(
            noise_scheduler=noise_scheduler,
            ema_nets=ema_nets,
            stats=stats,
            vision_based=vision_based,
            seed=seed0 + i,
        )
    return float(score) / float(num_experiments)


def sweep():
    results = dict()
    for vis in [False]:
        result = experiments(num_demos=200, vision_based=vis, num_experiments=50)
        results[str(vis)] = result
    for k, v in results.items():
        print(f"vision={k}, sr={v}")


def test():
    noise_scheduler, ema_nets, stats = train_policy(
        num_demos=200, vision_based=False, agent="block"
    )
    eval_policy(
        noise_scheduler,
        ema_nets,
        stats,
        False
    )

if __name__ == "__main__":
    sweep()
