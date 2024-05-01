import argparse
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gym
from gym import spaces
import supersuit as ss
import torch
import torch.nn.functional as F
# pip install git+https://github.com/Rohan138/marl-baselines3
import wandb
import socket
from stable_baselines3.independent_ppo import IndependentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from torch import nn
import numpy as np
import random
from social_dilemmas.envs.pettingzoo_env import parallel_env

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def parse_args():
    parser = argparse.ArgumentParser("MARL-Baselines3 PPO with Independent Learning")
    parser.add_argument(
        "--env-name",
        type=str,
        default="harvest",
        choices=["harvest", "cleanup", "coin3", "lbf10"],
        help="The SSD environment to use",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=5,
        help="The number of agents",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=1,
        help="The number of cpus",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=12,
        help="The number of envs",
    )
    parser.add_argument(
        "--kl-threshold",
        type=float,
        default=0.01,
        help="The number of envs",
    )
    parser.add_argument(
        "--rollout-len",
        type=int,
        default=1000,
        help="length of training rollouts AND length at which env is reset",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5e8,
        help="Number of environment timesteps",
    )
    parser.add_argument(
        "--use-collective-reward",
        type=bool,
        default=False,
        help="Give each agent the collective reward across all agents",
    )
    parser.add_argument(
        "--inequity-averse-reward",
        type=bool,
        default=False,
        help="Use inequity averse rewards from 'Inequity aversion \
            improves cooperation in intertemporal social dilemmas'",
    )
    parser.add_argument(
        "--svo",
        type=bool,
        default=False,
        help="Use inequity averse rewards from 'Inequity aversion \
            improves cooperation in intertemporal social dilemmas'",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Advantageous inequity aversion factor",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="Disadvantageous inequity aversion factor",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--user_name", type=str, default="1160677229")
    parser.add_argument("--model", type=str, default='baseline')
    parser.add_argument("--using_reward_timestep", type=int, default=2000000)
    parser.add_argument("--extractor", type=str, default='cnn')
    parser.add_argument("--enable_trajs_learning", type=int, default=0,choices=[0, 1])
    parser.add_argument("--add_spawn_prob", type=bool, default=False)
    args = parser.parse_args()
    return args


# Use this with lambda wrapper returning observations only
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out) 
    
class CBAM(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space: gym.spaces.Box, 
                 add_spawn_prob=False,
                 channel=18, 
                 features_dim=128,
                 view_len=7, 
                 num_frames=6, 
                 fcnet_hiddens=[1024, 128], 
                 reduction=16, 
                 kernel_size=7):
        
        super(CBAM, self).__init__(observation_space,features_dim)
        self.add_spawn_prob = add_spawn_prob
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)

        flat_out = num_frames * 3 * (view_len * 2 + 1) ** 2 # eliminate the padding?

        self.fc1 = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])
        self.fc2 = nn.Linear(in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.ca(x)
        x = self.sa(x)

        # flatten features
        features = torch.flatten(F.relu(x), start_dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return features





class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim=128,
        view_len=7,
        num_frames=6,
        fcnet_hiddens=[1024, 128],
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        flat_out = num_frames * 6 * (view_len * 2 - 1) ** 2
        self.conv = nn.Conv2d(
            in_channels=num_frames * 3,  # Input: (3 * 4) x 15 x 15
            out_channels=num_frames * 6,  # Output: 24 x 13 x 13
            kernel_size=3,
            stride=1,
            padding="valid",
        )
        self.fc1 = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])
        self.fc2 = nn.Linear(in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1])

    def forward(self, observations) -> torch.Tensor:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        observations = observations.permute(0, 3, 1, 2)
        features = torch.flatten(F.relu(self.conv(observations)), start_dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return features


def main(args):
    # Config
    set_seed(args.seed)
    model=args.model
    extractor = args.extractor
    env_name = args.env_name
    num_agents = args.num_agents
    rollout_len = args.rollout_len
    total_timesteps = args.total_timesteps
    use_collective_reward = args.use_collective_reward
    inequity_averse_reward = args.inequity_averse_reward
    svo = args.svo
    add_spawn_prob = args.add_spawn_prob
    alpha = args.alpha
    beta = args.beta
    num_cpus = args.num_cpus
    num_envs = args.num_envs
    using_reward_timestep = args.using_reward_timestep
    if args.enable_trajs_learning == 0:
        enable_trajs_learning = False
    else:
        enable_trajs_learning = True
    target_kl = args.kl_threshold
    # Training
    num_frames = 6  # number of frames to stack together; use >4 to avoid automatic VecTransposeImage
    features_dim = (
        128  # output layer of cnn extractor AND shared layer for policy and value functions
    )
    fcnet_hiddens = [1024, 128]  # Two hidden layers for cnn extractor
    ent_coef = 0.001  # entropy coefficient in loss
    batch_size = rollout_len * num_envs // 2  # This is from the rllib baseline implementation
    lr = 0.0001
    n_epochs = 30
    gae_lambda = 1.0
    gamma = 0.99
    grad_clip = 40
    verbose = 3

    env = parallel_env(
        max_cycles=rollout_len,
        env=env_name,
        num_agents=num_agents,
        use_collective_reward=use_collective_reward,
        inequity_averse_reward=inequity_averse_reward,
        alpha=alpha,
        beta=beta,
    )
    if add_spawn_prob:
        env = ss.observation_lambda_v0(env, lambda x, _: {"curr_obs": x["curr_obs"], "vector_state": x["vector_state"]}, lambda s: spaces.Dict({"curr_obs": s["curr_obs"], "vector_state": s["vector_state"]}))
    else:
        env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
    env = ss.frame_stack_v1(env, num_frames)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="stable_baselines3"
    )
    env = VecMonitor(env)

    if model == 'baseline':
        if inequity_averse_reward:
            model_name = "inequity_aversion"
        elif use_collective_reward:
            model_name = "collective"
        elif svo:
            model_name = "svo"
        else:
            model_name = "baseline"
    else:
        model_name = model

    run = wandb.init(config=args,
                     project="Neurips2024",
                    entity=args.user_name, 
                    notes=socket.gethostname(),
                    name=str(env_name) +"_" + str(extractor) + str(model_name),
                    group=str(env_name) + str(model_name)+ "_independent_" + str(args.seed)+ "_" + str(args.alpha),
                    dir="./",
                    job_type="training",
                    reinit=True)



    # if enable_trajs_learning:
    #     run = wandb.init(config=args,
    #                         project="SSD_pytorch",
    #                         entity=args.user_name, 
    #                         notes=socket.gethostname(),
    #                         name=str(env_name) + "_" + str(extractor) + "_" + str(model),
    #                         group=str(env_name) +"_trajs_motive_" + str(model)+ "_independent_" + str(args.seed)+ "_" + str(args.alpha),
    #                         dir="./",
    #                         job_type="training",
    #                         reinit=True)
    # else:
    #     if env_name == 'harvest':
    #         run = wandb.init(config=args,
    #                         project="SSD_pytorch",
    #                         entity=args.user_name, 
    #                         notes=socket.gethostname(),
    #                         name=str(env_name) + "_" + str(extractor) + "_" + str(model)+ "_" + "test" ,
    #                         group=str(env_name) +"_hard_cf_modified_discrete_" + str(model)+ "_independent_" + str(args.seed)+ "_" + str(args.alpha) + "_add_spawn_prob_" + str(add_spawn_prob),
    #                         dir="./",
    #                         job_type="training",
    #                         reinit=True)
    #     else:
    #         run = wandb.init(config=args,
    #                             project="SSD_pytorch",
    #                             entity=args.user_name, 
    #                             notes=socket.gethostname(),
    #                             name=str(env_name) +"_" + str(extractor) + "_" + str(model),
    #                             group=str(env_name) + "_cf_modified_discrete_" + str(model)+ "_independent_" + str(args.seed)+ "_" + str(args.alpha),
    #                             dir="./",
    #                             job_type="training",
    #                             reinit=True)
    
    args = wandb.config # for wandb sweep
    if extractor == 'cbam':
        policy_kwargs = dict(
            features_extractor_class=CBAM,
            features_extractor_kwargs=dict(
                features_dim=features_dim, num_frames=num_frames, fcnet_hiddens=fcnet_hiddens,  add_spawn_prob=add_spawn_prob
            ),
            net_arch=[features_dim],
            num_agents=args.num_agents,
            # env_name=env_name,       
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(
                features_dim=features_dim, num_frames=num_frames, fcnet_hiddens=fcnet_hiddens
            ),
            net_arch=[features_dim],
            num_agents=args.num_agents,
            # env_name=env_name
        )


    tensorboard_log = f"./results/{env_name}_ppo_independent"
    if model == 'baseline':
        model = IndependentPPO(
            "CnnPolicy",
            num_agents=num_agents,
            env=env,
            learning_rate=lr,
            n_steps=rollout_len,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            max_grad_norm=grad_clip,
            target_kl=target_kl,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            alpha=alpha,
            model=args.model,
            using_reward_timestep=using_reward_timestep,
            enable_trajs_learning=enable_trajs_learning,
            env_name=env_name,
            add_spawn_prob=add_spawn_prob,
            svo=svo
        )
    elif model == 'causal' or model == 'team':
        model = IndependentPPO(
            "RewardPolicy",
            num_agents=num_agents,
            env=env,
            learning_rate=lr,
            n_steps=rollout_len,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            max_grad_norm=grad_clip,
            target_kl=target_kl,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            alpha=alpha,
            model=args.model,
            using_reward_timestep=using_reward_timestep,
            enable_trajs_learning=enable_trajs_learning,
            env_name=env_name,
            add_spawn_prob=add_spawn_prob
        )
    elif model == 'vae':
        model = IndependentPPO(
            "TransitionPolicy",
            num_agents=num_agents,
            env=env,
            learning_rate=lr,
            n_steps=rollout_len,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            max_grad_norm=grad_clip,
            target_kl=target_kl,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            alpha=alpha,
            # beta=beta,
            model=args.model,
            using_reward_timestep=using_reward_timestep,
            enable_trajs_learning=enable_trajs_learning,
            env_name=env_name,
            # use_collective_reward=use_collective_reward,
            # inequity_averse_reward=inequity_averse_reward,
        )
    model.learn(total_timesteps=total_timesteps)

    logdir = model.logger.dir
    model.save(logdir)
    del model
    model = IndependentPPO.load(  # noqa: F841
        logdir, "CnnPolicy", num_agents, env, rollout_len, policy_kwargs, tensorboard_log, verbose
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
