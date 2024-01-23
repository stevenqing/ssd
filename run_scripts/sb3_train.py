import argparse

import gym
import supersuit as ss
import torch
import socket
import wandb
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from torch import nn
import os
import numpy as np 
import random
from social_dilemmas.envs.pettingzoo_env import parallel_env



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
    parser = argparse.ArgumentParser("Stable-Baselines3 PPO with Parameter Sharing")
    parser.add_argument(
        "--env-name",
        type=str,
        default="harvest",
        choices=["harvest", "cleanup", "coin3", "lbf10"],
        help="The SSD environment to use",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="note for the env",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=5,
        help="The number of agents",
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
        default=3e8,
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
        "--alpha",
        type=float,
        default=5,
        help="Advantageous inequity aversion factor",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="Disadvantageous inequity aversion factor",
    )
    parser.add_argument(
        "--l",
        type=float,
        default=1,
        help="balance constant of AGA",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=4,
        help="cpu numbers",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=12,
        help="envs numbers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="seed",
    )
    parser.add_argument("--eval-interval", type=int, default=4000)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--user_name", type=str, default="ssd_pytorch")
    parser.add_argument("--aga", action='store_true', default=False)
    parser.add_argument("--selfish-level", type=float, default=1.0)
    parser.add_argument("--play-altruistic-game", action='store_true', default=False)
    parser.add_argument("--altruistic-model", type=str, default="A")
    parser.add_argument("--alt-alpha", type=float, default=0.0)
    args = parser.parse_args()
    return args


# Use this with lambda wrapper returning observations only
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

        #flat_out = num_frames * 6 * (view_len * 2 - 1) ** 2
        
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
    
def get_vec_env(env, num_envs, num_cpus, num_frames):

    env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
    env = ss.frame_stack_v1(env, num_frames)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="stable_baselines3"
    )
    env = VecMonitor(env)

    return env

def main(args):
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    # Config
    set_seed(args.seed)
    env_name = args.env_name
    num_agents = args.num_agents
    rollout_len = args.rollout_len
    total_timesteps = args.total_timesteps
    use_collective_reward = args.use_collective_reward
    inequity_averse_reward = args.inequity_averse_reward
    play_altruistic_game = args.play_altruistic_game
    altruistic_model = args.altruistic_model
    alt_alpha = args.alt_alpha
    selfish_level = args.selfish_level
    alpha = args.alpha
    beta = args.beta

    eval_interval = args.eval_interval
    eval_episodes = args.eval_episodes
    
    print("###################")
    if args.aga:
        print(f"The flag AGA is set as {args.aga}. Thus, you are using AgA optimization.")
    else:
        print(f"The flag AGA is set as {args.aga}. Thus, you are using Consensus optimization.")
    print("###################")
    # Training
    num_cpus = args.num_cpus  # number of cpus
    config = args.__dict__
    # Training
    num_cpus = args.num_cpus #4  # number of cpus
    num_envs = args.num_envs  # number of parallel multi-agent environments
    num_frames = 6  # number of frames to stack together; use >4 to avoid automatic VecTransposeImage
    config["num_frames"] = num_frames
    features_dim = (
        128  # output layer of cnn extractor AND shared layer for policy and value functions
    )
    config["features_dim"] = features_dim
    fcnet_hiddens = [1024, 128]  # Two hidden layers for cnn extractor
    config["fcnet_hiddens"] = fcnet_hiddens
    ent_coef = 0.001  # entropy coefficient in loss
    config["ent_coef"] = ent_coef
    # batch_size = rollout_len * num_envs // 2  # This is from the rllib baseline implementation
    batch_size = rollout_len // 2 #the batch size is different with original one, it is on buffer size level
    config["batch_size"] = batch_size
    n_epochs = 30
    config["n_epochs"] = n_epochs
    gae_lambda = 1.0
    config["gae_lambda"] = gae_lambda
    gamma = 0.99
    config["gamma"] = gamma
    # target_kl = 0.01
    target_kl = 0.05
    config["target_kl"] = target_kl
    grad_clip = 40
    config["grad_clip"] = grad_clip
    verbose = 0
    config["verbose"] = verbose
    
    config["seed"] = args.seed
    selfish_level = args.selfish_level
    
    if "coin" in env_name:
        view_len = 5
    else:
        view_len = 7
    config["view_len"] = view_len
    
    lr = 0.001
    lr_schedule_args = {
        "lr": lr,
        "steps": 50_000_000,
        "weight": 0.1
    }
    
    l_schedule_args = {
        "l": args.l,
        "steps": 50_000_000,
        "weight": 1
    }

    config["lr"] = lr
    config["lr_schedule_args"] = lr_schedule_args
    config["l_schedule_args"] = l_schedule_args
        
    env = parallel_env(
        max_cycles=rollout_len,
        env=env_name,
        num_agents=num_agents,
        use_collective_reward=use_collective_reward,
        inequity_averse_reward=inequity_averse_reward,
        alpha=selfish_level,
        beta=beta,
    )
    eval_env = parallel_env(
        max_cycles=rollout_len,
        env=env_name,
        num_agents=num_agents,
        use_collective_reward=False,
        inequity_averse_reward=False,
        alpha=alpha,
        beta=beta,
    )
    env = get_vec_env(env, num_envs, num_cpus, num_frames)
    eval_env = get_vec_env(eval_env, num_envs, num_cpus, num_frames)

    run = wandb.init(config=args,
                         project="ssd_pytorch",
                         entity=args.user_name, 
                         notes=socket.gethostname(),
                         name="seed_" + str(args.seed) + f"note:{args.note}",
                         group=str(env_name) + f"_l_{args.l}/" + f"_aga_{args.aga}",
                         dir="./",
                         job_type="training",
                         reinit=True)
    
    args = wandb.config # for wandb sweep
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(
            features_dim=features_dim, num_frames=num_frames, fcnet_hiddens=fcnet_hiddens, view_len=view_len
        ),
        net_arch=[features_dim],
    )

    tensorboard_log = f"./results/sb3/{env_name}/aga_{args.aga}_l_{args.l}"
    
    model = PPO(
        "CnnPolicy",
        env=env,
        eval_env=eval_env,
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
        device=args.gpu,
        aga=args.aga,
        seed=args.seed,
        agent_nums=num_agents,
        env_nums=num_envs,
        lr_schedule_args=lr_schedule_args
    )
    model.learn(total_timesteps=total_timesteps, num_envs=num_envs, num_agents=num_agents, l_schedule_args=l_schedule_args)

    # logdir = model.logger.dir
    # model.save(logdir + "/model")
    # del model
    # model = PPO.load(logdir + "/model")  # noqa: F841


if __name__ == "__main__":
    args = parse_args()
    main(args)