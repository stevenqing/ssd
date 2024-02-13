# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy, RewardActorCriticPolicy, MultiInput_CNNVector_ActorCriticPolicy, MultiInput_CNNVector_REWARDActorCriticPolicy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy
RewardPolicy = RewardActorCriticPolicy
CNNVectorPolicy = MultiInput_CNNVector_ActorCriticPolicy
CNNVectorRewardPolicy = MultiInput_CNNVector_REWARDActorCriticPolicy
