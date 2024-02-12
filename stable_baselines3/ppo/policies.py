# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy, RewardActorCriticPolicy, MultiInput_CNNVector_ActorCriticPolicy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy
RewardPolicy = RewardActorCriticPolicy
CNNVectorPolicy = MultiInput_CNNVector_ActorCriticPolicy
