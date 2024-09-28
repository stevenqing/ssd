from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.switch import SwitchEnv
from social_dilemmas.envs.coin import CoinEnv
from social_dilemmas.envs.coin3 import Coin3Env
from social_dilemmas.envs.coin4 import Coin4Env
from social_dilemmas.envs.coin5 import Coin5Env
from social_dilemmas.envs.lbf10 import LBF10Env
from social_dilemmas.envs.lbf15 import LBF15Env
def get_env_creator(
    env,
    num_agents,
    use_collective_reward=False,
    inequity_averse_reward=False,
    use_reward_model=False,
    alpha=0.0,
    beta=0.0,
    num_switches=6,
):
    if env == "harvest":

        def env_creator(_):
            return HarvestEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                alpha=alpha,
                beta=beta,
                env_name='HARVEST'
            )

    elif env == "coin":

        def env_creator(_):
            return CoinEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                alpha=alpha,
                beta=beta,
            )
    
    elif env == "coin3":
        def env_creator(_):
            return Coin3Env(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                use_reward_model=use_reward_model,
                alpha=alpha,
                beta=beta,
                env_name='COIN3'
            )
    elif env == "coin4":
        def env_creator(_):
            return Coin4Env(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                use_reward_model=use_reward_model,
                alpha=alpha,
                beta=beta,
                env_name='COIN4'
            )
    elif env == "coin5":
        def env_creator(_):
            return Coin5Env(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                use_reward_model=use_reward_model,
                alpha=alpha,
                beta=beta,
                env_name='COIN5'
            )
    elif env == "lbf10":
        def env_creator(_):
            return LBF10Env(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                alpha=alpha,
                beta=beta,
                env_name='LBF10'
            )
    elif env == "lbf15":
        def env_creator(_):
            return LBF15Env(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                alpha=alpha,
                beta=beta,
                env_name='LBF15'
            )

    elif env == "cleanup":

        def env_creator(_):
            return CleanupEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                alpha=alpha,
                beta=beta,
                env_name='CLEANUP'
            )

    elif env == "switch":

        def env_creator(_):
            return SwitchEnv(num_agents=num_agents, num_switches=num_switches)

    else:
        raise ValueError(f"env must be one of coin, harvest, cleanup, switch, not {env}")

    return env_creator
