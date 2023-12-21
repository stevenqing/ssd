import numpy as np
from numpy.random import rand
import random
from social_dilemmas.envs.agent import CoinAgent
from social_dilemmas.envs.gym.discrete_with_dtype import DiscreteWithDType
from social_dilemmas.envs.map_env import MapEnv
from social_dilemmas.maps import COIN_MAP


# Add custom actions to the agent
_HARVEST_ACTIONS = {"FIRE": 5}  # length of firing range


COIN_VIEW_SIZE = 5


class CoinEnv(MapEnv):
    def __init__(
        self,
        ascii_map=COIN_MAP,
        num_agents=2,
        return_agent_actions=False,
        use_collective_reward=False,
        inequity_averse_reward=False,
        alpha=0.0,
        beta=0.0,
    ):
        super().__init__(
            ascii_map,
            _HARVEST_ACTIONS,
            COIN_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            beta=beta,
        )
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append([row, col, "A"])
                elif self.base_map[row, col] == b"B":
                    self.apple_points.append([row, col, "B"])

    @property
    def action_space(self):
        return DiscreteWithDType(4, dtype=np.uint8)

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()
        penalty = 0
        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = CoinAgent(agent_id, spawn_point, rotation, grid, COIN_VIEW_SIZE,penalty)
            self.agents[agent_id] = agent
            # there maybe a problem with the penalty
            agent_id_1 = "agent-" + str(1)
            if agent_id == agent_id_1 and agent.penalty > 0:
                agent1 = self.agents[1]
                tmp_agent = CoinAgent(agent1.agent_id, agent1.spawn_point, agent1.rotation, agent1.grid, agent1.COIN_VIEW_SIZE, agent.panalty)
                self.agents[1] = tmp_agent
            penalty = agent.penalty
            

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            if apple_point[2] == 'A':
                self.single_update_map(apple_point[0], apple_point[1], b"A")
            elif apple_point[2] == 'B':
                self.single_update_map(apple_point[0], apple_point[1], b"B")

    def custom_action(self, agent, action):
        agent.fire_beam(b"F")
        updates = self.update_map_fire(
            agent.pos.tolist(),
            agent.get_orientation(),
            self.all_actions["FIRE"],
            fire_char=b"F",
        )
        return updates

    def custom_map_update(self):
        """See parent class"""
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """
        current_apples = []
        new_apple_points = []
        agent_positions = self.agent_pos
        # apples can't spawn where agents are standing or where an apple already is
        isapple = False
        for row in range(1,np.shape(self.world_map)[0]-1):
            for col in range(1,np.shape(self.world_map)[1]-1):
                char = self.world_map[row, col]
                if char == b'A' or char == b'B':
                    current_apples.append([row, col])
                    isapple = True
        if isapple == False:
            row = random.randint(1,np.shape(self.world_map)[0]-1)
            col = random.randint(1,np.shape(self.world_map)[1]-1)
            if [row, col] not in agent_positions:
                spawn_prob = 0.1
                rand_num = np.random.choice([1,0.1],p=[0.9,0.1])
                if rand_num == spawn_prob:
                    num = random.randint(0,1)
                    if num == 0:
                        new_apple_points.append((row, col, b"A"))
                    else:
                        new_apple_points.append((row, col, b"B"))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get(b"A", 0)
        return num_apples
