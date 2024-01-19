import numpy as np
from numpy.random import rand
import random
from social_dilemmas.envs.agent import Coin3Agent
from social_dilemmas.envs.gym.discrete_with_dtype import DiscreteWithDType
from social_dilemmas.envs.map_env import MapEnv
from social_dilemmas.maps import COIN3_MAP


# Add custom actions to the agent
_HARVEST_ACTIONS = {"FIRE": 5}  # length of firing range


COIN_VIEW_SIZE = 7


class Coin3Env(MapEnv):
    def __init__(
        self,
        ascii_map=COIN3_MAP,
        num_agents=3,
        return_agent_actions=False,
        use_collective_reward=False,
        inequity_averse_reward=False,
        use_reward_model=False,
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
        # The initial position of the apples are determined
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append([row, col, "A"])
                elif self.base_map[row, col] == b"B":
                    self.apple_points.append([row, col, "B"])
                elif self.base_map[row, col] == b"C":
                    self.apple_points.append([row, col, "C"])

    @property
    def action_space(self):
        return DiscreteWithDType(4, dtype=np.uint8)

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()
        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = Coin3Agent(agent_id, spawn_point, rotation, grid, COIN_VIEW_SIZE)
            self.agents[agent_id] = agent
            

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            if apple_point[2] == 'A':
                self.single_update_map(apple_point[0], apple_point[1], b"A")
            elif apple_point[2] == 'B':
                self.single_update_map(apple_point[0], apple_point[1], b"B")
            elif apple_point[2] == 'C':
                self.single_update_map(apple_point[0], apple_point[1], b"C")


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
                if char == b'A' or char == b'B' or char == b'C':
                    current_apples.append([row, col])
        if len(current_apples) <= 1:
            row = random.randint(1,np.shape(self.world_map)[0]-1)
            col = random.randint(1,np.shape(self.world_map)[1]-1)
            while [row, col] not in agent_positions and [row, col] not in current_apples:
                row = random.randint(1,np.shape(self.world_map)[0]-1)
                col = random.randint(1,np.shape(self.world_map)[1]-1)
                break
            if [row, col] not in agent_positions and [row, col] not in current_apples:
                spawn_prob = 0.1
                rand_num = np.random.choice([1,0.1],p=[0.9,0.1])
                if rand_num == spawn_prob:
                    num = random.randint(0,2)
                    if num == 0:
                        new_apple_points.append((row, col, b"A"))
                    if num == 1:
                        new_apple_points.append((row, col, b"B"))
                    else:
                        new_apple_points.append((row, col, b"C"))
            
        return new_apple_points

    def count_apples(self):
        # Return apples pos and type
        apple_pos = [[0,0],[0,0],[0,0]]
        apple_type = [0,0,0]
        for row in range(1,np.shape(self.world_map)[0]-1):
           for col in range(1,np.shape(self.world_map)[1]-1):
               char = self.world_map[row, col]
               if char == b'A':
                   apple_pos[0] = [int(row),int(col)]
                   apple_type[0] = 1
               elif char == b'B':
                   apple_pos[1] = [int(row),int(col)]
                   apple_type[1] = 2
               elif char == b'C':
                   apple_pos[2] = [int(row),int(col)]
                   apple_type[2] = 3
        return apple_pos, apple_type
