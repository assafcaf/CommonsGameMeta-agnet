import gym
import numpy as np
import copy
# from social_dilemmas.envs
from .commons_agent import HarvestCommonsAgent, HARVEST_DEFAULT_VIEW_SIZE
from .constants import HARVEST_MAP
from .map_env import MapEnv, ACTIONS
from .agent import MetaHarvestAgent
from Danfoa.envs.utils.utility_funcs import rgb2gray, pad, depad, safe_mean

from Danfoa.envs.gym.discrete_with_dtype import DiscreteWithDType

APPLE_RADIUS = 2

# Add custom actions to the agent
ACTIONS['FIRE'] = 5  # length of firing range

# SPAWN_PROB = [0, 0.005, 0.02, 0.05]

SPAWN_PROB = {0: np.array([0, 0.0025, 0.005, 0.025]),
              1: np.array([0, 0.005, 0.01, 0.05]),
              2: np.array([0, 0.01, 0.05, 0.1]),
              3: np.array([0, 0.02, 0.075, 0.125]),
              4: np.array([0, 0.05, 0.1, 0.15]),
              5: np.array([0, 0.075, 0.15, 0.2]),
              6: np.array([0, 0.1, 0.15, 0.25]),
              7: np.array([0, 0.15, 0.25, 0.35]),
              }
TIMEOUT_TIME = 25
META_ACTION = {0: -1, # no-op
               1: -.5,
               2: 0,
               3: .5,
               4: 1
               }

OUTCAST_POSITION = -99

AGENT_COLOR = [181, 4, 10]  #
DEFAULT_COLORMAP = {' ': [0, 0, 0],  # Black background
                   '0': [0, 0, 0],  # Black background beyond map walls
                   '': [180, 180, 180],  # Grey board walls
                   '@': [180, 180, 180],  # Grey board walls
                   'A': [0, 255, 0],  # Green apples
                   'F': [255, 255, 0],  # Yellow fining beam
                   'P': [159, 67, 255],  # Purple player

                   # Colours for agents. R value is a unique identifier
                   '1': [159, 67, 255],  # Purple
                   '2': [2, 81, 154],  # Blue
                   '3': [204, 0, 204],  # Magenta
                   '4': [216, 30, 54],  # Red
                   '5': [254, 151, 0],  # Orange
                   '6': [100, 255, 255],  # Cyan
                   '7': [99, 99, 255],  # Lavender
                   '8': [250, 204, 255],  # Pink
                   '9': [238, 223, 16]}  # Yellow

MEDIUM_HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@P A P   A    A    A    A  A    A    @',
    '@  AAA  AAA  AAA  AAA  AAAAAA  AAA   @',
    '@ A A    A    A    A    A  A    A   P@',
    '@PA             A      A       A     @',
    '@ A   A    A    A    A  A A  A    A  @',
    '@PAA AAA  AAA  AAA  AAA     AAA  AAA @',
    '@ A   A    A  A A  A A   P   A    A  @',
    '@PA                                P @',
    '@ A    A    A    A    A  A    A    A @',
    '@AAA  AAA  AAA  AAA  AA AAA  AAA  AA @',
    '@ A    A    A    A    A  A    A    A @',
    '@P A A A               P             @',
    '@P  A    A    A    A       P     P   @',
    '@  AAA  AAA  AAA  AAA         P    P @',
    '@P  A    A    A    A   P   P  P  P   @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@', ]

SMALL_HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@P A    A    A    A  P AP@',
    '@PAAA  AAA  AAA  AAA  AAA@',
    '@  A    A    A    A    A @',
    '@P                       @',
    '@    A    A    A    A    @',
    '@   AAA  AAA  AAA  AAA   @',
    '@P P A    A    A    A P P@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@', ]

MAP = {"small": SMALL_HARVEST_MAP,
       "medium": MEDIUM_HARVEST_MAP}


class MetaHarvestCommonsEnv(MapEnv):

    def __init__(self, ascii_map=HARVEST_MAP, num_agents=1, harvest_view_size=HARVEST_DEFAULT_VIEW_SIZE,
                 color_map=None, ep_length=1000, k=25):
        if color_map is None:
            color_map = DEFAULT_COLORMAP
        self.apple_points = []

        if color_map is None:
            color_map = DEFAULT_COLORMAP

        super().__init__(ascii_map, num_agents, color_map=color_map, ep_length=ep_length, view_len=harvest_view_size)
        self.agents["meta"] = MetaHarvestAgent()
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'A':
                    self.apple_points.append([row, col])
        self.spawn_prob = SPAWN_PROB[2]
        self.timeout_time = copy.copy(TIMEOUT_TIME)
        self.meta_history = {"spawn_prob": [], "timeout_time": []}
        self.rewards_record = {}
        self.timeout_record = {}
        self.k = k
        self.t = 0
        self.metrics = {"r": [],
                        "equality": [],
                        "sustainability": [],
                        "peace": [],
                        "l": []}

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(META_ACTION.keys()))

    @property
    def observation_space(self):
        obs_space = {
            "curr_obs": gym.spaces.Box(
                low=0,
                high=255,
                shape=self.base_map.shape + (1,),
                dtype=np.uint8)}

        # for when the actions of other agents are part of agents observations space
        if self.return_agent_actions:
            # Append the actions of other agents
            obs_space = {
                **obs_space,
                "other_agent_actions": gym.spaces.Box(
                    low=0,
                    high=len(self.all_actions),
                    shape=(self.num_agents - 1,),
                    dtype=np.uint8,
                ),
                "visible_agents": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.num_agents - 1,),
                    dtype=np.uint8,
                ),
                "prev_visible_agents": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.num_agents - 1,),
                    dtype=np.uint8,
                ),
            }
        obs_space = gym.spaces.Dict(obs_space)
        # Change dtype so that ray can put all observations into one flat batch
        # with the correct dtype.
        # See DictFlatteningPreprocessor in ray/rllib/models/preprocessors.py.
        obs_space.dtype = np.uint8
        return obs_space

    def meta_status(self):
        return {"spawn_prob": self.spawn_prob, "timeout_time": self.timeout_time}

    def meta_action(self, action):
        if action != -1:
            self.spawn_prob = SPAWN_PROB[action]
        self.meta_history["spawn_prob"].append(np.mean(self.spawn_prob))

    def meta_observation(self):
        return rgb2gray(self.map_to_colors(self.get_map_with_agents(), self.color_map))

    def reset(self):
        self.t = 0
        observations = super().reset()
        for agent_id, obs in observations.items():
            observations[agent_id]["curr_obs"] = pad(observations[agent_id]["curr_obs"], self.base_map.shape)
        observations["meta"] = {"curr_obs": self.meta_observation()}
        return observations

    def step(self, actions):
        # self.meta_action(action=actions["meta"])
        actions_ = {key: val for key,val in actions.items() if key != 'meta'}
        nObservations, nRewards, nDone, nInfo = super().step(actions_)
        self.update_social_metrics(nRewards)
        for agent_id, obs in nObservations.items():
            nObservations[agent_id]["curr_obs"] = pad(nObservations[agent_id]["curr_obs"], self.base_map.shape)
        nObservations["meta"] = {"curr_obs": self.meta_observation()}
        nRewards["meta"] = self.compute_efficiency(0, True) * self.compute_peace(0)
        nDone["meta"] = np.any(list(nDone.values()))
        nInfo["meta"] = {}
        return nObservations, nRewards, nDone, nInfo

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = HarvestCommonsAgent(agent_id, spawn_point, rotation, grid, lateral_view_range=self.view_len,
                                        frontal_view_range=self.view_len)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        self.spawn_prob = SPAWN_PROB[2]
        self.timeout_time = copy.copy(TIMEOUT_TIME)
        self.meta_history = {"spawn_prob": [], "timeout_time": []}
        for apple_point in self.apple_points:
            self.world_map[apple_point[0], apple_point[1]] = 'A'
        self.compute_social_metrics()

    def custom_action(self, agent, action):
        agent.fire_beam('F')
        updates = self.update_map_fire(agent.get_pos().tolist(),
                                       agent.get_orientation(),
                                       self.view_len, fire_char='F')
        return updates

    def custom_map_update(self):
        "See parent class"
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)

        # Outcast timed-out agents
        for agent_id, agent in self.agents.items():
            if agent.remaining_timeout > 0:
                agent.remaining_timeout -= 1
                # print("Agent %s its on timeout for %d n_steps" % (agent_id, agent.remaining_timeout))
                if not np.any(agent.pos == OUTCAST_POSITION):
                    self.update_map([[agent.pos[0], agent.pos[1], ' ']])
                    agent.pos = np.array([OUTCAST_POSITION, OUTCAST_POSITION])
            # Return agent to environment
            if agent.remaining_timeout == 0 and np.any(agent.pos == OUTCAST_POSITION):
                # print("%s has finished timeout" % agent_id)
                spawn_point = self.spawn_point()
                spawn_rotation = self.spawn_rotation()
                agent.update_agent_pos(spawn_point)
                agent.update_agent_rot(spawn_rotation)

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in self.agent_pos and self.world_map[row, col] != 'A':
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j ** 2 + k ** 2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if 0 <= x + j < self.world_map.shape[0] and \
                                    self.world_map.shape[1] > y + k >= 0:
                                symbol = self.world_map[x + j, y + k]
                                if symbol == 'A':
                                    num_apples += 1

                spawn_prob = self.spawn_prob[min(num_apples, 3)]
                rand_num = np.random.rand(1)[0]
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, 'A'))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get('A', 0)
        return num_apples

    def update_social_metrics(self, rewards):
        # Save a record of rewards by agent as they are needed for the social metrics computation
        for agent_id, reward in rewards.items():
            if agent_id in self.rewards_record.keys():
                self.rewards_record[agent_id].append(reward)
            else:
                self.rewards_record[agent_id] = [reward]

            is_agent_in_timeout = True if self.agents[agent_id].remaining_timeout > 0 else False
            if agent_id in self.timeout_record.keys():
                self.timeout_record[agent_id].append(is_agent_in_timeout)
            else:
                self.timeout_record[agent_id] = [is_agent_in_timeout]

    def compute_social_metrics(self):
        if len(self.rewards_record) < 1:
            return None

        # Compute sum of rewards
        efficiency = self.compute_efficiency(0)
        equality = self.compute_equality()
        sustainability = self.compute_sustainability()
        peace = self.compute_peace(0)

        self.metrics["r"].append(efficiency)
        self.metrics["equality"].append(equality)
        self.metrics["sustainability"].append(sustainability)
        self.metrics["peace"].append(peace)
        self.metrics["l"].append(self.ep_length)
        self.metrics |= self.meta_history
        self.timeout_record = {}
        self.rewards_record = {}

    def compute_efficiency(self, k, div=False):
        sum_of_rewards = dict(zip(self.agents.keys(), [0] * self.num_agents))
        for agent_id, rewards in self.rewards_record.items():
            sum_of_rewards[agent_id] = np.sum(rewards[-k:])

        agents_sum_rewards = np.sum(list(sum_of_rewards.values()))
        efficiency = agents_sum_rewards / self.num_agents
        if div:
            efficiency /= self.t
        return efficiency

    def compute_peace(self, k):
        timeout_steps = 0
        for agent_id, peace_record in self.timeout_record.items():
            timeout_steps += np.sum(peace_record[-k:])
        peace = (self.num_agents * self.ep_length - timeout_steps) / (self.num_agents * self.ep_length)
        return peace

    def compute_sustainability(self):
        avg_time = 0
        for agent_id, rewards in self.rewards_record.items():
            pos_reward_time_steps = np.argwhere(np.array(rewards) > 0)
            if pos_reward_time_steps.size != 0:
                avg_time += np.mean(pos_reward_time_steps)

        sustainability = avg_time / (self.num_agents * self.ep_length)
        return sustainability

    def compute_equality(self):
        sum_of_rewards = dict(zip(self.agents.keys(), [0] * self.num_agents))
        for agent_id, rewards in self.rewards_record.items():
            sum_of_rewards[agent_id] = np.sum(rewards)

        agents_sum_rewards = np.sum(list(sum_of_rewards.values()))
        sum_of_diff = 0
        for agent_id_a, rewards_sum_a in sum_of_rewards.items():
            for agent_id_b, rewards_sum_b in sum_of_rewards.items():
                sum_of_diff += np.abs(rewards_sum_a - rewards_sum_b)

        equality = 1 - sum_of_diff / (2 * self.num_agents * agents_sum_rewards)
        return equality

    def get_social_metrics(self):
        metrics = {key: safe_mean(value) for key, value in self.metrics.items()}
        self.metrics = {"r": [],
                        "equality": [],
                        "sustainability": [],
                        "peace": [],
                        "l": []}
        return metrics

