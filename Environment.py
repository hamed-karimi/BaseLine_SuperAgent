import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from Controller import Controller


class Environment(gym.Env):
    def __init__(self, params, few_many_objects):
        self.params = params
        self.height = params.HEIGHT
        self.width = params.WIDTH
        self.few_many_objects = few_many_objects
        self.metadata = {"render_modes": None}
        self.object_type_num = params.OBJECT_TYPE_NUM
        self._no_reward_threshold = -5
        self._goal_selection_step = 0
        self.cost_of_non_object_location = 100
        self._env_map = np.zeros((1 + self.object_type_num, self.height, self.width), dtype=int)
        self._mental_states = np.empty((self.object_type_num,), dtype=np.float64)
        self._mental_states_slope = np.empty((self.object_type_num,), dtype=np.float64)
        self._environment_object_reward = np.empty((self.object_type_num,), dtype=np.float64)
        self._environment_states_parameters = [self._mental_states_slope, self._environment_object_reward]
        self._environment_states_parameters_range = [self.params.MENTAL_STATES_SLOPE_RANGE,
                                                     self.params.ENVIRONMENT_OBJECT_REWARD_RANGE]

        self.controller = Controller(self.height, self.width)

        self.observation_space = spaces.flatten_space(spaces.Tuple(
            # Usually, it will not be possible to use elements of this space directly in learning code. However, you can easily convert Dict observations to flat arrays by using a gymnasium.wrappers.FlattenObservation wrapper
            [spaces.Box(0, 1, shape=(1 + self.object_type_num,
                                     self.height, self.width), dtype=int),  # 'env_map'
             spaces.Box(self.params.INITIAL_MENTAL_STATES_RANGE[0], 2**63 - 2,
                        shape=(self.object_type_num,), dtype=float),  # 'mental_states'
             spaces.Box(self.params.MENTAL_STATES_SLOPE_RANGE[0], self.params.MENTAL_STATES_SLOPE_RANGE[1],
                        shape=(self.object_type_num,), dtype=float),  # 'mental_states_slope'
             spaces.Box(self.params.ENVIRONMENT_OBJECT_REWARD_RANGE[0], self.params.ENVIRONMENT_OBJECT_REWARD_RANGE[1],
                        shape=(self.object_type_num,), dtype=float)]  # 'environment_object_reward'
        ))
        self.action_space = spaces.MultiDiscrete([self.height, self.width])

    def sample(self):  # A dictionary with the same key and sampled values from :attr:`self.spaces`
        self._env_map = np.zeros_like(self._env_map, dtype=int)
        self._init_random_map()
        self._init_random_mental_states()
        self._init_random_parameters()
        flat_observation = self._flatten_observation()
        return flat_observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._env_map = np.zeros_like(self._env_map, dtype=int)
        sample_observation = self.sample()

        return sample_observation, dict()

    def step(self, goal_location):
        # print(goal_location)
        # (observation, reward, terminated, truncated, info)

        steps = self.controller.get_shortest_path_to_object(np.expand_dims(self._agent_location, axis=0),
                                                            np.expand_dims(goal_location, axis=0))
        if self._env_map[1:, goal_location[0], goal_location[1]].sum() == 0:
            reward = -1 * self.cost_of_non_object_location
        else:
            reward = 0
        for step in steps:
            # left_location = self._agent_location.copy()
            self._update_agent_locations(step)
            step_length = np.linalg.norm(step)
            dt = np.array(1) if step_length < 1.4 else step_length
            mental_states_cost = self._total_positive_mental_states() * dt
            object_reward = self._env_map[1:, self._agent_location[0], self._agent_location[1]] * self._environment_object_reward
            self._update_object_locations()

            self._update_mental_state_after_step(dt=dt)
            positive_mental_states_before_reward = self._total_positive_mental_states()
            self._update_mental_states_after_object(u=object_reward)
            positive_mental_states_after_reward = self._total_positive_mental_states()
            mental_states_reward = np.maximum(0, positive_mental_states_before_reward - positive_mental_states_after_reward)
            step_reward = mental_states_reward - step_length - mental_states_cost
            reward += step_reward

        terminated = True  # be careful about this, we might need to try to have always (or after 5 goal selection step) terminated=False, and just maximize the reward.
        # (observation, reward, terminated, truncated, info)
        return self._flatten_observation(), reward, terminated, False, dict()

    def render(self):
        return None

    def _flatten_observation(self):
        observation = [self._env_map.flatten(), self._mental_states.flatten()]
        for i in range(len(self._environment_states_parameters)):
            observation.append(self._environment_states_parameters[i].flatten())
        return np.concatenate(observation)

    def _update_agent_locations(self, step):
        self._env_map[0, self._agent_location[0], self._agent_location[1]] = 0
        self._agent_location += step
        self._env_map[0, self._agent_location[0], self._agent_location[1]] = 1

    def _update_object_locations(self):
        if self._env_map[1:, self._agent_location[0], self._agent_location[1]].sum() == 0: # not reached an object
            return
        reached_object_type = np.argwhere(self._env_map[1:, self._agent_location[0], self._agent_location[1]])[0, 0]
        self.each_type_object_num[reached_object_type] += 1
        self._init_random_map(object_num_on_map=self.each_type_object_num)  # argument is kind of redundant
        self._env_map[reached_object_type+1, self._agent_location[0], self._agent_location[1]] = 0
        self.each_type_object_num[reached_object_type] -= 1

    def _total_positive_mental_states(self):
        total_need = np.maximum(0, self._mental_states).sum()
        return total_need

    def _update_mental_state_after_step(self, dt):
        dz = (self._mental_states_slope * dt)
        self._mental_states += dz

    def _update_mental_states_after_object(self, u):  # u > 0
        self._mental_states += -(1 * u)
        self._mental_states = np.maximum(self._mental_states, self._no_reward_threshold)

    def _init_object_num_on_map(self) -> np.array:
        # e.g., self.few_many_objects : ['few', 'many']
        few_range = np.array([1, 2, 3, 4])
        many_range = np.array([1, 2, 3, 4])
        ranges = {'few': few_range,
                  'many': many_range}
        each_type_object_num = np.zeros((self.object_type_num,), dtype=int)
        for i, item in enumerate(self.few_many_objects):
            at_type_obj_num = np.random.choice(ranges[item])
            each_type_object_num[i] = at_type_obj_num

        return each_type_object_num

    def _init_random_map(self, object_num_on_map=None):  # add agent location
        if self._env_map[0, :, :].sum() == 0:  # no agent on map
            self._agent_location = np.random.randint(low=0, high=self.height, size=(2,))
            self._env_map[0, self._agent_location[0], self._agent_location[1]] = 1

        object_num_already_on_map = self._env_map[1:, :, :].sum(axis=(1, 2))
        if object_num_on_map is None:
            self.each_type_object_num = self._init_object_num_on_map()
        else:
            self.each_type_object_num = object_num_on_map
        object_num_to_init = self.each_type_object_num - object_num_already_on_map
        # temp_object_locations = -1 * np.ones((self.each_type_object_num.sum(), 3), dtype=int)
        # self.object_locations = -1 * np.ones((self.each_type_object_num.sum(), 3), dtype=int)  # (object_type, x, y)
        # self.object_locations = dict()
        # for obj_type in range(self.object_type_num):
        #     if not obj_type in self.object_locations.keys():
        #         self.object_locations[obj_type] = []
        object_count = 0
        for obj_type in range(self.object_type_num):
            for at_obj in range(object_num_to_init[obj_type]):
                while True:
                    sample_location = np.random.randint(low=0, high=[self.height, self.width],
                                                        size=(self.object_type_num,))
                    if self._env_map[:, sample_location[0], sample_location[1]].sum() == 0:
                        self._env_map[1 + obj_type, sample_location[0], sample_location[1]] = 1
                        break

                object_count += 1

    def _init_random_mental_states(self):
        self._mental_states[:] = self._get_random_vector(attr_range=self.params.INITIAL_MENTAL_STATES_RANGE,
                                                         prob_equal=self.params.PROB_EQUAL_PARAMETERS)

    def _init_random_parameters(self):
        for i in range(len(self._environment_states_parameters_range)):
            self._environment_states_parameters[i][:] = self._get_random_vector(
                attr_range=self._environment_states_parameters_range[i],
                prob_equal=self.params.PROB_EQUAL_PARAMETERS)

    def _get_random_vector(self, attr_range, prob_equal=0):
        p = random.uniform(0, 1)
        if p <= prob_equal:
            size = 1
        else:
            size = self.object_type_num

        return np.random.uniform(low=attr_range[0],
                                 high=attr_range[1],
                                 size=(size,))
