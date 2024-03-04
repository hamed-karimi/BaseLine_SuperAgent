import math

from stable_baselines3 import PPO
import os
import torch
import numpy as np
import itertools
from Environment import Environment
import matplotlib.pyplot as plt


def get_predefined_parameters(num_object, param_name):
    if param_name == 'all_mental_states':
        all_param = [[-10, -5, 0, 5, 10]] * num_object
    elif param_name == 'all_object_rewards':
        all_param = [[0, 4, 8, 12, 16, 20]] * num_object
    elif param_name == 'all_mental_states_change':
        all_param = [[0, 1, 2, 3, 4, 5]] * num_object
    else:
        print('no such parameters')
        return
    num_param = len(all_param[0]) ** num_object
    param_batch = []
    for i, ns in enumerate(itertools.product(*all_param)):
        param_batch.append(list(ns))
    return param_batch


class Test:
    def __init__(self, utils):
        self.params = utils.params
        self.res_folder = utils.res_folder
        self.model = self.load_model()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = utils.params.HEIGHT
        self.width = utils.params.WIDTH
        self.object_type_num = utils.params.OBJECT_TYPE_NUM
        # self.episode_num = utils.params.META_CONTROLLER_EPISODE_NUM
        self.all_actions = torch.tensor([[0, 0],
                                         [1, 0], [-1, 0], [0, 1], [0, -1],
                                         [1, 1], [-1, -1], [-1, 1], [1, -1]])
        self.action_mask = np.zeros((self.height, self.width, 1, len(self.all_actions)))
        self.initialize_action_masks()

        self.all_mental_states = get_predefined_parameters(self.object_type_num, 'all_mental_states')
        self.all_object_rewards = get_predefined_parameters(self.object_type_num, 'all_object_rewards')
        self.all_mental_states_change = get_predefined_parameters(self.object_type_num, 'all_mental_states_change')

        self.color_options = [[1, 0, .2], [0, .8, .2], [0, 0, 0]]
        self.goal_shape_options = ['*', 's', 'P', 'o', 'D', 'X']
        self.objects_color_name = ['red', 'green', 'black']  # 2: stay
        # self.row_num = 5
        # self.col_num = 6

    def initialize_action_masks(self):
        for i in range(self.height):
            for j in range(self.width):
                agent_location = torch.tensor([[i, j]])
                aa = np.ones((agent_location.size(0), len(self.all_actions)))
                for ind, location in enumerate(agent_location):
                    if location[0] == 0:
                        aa[ind, 2] = 0
                        aa[ind, 6] = 0
                        aa[ind, 7] = 0
                    if location[0] == self.height - 1:
                        aa[ind, 1] = 0
                        aa[ind, 5] = 0
                        aa[ind, 8] = 0
                    if location[1] == 0:
                        aa[ind, 4] = 0
                        aa[ind, 6] = 0
                        aa[ind, 8] = 0
                    if location[1] == self.width - 1:
                        aa[ind, 3] = 0
                        aa[ind, 5] = 0
                        aa[ind, 7] = 0
                self.action_mask[i, j, :, :] = aa

    def get_figure_title(self, mental_states):
        title = '$n_{0}: {1:.2f}'.format('{' + self.objects_color_name[0] + '}', mental_states[0])
        for i in range(1, self.object_type_num):
            title += ", n_{0}: {1:.2f}$".format('{' + self.objects_color_name[i] + '}', mental_states[i])
        return title

    def get_object_shape_dictionary(self, object_locations, agent_location, each_type_object_num):
        shape_map = dict()
        for obj_type in range(self.object_type_num):
            at_type_object_locations = object_locations[object_locations[:, 0] == obj_type]
            for at_obj in range(each_type_object_num[obj_type]):
                key = tuple(at_type_object_locations[at_obj, 1:].tolist())
                shape_map[key] = self.goal_shape_options[at_obj]
        key = tuple(agent_location)
        shape_map[key] = '.'
        return shape_map

    def get_goal_location_from_values(self, env_map, values):
        goal_values = values.reshape(self.params.HEIGHT, self.params.WIDTH)
        object_mask = env_map.sum(axis=0) > 0
        goal_values[~object_mask] = -math.inf
        goal_location = np.array(np.unravel_index(goal_values.argmax(), goal_values.shape))
        return goal_location

    def next_agent_and_environment(self):
        for object_reward in self.all_object_rewards:
            for mental_state_slope in self.all_mental_states_change:
                environment = Environment(self.params, ['few', 'many'])

                for subplot_id, mental_state in enumerate(self.all_mental_states):
                    for i in range(self.height):
                        for j in range(self.width):
                            env, flat_env, object_locations, each_type_object_num = environment.init_environment_for_test(
                                [i, j],
                                mental_state,
                                mental_state_slope,
                                object_reward)
                            env_parameters = [mental_state, mental_state_slope, object_reward]
                            yield env, flat_env, [i, j], object_locations, each_type_object_num, env_parameters, subplot_id

    def get_goal_directed_actions(self):
        fig, ax = None, None
        which_goal = None
        row_num = 5
        col_num = 5
        created_subplot = np.zeros((row_num * col_num,), dtype=bool)
        for setting_id, outputs in enumerate(self.next_agent_and_environment()):
            environment = outputs[0]
            flat_environment = outputs[1]
            agent_location = outputs[2]
            object_locations = outputs[3]
            each_type_object_num = outputs[4]
            env_parameters = outputs[5]  # [mental_state, mental_state_slope, object_reward]
            subplot_id = outputs[6]

            # print('ms: ', agent.mental_states, 'or: ', environment.object_reward, 'sc: ', agent.states_change)
            if setting_id % (col_num * row_num * self.width * self.height) == 0:
                fig, ax = plt.subplots(row_num, col_num, figsize=(15, 12))
                # created_subplot[subplot_id] = True
            if setting_id % (self.height * self.width) == 0:
                which_goal = np.empty((self.height, self.width), dtype=str)
            # else:
            #     continue

            r = subplot_id // col_num
            c = subplot_id % col_num

            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            ax[r, c].invert_yaxis()

            shape_map = self.get_object_shape_dictionary(object_locations, agent_location, each_type_object_num)

            with torch.no_grad():
                goal_values = self.model.predict(observation=flat_environment, deterministic=False)[0]
                goal_location = self.get_goal_location_from_values(env_map=environment, values=goal_values)

            if tuple(goal_location.tolist()) in shape_map.keys():
                # which_goal[agent_location[0], agent_location[1]] = shape_map[tuple(goal_location.tolist())]
                selected_goal_shape = shape_map[tuple(goal_location.tolist())]
                goal_type = np.where(environment[:, goal_location[0], goal_location[1]])[0].min()
            else:
                # which_goal[agent_location[0], agent_location[1]] = '_'
                selected_goal_shape = '_'
                goal_type = 0

            goal_type = 2 if goal_type == 0 else goal_type - 1
            # selected_goal_shape = shape_map[tuple(goal_location.tolist())]
            size = 10 if selected_goal_shape == '.' else 50
            ax[r, c].scatter(agent_location[1], agent_location[0],
                             marker=selected_goal_shape,
                             s=size,
                             alpha=0.4,
                             facecolor=self.color_options[goal_type])

            if agent_location[0] == self.height - 1 and agent_location[1] == self.width - 1:
                # which_goal = np.empty((self.height, self.width), dtype=str)
                ax[r, c].set_title(self.get_figure_title(env_parameters[0]), fontsize=10)

                # for obj_type in range(self.object_type_num):
                #     at_type_object_locations = object_locations[object_locations[:, 0] == obj_type]
                #     for at_obj in range(each_type_object_num[obj_type]):

                for obj_type in range(self.object_type_num):
                    at_type_object_locations = object_locations[object_locations[:, 0] == obj_type]
                    for obj in range(each_type_object_num[obj_type]):
                        # if environment.object_locations[obj_type, obj, 0] == -1:
                        #     break
                        ax[r, c].scatter(at_type_object_locations[obj, 1:][1],
                                         at_type_object_locations[obj, 1:][0],
                                         marker=self.goal_shape_options[obj],
                                         s=200,
                                         edgecolor=self.color_options[obj_type],
                                         facecolor='none')
                ax[r, c].tick_params(length=0)
                ax[r, c].set(adjustable='box')
            if (setting_id + 1) % (col_num * row_num * self.width * self.height) == 0:
                plt.tight_layout(pad=0.1, w_pad=6, h_pad=1)
                fig.savefig('{0}/slope_{1}-{2}_or_{3}-{4}.png'.format(self.res_folder,
                                                                      env_parameters[1][0],
                                                                      env_parameters[1][1],
                                                                      env_parameters[2][0],
                                                                      env_parameters[2][1]))
                plt.close()

    def load_model(self):
        model_path = os.path.join(self.res_folder, 'model.zip')
        return PPO.load(path=model_path)
