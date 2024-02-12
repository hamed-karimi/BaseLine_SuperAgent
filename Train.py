import os.path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from CallBack import CallBack
from Environment import Environment


class Train:
    def __init__(self, utils):
        self.params = utils.params
        self.policy_net_size = self.params.POLICY_NET_SIZE
        self.value_net_size = self.params.VALUE_NET_SIZE
        self.episode_num = self.params.EPISODE_NUM
        self.batch_size = self.params.BATCH_SIZE
        self.device = 'cpu'
        self.res_folder = utils.make_res_folder()
        self.log_dir = os.path.join(self.res_folder, 'log')
        self.call_back = CallBack(log_freq=self.params.PRINT_REWARD_FREQ, )
        # environment = Environment(params, ['many', 'many'])
        # check_env(environment)

    def train_policy(self):
        vec_env = make_vec_env(Environment, n_envs=self.batch_size, env_kwargs=dict(params=self.params,
                                                                       few_many_objects=['few', 'many']))
        vec_env = VecMonitor(venv=vec_env, filename=self.log_dir)
        policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                             net_arch=dict(pi=self.policy_net_size,
                                           vf=self.value_net_size))
        # Create the agent
        model = PPO("MlpPolicy",
                    vec_env,
                    policy_kwargs=policy_kwargs,
                    learning_rate=self.params.LEARNING_RATE,
                    gamma=self.params.GAMMA,
                    batch_size=self.params.BATCH_SIZE,
                    verbose=1,
                    n_steps=1,
                    n_epochs=1,
                    tensorboard_log='./runs',
                    device='auto')

        # model = A2C("MlpPolicy", vec_env, verbose=0)
        model.learn(self.episode_num, callback=self.call_back, tb_log_name=self.res_folder)
        model.save(self.res_folder)
