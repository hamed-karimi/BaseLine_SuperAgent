import os.path
import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from FeatureExtractor import FeatureExtractor
from CallBack import CallBack
from stable_baselines3.common.callbacks import CheckpointCallback
from Environment import Environment
from ActorCritic import Policy


class Train:
    def __init__(self, utils):
        self.params = utils.params
        self.policy_net_size = self.params.POLICY_NET_SIZE
        self.value_net_size = self.params.VALUE_NET_SIZE
        self.episode_num = self.params.EPISODE_NUM
        self.batch_size = self.params.BATCH_SIZE
        self.step_num = self.params.EPISODE_STEPS
        self.device = self.params.DEVICE
        self.res_folder = utils.make_res_folder()
        self.log_dir = os.path.join(self.res_folder, 'log')
        self.tensorboard_call_back = CallBack(log_freq=self.params.PRINT_REWARD_FREQ, )
        # environment = Environment(params, ['many', 'many'])
        # check_env(environment)

    def train_policy(self):
        vec_env = make_vec_env(Environment, n_envs=self.batch_size,
                               env_kwargs=dict(params=self.params,
                                               few_many_objects=['few', 'many']))
        vec_env = VecMonitor(venv=vec_env, filename=self.log_dir)
        checkpoint_callback = CheckpointCallback(
            save_freq=self.params.CHECKPOINT_SAVE_FREQUENCY,
            save_path=self.res_folder,
            name_prefix="A2C",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        policy_kwargs = dict(
            features_extractor_class=FeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        model = A2C(Policy,
                    vec_env,
                    policy_kwargs=policy_kwargs,
                    learning_rate=self.params.INIT_LEARNING_RATE,
                    gamma=self.params.GAMMA,
                    verbose=0,
                    n_steps=self.step_num,
                    tensorboard_log='./runs',
                    device=self.device)

        model.learn(self.episode_num,
                    callback=[self.tensorboard_call_back, checkpoint_callback],
                    tb_log_name=self.res_folder)
        model.save(os.path.join(self.res_folder, 'model'))
