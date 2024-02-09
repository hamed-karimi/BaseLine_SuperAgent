import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from types import SimpleNamespace
import json

import Utils
from Environment import Environment
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.env_util import make_vec_env
from Train import Train


if __name__ == '__main__':

    utils = Utils.Utils()
    train_obj = Train(utils=utils)
    train_obj.train_policy()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
