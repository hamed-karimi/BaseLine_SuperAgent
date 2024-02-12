import Utils
from Train import Train


if __name__ == '__main__':

    utils = Utils.Utils()
    train_obj = Train(utils=utils)
    train_obj.train_policy()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
