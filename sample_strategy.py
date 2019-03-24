import numpy as np
import file_op as fp
from baselines import deepq
from baselines.common import models
import os

DIR = os.getcwd() + '/'
#TODO: check all path correct.
def sample_strategy_from_mixed(env, str_set, mix_str, identity):
    #TODO: Add path
    #TODO: explain opponent_str is None
    if not len(str_set) == len(mix_str):
        raise ValueError("Length of mixed strategies does not match number of strategies.")

    picked_str = np.random.choice(str_set,p=mix_str)
    if not fp.isInName('.pkl',name=picked_str):
        raise ValueError('The strategy picked is not a pickle file.')

    if identity == 0: # pick a defender's strategy
        path = DIR + 'defender_strategies/'
    elif identity == 1:
        path = DIR + 'attacker_strategies/'
    else:
        raise ValueError("identity is neither 0 or 1!")

    if not fp.isExist(path + picked_str):
        raise ValueError('The strategy picked does not exist!')

    #TODO: assign nn info from game
    act = deepq.learn(
        env,
        network=models.mlp(num_hidden=256, num_layers=1),
        total_timesteps=0,
        load_path=path+picked_str,
        opponent_str = None
    )

    return act

def sample_both_strategies(env, att_str_set, att_mix_str, def_str_set, def_mix_str):
    # TODO: Add path
    # TODO: explain opponent_str is None
    if not len(att_str_set) == len(att_mix_str):
        raise ValueError("Length of mixed strategies does not match number of strategies for the attacker.")
    if not len(def_str_set) == len(def_mix_str):
        raise ValueError("Length of mixed strategies does not match number of strategies for the defender.")

    att_picked_str = np.random.choice(att_str_set, p=att_mix_str)
    def_picked_str = np.random.choice(def_str_set, p=def_mix_str)

    if not fp.isInName('.pkl', name=def_picked_str):
        raise ValueError('The strategy picked is not a pickle file for the defender.')
    if not fp.isInName('.pkl', name=att_picked_str):
        raise ValueError('The strategy picked is not a pickle file for the attacker.')

    path_def = DIR + 'defender_strategies/'
    path_att = DIR + 'attacker_strategies/'

    if not fp.isExist(path_def + def_picked_str):
        raise ValueError('The strategy picked does not exist for the defender!')
    if not fp.isExist(path_att + att_picked_str):
        raise ValueError('The strategy picked does not exist for the attacker!')

    act_att = deepq.learn(
        env,
        network=models.mlp(num_hidden=256, num_layers=1),
        total_timesteps=0,
        load_path=path_att + att_picked_str,
        opponent_str=None
    )

    act_def = deepq.learn(
        env,
        network=models.mlp(num_hidden=256, num_layers=1),
        total_timesteps=0,
        load_path=path_def + def_picked_str,
        opponent_str=None
    )

    return act_att, act_def