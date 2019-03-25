import numpy as np
import file_op as fp
from baselines import deepq
from baselines.common import models
import os

DIR = os.getcwd() + '/'
DIR_def = os.getcwd() + '/defender_strategies/'
DIR_att = os.getcwd() + '/attacker_strategies/'

def sample_strategy_from_mixed(env, str_set, mix_str, identity):

    if not isinstance(mix_str,np.ndarray):
        raise ValueError("mix_str in sample func is not a numpy array.")

    if not len(str_set) == len(mix_str):
        raise ValueError("Length of mixed strategies does not match number of strategies.")

    picked_str = np.random.choice(str_set,p=mix_str)
    if not fp.isInName('.pkl', name = picked_str):
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
        load_path= path + picked_str
    )

    return act

def sample_both_strategies(env, att_str_set, att_mix_str, def_str_set, def_mix_str):

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
        load_path=path_att + att_picked_str
    )

    act_def = deepq.learn(
        env,
        network=models.mlp(num_hidden=256, num_layers=1),
        total_timesteps=0,
        load_path= path_def + def_picked_str
    )

    return act_att, act_def

#TODO: check the input dim of nn and check if this could initialize nn.
def rand_str_generator(env, game):
    # Generate random nn for attacker.
    num_layers = game.num_layers
    num_hidden = game.num_hidden

    act_att = deepq.learn(
        env,
        network=models.mlp(num_hidden=num_hidden, num_layers=num_layers-3),
        total_timesteps=0
    )

    act_def = deepq.learn(
        env,
        network=models.mlp(num_hidden=num_hidden, num_layers=num_layers-3),
        total_timesteps=0
    )

    print("Saving attacker's model to pickle. Epoch name is equal to 1.")
    act_att.save(DIR_att + "att_str_epoch" + str(1) + ".pkl")
    game.att_str.append("att_str_epoch" + str(1) + ".pkl")

    print("Saving defender's model to pickle. Epoch in name is equal to 1.")
    act_def.save(DIR_def + "def_str_epoch" + str(1) + ".pkl")
    game.def_str.append("def_str_epoch" + str(1) + ".pkl")