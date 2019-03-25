from baselines import deepq
from baselines.common import models
import os
#TODO: improvement can be done by not including all RL strategies.

DIR_def = os.getcwd() + '/defender_strategies/'
DIR_att = os.getcwd() + '/attacker_strategies/'

#TODO: pick a strategy from a mixed strategy.
#TODO: add strategy name to strategy name list.
#TODO: extend payoff matrix.
#TODO: network model should be rechecked.
def training_att(env, game, mix_str_def, epoch):
    env.reset_everything()
    env.set_training_flag = 1

    env.defender.set_mix_strategy(mix_str_def)
    env.defender.set_str_set(game.def_str)

    act_att = deepq.learn(
        env,
        network = models.mlp(num_hidden=256,num_layers=1),
        lr = 5e-5,
        total_timesteps=700000,
        exploration_fraction=0.5,
        exploration_final_eps=0.03,
        print_freq=250,
        param_noise=False,
        gamma=0.99,
        prioritized_replay=True,
        checkpoint_freq=30000
    )
    print("Saving attacker's model to pickle.")
    act_att.save(DIR_att + "att_str_epoch" + str(epoch) + ".pkl")
    game.att_str.append("att_str_epoch" + str(epoch) + ".pkl")




def training_def(env, game, mix_str_att, epoch):
    env.reset_everything()
    env.set_training_flag = 0

    env.attacker.set_mix_strategy(mix_str_att)
    env.attacker.set_str_set(game.att_str)

    act_def = deepq.learn(
        env,
        network = models.mlp(num_hidden=256,num_layers=1),
        lr = 5e-5,
        total_timesteps=700000,
        exploration_fraction=0.5,
        exploration_final_eps=0.03,
        print_freq=250,
        param_noise=False,
        gamma=0.99,
        prioritized_replay=True,
        checkpoint_freq=30000
    )
    print("Saving defender's model to pickle.")
    act_def.save(DIR_def + "def_str_epoch" + str(epoch) + ".pkl")
    game.def_str.append("def_str_epoch" + str(epoch) + ".pkl")


