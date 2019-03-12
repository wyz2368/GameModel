from baselines import deepq
from baselines.common import models

DIR_def = './defender_strategies/'
DIR_att = './attacker_strategies/'

#TODO: pick a strategy from a mixed strategy in deeq.learn.
#TODO: add strategy name to strategy name list.
#TODO: extend payoff matrix.
def training_att(env,mix_str_def,epoch):
    env.reset_everything()
    env.set_training_flag = 1
    act_att = deepq.learn(
        env,
        network = models.mlp(num_hidden=256,num_layers=2),
        lr = 5e-5,
        total_timesteps=700000,
        exploration_fraction=0.5,
        exploration_final_eps=0.03,
        print_freq=250,
        param_noise=False,
        gamma=0.99,
        prioritized_replay=True,
        checkpoint_freq=30000,
        opponent_str=mix_str_def
    )
    print("Saving attacker's model to pickle.")
    act_att.save(DIR_att + "att_str_epoch" + str(epoch) + ".pkl")
    return act_att


def training_def(env,mix_str_att,epoch):
    env.reset_everything()
    env.set_training_flag = 0
    act_def = deepq.learn(
        env,
        network = models.mlp(num_hidden=256,num_layers=2),
        lr = 5e-5,
        total_timesteps=700000,
        exploration_fraction=0.5,
        exploration_final_eps=0.03,
        print_freq=250,
        param_noise=False,
        gamma=0.99,
        prioritized_replay=True,
        checkpoint_freq=30000,
        opponent_str = mix_str_att
    )
    print("Saving defender's model to pickle.")
    act_def.save(DIR_def + "def_str_epoch" + str(epoch) + ".pkl")
    return act_def

