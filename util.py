import random
import numpy as np

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def payoff_mixed_NE(game, epoch):
    payoffmatrix_def = game.payoffmatrix_def
    payoffmatrix_att = game.payoffmatrix_att
    dim_row_def, dim_col_def = np.shape(payoffmatrix_def)
    dim_row_att, dim_col_att = np.shape(payoffmatrix_att)

    if dim_row_def != dim_col_def or dim_row_att != dim_col_att:
        raise ValueError("Dimension of payoff matrix does not match!")

    if dim_row_def != dim_row_att:
        raise ValueError("Dimension of payoff matrix does not match!")

    if epoch > dim_col_att:
        raise ValueError("Epoch exceeds the payoffmatrix dimension.")

    if not epoch in game.nasheq.keys():
        raise ValueError("Epoch is not a key of current nasheq.")

    ne = game.nasheq[epoch]
    nash_def = ne[0]
    nash_att = ne[1]
    if len(nash_def) != len(nash_att):
        raise ValueError("Length of nash does not match!")
    num_str = len(nash_def)
    sub_payoffmatrix_def = payoffmatrix_def[:num_str,:num_str]
    sub_payoffmatrix_att = payoffmatrix_att[:num_str,:num_str]

    nash_def = np.reshape(nash_def,newshape=(num_str,1))

    dPayoff = np.round(np.sum(nash_def * sub_payoffmatrix_def * nash_att), decimals=2)
    aPayoff = np.round(np.sum(nash_def * sub_payoffmatrix_att * nash_att), decimals=2)

    return aPayoff, dPayoff