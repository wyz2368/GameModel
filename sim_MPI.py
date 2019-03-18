from mpi4py import MPI
from parallel_sim import parallel_sim
from rand_strategies_payoff import rand_parallel_sim


def sim_and_modifiy_MPI(game):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    old_dim, old_dim1 = game.dim_payoff_def()
    new_dim, new_dim1 = game.num_str()
    if old_dim != old_dim1 or new_dim != new_dim1:
        raise ValueError("Payoff dimension does not match.")

    def_str_list = game.def_str
    att_str_list = game.att_str
    new_str_def = def_str_list[-1]
    new_str_att = att_str_list[-1]
    if new_dim <= size:
        return 0
    else:
        return 1



