from mpi4py import MPI
from parallel_sim import parallel_sim
from rand_strategies_payoff import rand_parallel_sim
from deepq import load_action
import file_op as fp


#TODO: assign epoch
def sim_and_modifiy_MPI():
    #TODO: load game
    path = './attackgraph/data/game.pkl'
    game = fp.load_pkl(path)
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    data = sim_and_modifiy(game, rank, size)

    newData = comm.gather(data, root=0)
    if rank == 0:
        fp.save_pkl(newData,path='./attackgraph/data/newdata_' + str(epoch) + ".pkl")

def sim_and_modifiy(game, rank, size):
    env = game.env
    num_episodes = game.num_episodes

    old_dim, old_dim1 = game.dim_payoff_def()
    new_dim, new_dim1 = game.num_str()
    if old_dim != old_dim1 or new_dim != new_dim1:
        raise ValueError("Payoff dimension does not match.")

    def_str_list = game.def_str
    att_str_list = game.att_str
    dir_def = game.dir_def
    dir_att = game.dir_att

    position_list = []
    for i in range(new_dim):
        position_list.append((i,new_dim-1))
    for j in range(new_dim-1):
        position_list.append((new_dim-1,j))

    num_tasks = 2 * new_dim - 1
    data = {}
    if num_tasks <= size:
        if rank < num_tasks:
            idx_def, idx_att = position_list[rank]
            str_path_def = dir_def + def_str_list[idx_def]
            str_path_att = dir_att + att_str_list[idx_att]
            nn_def = load_action(str_path_def, game)
            nn_att = load_action(str_path_att,game)
            data[position_list[rank]] = parallel_sim(env,nn_att, nn_def, num_episodes)
    else:
        num_task_per_proc = num_tasks // size
        extra_num_tasks = num_tasks % size
        num_task_per_proc_p1 = num_task_per_proc + 1

        #TODO: Does not finish.
        if rank < extra_num_tasks:
            for i in range(num_task_per_proc_p1):
                idx_def, idx_att = position_list[rank*num_task_per_proc_p1+i]
                str_path_def = dir_def + def_str_list[idx_def]
                str_path_att = dir_att + att_str_list[idx_att]
                nn_def = load_action(str_path_def, game)
                nn_att = load_action(str_path_att, game)
                data[position_list[rank*num_task_per_proc_p1+i]] = parallel_sim(env,nn_att, nn_def, num_episodes)
        else:
            for i in range(num_task_per_proc):
                idx_def, idx_att = position_list[rank*num_task_per_proc+1]
                str_path_def = dir_def + def_str_list[idx_def]
                str_path_att = dir_att + att_str_list[idx_att]
                nn_def = load_action(str_path_def, game)
                nn_att = load_action(str_path_att, game)
                data[position_list[rank*num_task_per_proc+1]] = parallel_sim(env,nn_att, nn_def, num_episodes)

    return data








