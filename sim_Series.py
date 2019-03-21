from parallel_sim import parallel_sim
from rand_strategies_payoff import rand_parallel_sim
from load_action import load_action
import file_op as fp

#TODO: make sure every
def sim_and_modifiy_Series():
    print('Begin simulation and modify payoff matrix.')
    path = './attackgraph/data/game.pkl'
    game = fp.load_pkl(path)
    data = sim_and_modifiy(game)

    #TODO: modify payoff matrix
    old_dim, _ = game.dim_payoff_def()
    new_dim, _ = game.num_str()

    new_col_def = []
    new_row_def = []
    new_col_att = []
    new_row_att = []

    for i in range(old_dim):
        aReward_col, dReward_col = data[(i,new_dim-1)]
        aReward_row, dReward_row = data[(new_dim - 1,i)]
        new_col_def.append([dReward_col])
        new_row_def.append(dReward_row)
        new_col_att.append([aReward_col])
        new_row_att.append(aReward_row)

    aReward, dReward = data[(new_dim - 1,new_dim - 1)]
    new_row_def.append(dReward)
    new_row_att.append(aReward)
    print("Done simulation and modify payoff matrix.")




def sim_and_modifiy(game):
    env = game.env
    num_episodes = game.num_episodes

    #TODO: add str first and then calculate payoff
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

    # num_tasks = 2 * new_dim - 1
    data = {}

    #TODO: check the path is correct
    for pos in position_list:
        idx_def, idx_att = pos
        str_path_def = dir_def + def_str_list[idx_def]
        str_path_att = dir_att + att_str_list[idx_att]
        nn_def = load_action(str_path_def, game)
        nn_att = load_action(str_path_att, game)
        data[pos] = parallel_sim(env, nn_att, nn_def, num_episodes)

    return data