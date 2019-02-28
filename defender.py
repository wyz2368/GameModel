import random
import numpy as np

class Defender(object):

    def __init__(self, G):
        self.observation = []
        self.prev_obs = [0]*G.number_of_nodes*self.history

        self.defact = set()
        self.prev_defact = [set()]*self.history

        self.history = 2
        self.rand_limit = 4

    def def_greedy_action_builder(self, G, timeleft, nn_def):
        self.defact.clear()
        isDup = False
        while not isDup:
            def_input = self.def_obs_constructor(G, timeleft)
            x = nn_def(def_input[None])[0] #corrensponding to baselines
            if not isinstance(x,int):
                raise ValueError("The chosen action is not an integer.")
            action_space = self.get_def_actionspace(G)
            action = action_space[x-1]
            if action == 'pass':
                break
            isDup = (action in self.defact)
            if not isDup:
                self.defact.add(action)

    #TODO: Since initialize with zeros, clean these funcs.
    def def_obs_constructor(self, G, timeleft):
        wasdef = self.get_def_wasDefended(G)
        indef = self.get_def_inDefenseSet(G)
        # history_length = len(self.prev_obs)
        # if history_length == 0:
        #     self.prev_obs = self.prev_obs + [0]*G.number_of_nodes*self.history
        # if history_length == 1:
        #     self.prev_obs = [0]*G.number_of_nodes + self.prev_obs
        def_input = self.prev_obs + self.observation + wasdef + indef + [timeleft]
        return np.array(def_input)

    def get_def_wasDefended(self, G):
        wasdef = []
        # history_length = len(self.prev_defact)
        # if history_length == 0:
        #     return [0]*G.number_of_nodes*self.history
        # elif history_length == 1:
        #     for obs in self.prev_defact:
        #         for node in G.nodes:
        #             if node in obs:
        #                 wasdef.append(1)
        #             else:
        #                 wasdef.append(0)
        #     return [0]*G.number_of_nodes + wasdef
        # else:
        for obs in self.prev_defact:
            for node in G.nodes:
                if node in obs:
                    wasdef.append(1)
                else:
                    wasdef.append(0)
        return wasdef

    def get_def_inDefenseSet(self, G):
        indef = []
        for node in G.nodes:
            if node in self.defact:
                indef.append(1)
            else:
                indef.append(0)
        return indef

    def get_def_actionspace(self, G):
        num_nodes = G.number_of_nodes()
        actionspace = [i+1 for i in range(num_nodes)] + ['pass']
        return actionspace

    def uniform_strategy(self, G):
        return set(random.choices(list(G.nodes), k = self.rand_limit))

    def cut_prev_obs(self):
        if len(self.prev_obs) > self.history:
            self.prev_obs = self.prev_obs[-self.history:]

    def cut_prev_defact(self):
        if len(self.prev_defact) > self.history:
            self.prev_defact = self.prev_defact[-self.history:]

    def save_defact2prev(self):
        self.prev_defact.append(self.defact)
        self.cut_prev_defact()

    def update_obs(self, obs):
        self.prev_obs.append(self.observation)
        self.cut_prev_obs()
        self.observation = obs
    # TODO: in env, feed noisy obs to the observation

    def update_history(self, history):
        self.history = history



    def reset_def(self, G):
        self.observation = []
        self.prev_obs = [0]*G.number_of_nodes*self.history
        self.defact.clear()
        self.prev_defact = [set()]*self.history

