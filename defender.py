import random

class Defender(object):

    def __init__(self):
        self.rand_limit = 4
        self.observation = []
        self.prev_obs = []
        self.defact = set()
        self.defact_tm1 = set()
        self.prev_defact = []

    def def_greedy_action_builder(self, G, nn_def):
        self.defact.clear()
        isDup = 0
        x = 0
        while not isDup:
            def_input = self.def_obs_constructor(G)
            x = nn_def(def_input)
            isDup = (x in self.defact)
            if not isDup and x != -1:
                self.defact.add(x)
        return self.defact

    def def_obs_constructor(self, G, timeleft):
        wasdef = self.get_def_wasDefended(G)
        indef = self.get_def_inDefenseSet(G)
        # no need for repeating timeleft, so it is not N
        def_input = self.prev_obs + self.observation + wasdef + indef + [timeleft]
        return def_input

    def get_def_wasDefended(self, G):
        # TODO: if nothing in the defact_tm1, put 0
        wasdef = []
        for node in G.nodes:
            if node in self.defact_tm1:
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

    def uniform_strategy(self, G):
        return random.choices(list(G.nodes),k = self.rand_limit)