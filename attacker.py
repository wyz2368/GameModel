import random
import numpy as np

class Attacker(object):

    def __init__(self, oredges, andnodes, actionspace):
        self.observation = []
        self.canAttack = []
        self.attact = set()
        self.ORedges = oredges
        self.ANDnodes = andnodes
        self.actionspace = actionspace
        self.rand_limit = 4

    def att_greedy_action_builder(self, G, timeleft):
        self.attact.clear()
        isDup = False
        while not isDup:
            att_input = self.att_obs_constructor(G, timeleft)
            x = self.nn_att(att_input[None])[0] #corrensponding to baselines
            action = self.actionspace[x-1]
            if action == 'pass':
                break
            isDup = (action in self.attact)
            if not isDup:
                self.attact.add(action)

    #construct the input of the neural network
    def att_obs_constructor(self, G, timeleft):
        canAttack, inAttackSet = self.get_att_canAttack_inAttackSet(G)
        att_input = self.observation + canAttack + inAttackSet + [timeleft]
        return np.array(att_input)

    # This function can also be used as masking illegal actions.
    def get_att_canAttack(self, G):
        canAttack = []
        for andnode in self.ANDnodes:
            if G.nodes[andnode]['root'] == 1:
                canAttack.append(1)
                continue
            precondflag = 1
            precond = G.predecessors(andnode)
            for prenode in precond:
                if G.nodes[prenode]['state'] == 0:
                    precondflag = 0
                    break
            if G.nodes[andnode]['state'] == 0 and precondflag:
                canAttack.append(1)
            else:
                canAttack.append(0)

        for (father, son) in self.ORedges:
            if G.nodes[father]['state'] == 1 and G.nodes[son]['state'] == 0:
                canAttack.append(1)
            else:
                canAttack.append(0)

        return canAttack

    def get_att_canAttack_inAttackSet(self, G):
        canAttack = []
        inAttackSet = []
        for andnode in self.ANDnodes:
            if G.nodes[andnode]['root'] == 1:
                canAttack.append(1)
                continue
            if andnode in self.attact:
                inAttackSet.append(1)
            else:
                inAttackSet.append(0)
            precondflag = 1
            precond = G.predecessors(andnode)
            for prenode in precond:
                if G.nodes[prenode]['state'] == 0:
                    precondflag = 0
                    break
            if G.nodes[andnode]['state'] == 0 and precondflag:
                canAttack.append(1)
            else:
                canAttack.append(0)

        for (father, son) in self.ORedges:
            if (father, son) in self.attact:
                inAttackSet.append(1)
            else:
                inAttackSet.append(0)
            if G.nodes[father]['state'] == 1 and G.nodes[son]['state'] == 0:
                canAttack.append(1)
            else:
                canAttack.append(0)

        return canAttack, inAttackSet

    def uniform_strategy(self,G):
        actmask = self.get_att_canAttack(G)
        attSet = self.ANDnodes + self.ORedges
        actset_masked = list(x for x, z in zip(attSet, actmask) if z)
        return random.choices(actset_masked,k = self.rand_limit)

    def update_obs(self, obs):
        self.observation = obs

    def reset_att(self):
        self.observation = []
        self.canAttack = []
        self.attact.clear()

    def update_canAttack(self,obs):
        self.canAttack = obs

    def set_current_strategy(self,strategy):
        self.nn_att = strategy