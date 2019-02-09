import random
import DagGenerator as dag

class Attacker(object):
    num_resource = 10
    max_num_attack = 10
    min_num_attack = 2
    observation = []
    att_candidate_set = set()
    attact = set()
    timeleft = 10

    def __init__(self,resource,timeleft):
        self.num_resource = resource
        self.timeleft = timeleft

    def att_greedy_action_builder(self,G, nn_att):
        self.attact.clear()
        isDup = 0
        #no pass
        while not isDup and self.check_num_attack(self.attact):
            att_input = self.att_obs_constructor(G, self.observation)
            x = nn_att(att_input)
            isDup = (x in self.attact)
            # if isinstance(x, tuple):
            #     isLegal = (x == -1 or x[0] in att_candidate_set)
            # else:
            #     isLegal = (x == -1 or dag.predecessors(G, x) in att_candidate_set)
            if not isDup:
                self.attact.add(x)

    def check_num_attack(self,attact):
        len_attact = len(attact)
        if len_attact <= self.max_num_attack and len_attact >= self.min_num_attack and len_attact <= self.num_resource:
            return True
        else:
            return False

    def update_obs(self,G):
        self.observation = []
        for node in G.nodes:
            self.observation.append(G.nodes[node]['state'])

    #construct the input of the neural network
    def att_obs_constructor(self, G, obs):
        canAttack, inAttackSet = self.get_att_canAttack_inAttackSet(G)
        att_input = obs + canAttack + inAttackSet + [self.timeleft]
        return att_input

    def get_att_canAttack(self, G):
        canAttack = []
        _, Andnodeset = dag.get_ANDnodes(G)
        for andnode in Andnodeset:
            precondflag = 1
            precond = dag.predecessors(G, andnode)
            for prenode in precond:
                if G.nodes[prenode]['state'] == 0:
                    precondflag = 0
                    break
            if G.nodes[andnode]['state'] == 0 and precondflag:
                canAttack.append(1)
            else:
                canAttack.append(0)

        oredgeset = dag.get_ORedges(G)
        for (father, son) in oredgeset:
            if G.nodes[father]['state'] == 1 and G.nodes[son]['state'] == 0:
                canAttack.append(1)
            else:
                canAttack.append(0)

        return canAttack

    def get_att_canAttack_inAttackSet(self, G):
        canAttack = []
        inAttackSet = []
        _, Andnodeset = dag.get_ANDnodes(G)
        for andnode in Andnodeset:
            if andnode in self.attact:
                inAttackSet.append(1)
            else:
                inAttackSet.append(0)
            precondflag = 1
            precond = dag.predecessors(G, andnode)
            for prenode in precond:
                if G.nodes[prenode]['state'] == 0:
                    precondflag = 0
                    break
            if G.nodes[andnode]['state'] == 0 and precondflag:
                canAttack.append(1)
            else:
                canAttack.append(0)

        oredgeset = dag.get_ORedges(G)
        for (father, son) in oredgeset:
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
        ANDnodes = list(dag.get_ANDnodes(G))
        ORedges = dag.get_ORedges(G)
        attSet = ANDnodes + ORedges
        actset_masked = list(x for x, z in zip(attSet, actmask) if z)
        return random.choices(actset_masked,k=self.num_resource)
