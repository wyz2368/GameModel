import numpy as np

class Game_data(object):
    def __init__(self,id,env):
        self.env = env
        self.graph_id = id
        self.att_str = []
        self.def_str = []
        self.nasheq = {}
        self.payoffmatrix_def = np.zeros((1,1))
        self.payoffmatrix_att = np.zeros((1,1))

    def add_att_str(self, str_name):
        # TODO: check if str_name is an attacker's strategy and if it exists.
        if not isinstance(str_name,str):
            raise ValueError("The name to be added is not a str." )
        self.att_str.append(str_name)
        print(str_name + " has been added to attacker's strategy set")

    def add_def_str(self, str_name):
        # TODO: check if str_name is an attacker's strategy and if it exists.
        if not isinstance(str_name,str):
            raise ValueError("The name to be added is not a str." )
        self.def_str.append(str_name)
        print(str_name + " has been added to attacker's strategy set")

    def get_graph_id(self):
        return self.graph_id

    def init_payoffmatrix(self, payoff_def, payoff_att):
        self.payoffmatrix_def[0,0] = payoff_def
        self.payoffmatrix_att[0,0] = payoff_att
        print("Payoff matrix has been initilized by" + " " + str(payoff_def) + " for the defender.")
        print("Payoff matrix has been initilized by" + " " + str(payoff_att) + " for the attacker.")

    def add_nasheq(self, att_str, def_str, ne):
        if att_str not in self.att_str:
            raise ValueError("Attacker's strategy does not exist.")
        if def_str not in self.def_str:
            raise ValueError("Defender's strategy does not exist.")
        self.nasheq[(def_str,att_str)] = ne

    ''' 
    >>> import numpy as np
    >>> p = np.array([[1,2],[3,4]])

    >>> p = np.append(p, [[5,6]], 0)
    >>> p = np.append(p, [[7],[8],[9]],1)

    >>> p
        array([[1, 2, 7],
                [3, 4, 8],
                [5, 6, 9]])
                
    To extend a matrix, extend col fist then extend row.
    '''

    def add_col_att(self, col):
        num_row, _ = np.shape(self.payoffmatrix_att)
        num_row_new, _ = np.shape(col)
        if num_row != num_row_new:
            raise ValueError("Cannot extend attacker column since dim does not match")
        self.payoffmatrix_att = np.append(self.payoffmatrix_att, col, 1)

    def add_row_att(self, row):
        _, num_col = np.shape(self.payoffmatrix_att)
        _, num_col_new = np.shape(row)
        if num_col == num_col_new:
            raise ValueError("Cannot extend attacker row since dim does not match")
        self.payoffmatrix_att = np.append(self.payoffmatrix_att,row, 0)

    def add_col_def(self, col):
        num_row, _ = np.shape(self.payoffmatrix_def)
        num_row_new, _ = np.shape(col)
        if num_row != num_row_new:
            raise ValueError("Cannot extend defender column since dim does not match")
        self.payoffmatrix_def = np.append(self.payoffmatrix_def, col, 1)

    def add_row_def(self, row):
        _, num_col = np.shape(self.payoffmatrix_def)
        _, num_col_new = np.shape(row)
        if num_col == num_col_new:
            raise ValueError("Cannot extend defender row since dim does not match")
        self.payoffmatrix_def = np.append(self.payoffmatrix_def, row, 0)
