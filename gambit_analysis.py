import numpy as np
import subproc
import file_op as fp

# Input arguments: payoff matrix for the defender, poDef; payoff matrix for the attacker, poAtt.
# In a payoff matrix example: 1 2
# 							  3 5
# 							  6 7
# There are 3 defender strategies (3 rows) and 2 attacker strategies (2 columns).
# NFG File format: Payoff version
#TODO: delete attackgraph when running on flux
gambit_DIR = './attackgraph/gambit_data/payoffmatrix.nfg'

def encode_gambit_file(poDef, poAtt):
	try:
		if poDef.shape != poAtt.shape:
			raise Exception("Inputted payoff matrix for defender and attacker must be of same shape.")
	except Exception as error:
		print(repr(error))
		return -1
	# Write header
	with open(gambit_DIR, "w") as nfgFile:
		nfgFile.write('NFG 1 R "Attackgroup"\n{ "Defender" "Attacker" } ')
		# Write strategies
		nfgFile.write('{ ' + str(poDef.shape[0]) + ' ' + str(poDef.shape[1]) +' }\n\n')
		# Write outcomes
		for i in range(poDef.shape[1]):
			for j in range(poDef.shape[0]):
				nfgFile.write(str(poDef[j][i]) + " ")
				nfgFile.write(str(poAtt[j][i]) + " ")

	# Gambit passing and NE calculation to come later.

def gambit_analysis():
	if not fp.isExist(gambit_DIR):
		raise ValueError(".nfg file does not exist!")
	command_str = "gambit-lcp -q ./attackgraph/gambit_data/payoffmatrix.nfg > ./attackgraph/gambit_data/nash.txt"
	subproc.call_and_wait_with_timeout(command_str)
	print('gambit_analysis done!')

def decode_gambit_file():
	nash_DIR = './attackgraph/gambit_data/nash.txt'
	if not fp.isExist(nash_DIR):
		raise ValueError("nash.txt file does not exist!")
	with open(nash_DIR,'r') as f:
		nash = f.readline()

	nash = nash[3:]
	nash = nash.split(',')
	# print(nash)
	# nash= np.fromstring(nash, dtype=np.float, sep=',')
	# print(nash)
	# nash = np.round(nash,decimals=2)
	# print(nash)
	new_nash = []
	for i in range(len(nash)):
		new_nash.append(convert(nash[i]))

	new_nash = np.array(new_nash)
	new_nash = np.round(new_nash,decimals=2)
	nash_def = new_nash[:int(len(new_nash)/2)]
	nash_att = new_nash[int(len(new_nash)/2):]

	return nash_att, nash_def

def do_gambit_analysis(poDef, poAtt):
	encode_gambit_file(poDef, poAtt)
	gambit_analysis()
	nash_att, nash_def = decode_gambit_file()
	return nash_att, nash_def


def convert(s):
	try:
		return float(s)
	except ValueError:
		num, denom = s.split('/')
		return float(num) / float(denom)