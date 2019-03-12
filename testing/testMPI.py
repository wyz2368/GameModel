from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.size
name = MPI.Get_processor_name()
print('This is my rank:',comm.rank)
print(size)
print(name)