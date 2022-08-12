from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
IAMROOT = rank == 0
#print(dir(rank))
sendbuf = None
if IAMROOT:
    sendbuf = np.empty([4*size, 100], dtype='i')
    sendbuf.T[:,:] = range(4*size)
    print("Rank:",rank)
    print(f"{sendbuf=}")
recvbuf = np.empty([4,100], dtype='i')
if IAMROOT:
    comm.Scatter([sendbuf,MPI.INT], recvbuf, root=0)
# print(f"{sendbuf=}")
print(f"{rank=}",f"{recvbuf=}",sep="\n")