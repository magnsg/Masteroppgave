import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.special as sp

from mcmc_file3 import mcmc

if __name__ == "__main__":
    data = sio.loadmat('MouseData.mat')

    celldata = data['celldata']
    celldata = celldata.astype(int)
    # Maybe remove less active cells
    numstates = 1000
    N_iter = 200

    S, pi, lambdaRate = mcmc(celldata, numstates, N_iter)

    np.savez('result1.npz',S,pi,lambdaRate,numstates)





