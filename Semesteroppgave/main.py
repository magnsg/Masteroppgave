import numpy as np
import scipy.io as sio
import scipy.special as sp

from mcmc_file5 import mcmc
#from mcmc_file4 import mcmc
#from mcmc_file3 import mcmc

if __name__ == "__main__":
    data = sio.loadmat('MouseData.mat')

    celldata = data['celldata']
    celldata = np.array(celldata.astype(int))

    # Remove less active cells (cells with fewer than 5 total firings)
    total_firings = np.sum(celldata, axis=1)  # Sum firings for each cell
    active_cells = total_firings >= 5  # Boolean mask for cells with >= 5 firings
    celldata_filtered = celldata[active_cells, :]  # Filter the data
    print(f"Original number of cells: {celldata.shape[0]}")
    print(f"Number of cells after filtering: {celldata_filtered.shape[0]}")
    
    numstates = 10
    N_iter = 5000

    S, pi, lambdaRate = mcmc(celldata, numstates, N_iter)

    np.savez('result1.npz',S,pi,lambdaRate,numstates)

