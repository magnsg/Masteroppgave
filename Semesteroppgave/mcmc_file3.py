import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Import tqdm for progress bar

def update_lambda_rate(args):
    y, S, alpha, beta, c, i = args
    S_is_i = S == i
    return np.random.gamma(alpha + np.sum(y[c, S_is_i]), 1 / (beta + np.sum(S_is_i)))

def update_state_sequence(args):
    y, pi, lambdaRate, states, t, S, C, log_lambdaRate = args
    m = len(states)  # Number of states
    T = y.shape[1]  # Number of time points

    i = S[t - 1] if t != 0 else None  # Previous state (None for t = 0)
    k = S[t + 1] if t != T-1 else None  # Next state (None for t = T-1)

    # Compute the log-sum term for all j using matrix operations
    ln_sum = np.sum(y[:, t][:, np.newaxis] * log_lambdaRate - lambdaRate, axis=0)

    # Compute w for all j using vectorized operations
    w = np.exp(ln_sum)
    if i is not None:
        w *= pi[i, :]  # Apply transition probability from the previous state
    if k is not None:
        w *= pi[:, k]  # Apply transition probability to the next state

    # Normalize w to get w_star
    w_star = w / np.sum(w) if np.sum(w) != 0 else np.full(m, 1 / m)

    # Sample new state
    return np.random.choice(states, p=w_star)

def mcmc(y, numstates, N_iter = 10):
    T = y.shape[1]  # Number of time points
    C = y.shape[0]  # Number of cells
    m = numstates  # Number of states
    states = np.arange(0, m)  # States
    N_iter = N_iter  # Number of iterations

    pi = np.full((m, m), 1 / m)  # Transition matrix
    lambdaRate = np.random.rand(C, m)  # Rate matrix
    S = np.random.randint(0, m, T)  # State sequence

    alpha = 0.1  # Gamma hyperparameter
    beta = 0.1  # Gamma hyperparameter
    eta = np.full(m, 1 / m)  # Dirichlet hyperparameter

    pool = Pool(cpu_count())  # Create a pool of workers

    for l in tqdm(range(N_iter), desc="MCMC Progress"):  # Add tqdm progress bar

        # Update Rate matrix in parallel
        args = [(y, S, alpha, beta, c, i) for c in range(C) for i in range(m)]
        results = pool.map(update_lambda_rate, args)
        lambdaRate = np.array(results).reshape(C, m)

        # Update State sequence in parallel
        log_lambdaRate = np.log(lambdaRate) # Precompute log(lambdaRate) for efficiency
        args = [(y, pi, lambdaRate, states, t, S, C, log_lambdaRate) for t in range(T)]
        S = np.array(pool.map(update_state_sequence, args))

        # Update Transition matrix
        N = np.zeros((m, m))
        for t in range(T - 1):
            N[S[t], S[t + 1]] += 1
        for i in range(m):
            pi[i] = np.random.dirichlet(eta + N[i])

    pool.close()
    pool.join()

    return S, pi, lambdaRate