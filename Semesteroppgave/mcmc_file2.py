import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.special as sp
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Import tqdm for progress bar


def update_lambda_rate(args):
    y, S, alpha, beta, c, i = args
    S_is_i = S == i
    return np.random.gamma(alpha + np.sum(y[c, S_is_i]), 1 / (beta + np.sum(S_is_i)))

def update_state_sequence(args):
    y, pi, lambdaRate, states, t, S, C = args
    if t == 0:
        k = S[t+1]
        w = np.zeros(len(states))
        for j in range(len(states)):
            ln_sum = 0
            for c in range(C):
                ln_sum += y[c, 0] * np.log(lambdaRate[c, j]) - lambdaRate[c, j]
            w[j] = pi[j, k]*np.exp(ln_sum)
        w_star = w / np.sum(w) if np.sum(w) != 0 else np.full(len(states), 1 / len(states))
        return np.random.choice(states, p=w_star)
    elif t == y.shape[1] - 1:
        i = S[t - 1]
        w = np.zeros(len(states))
        for j in range(len(states)):
            ln_sum = 0
            for c in range(C):
                ln_sum += y[c, t] * np.log(lambdaRate[c, j]) - lambdaRate[c, j]
            w[j] = pi[i, j] * np.exp(ln_sum)
        w_star = w / np.sum(w) if np.sum(w) != 0 else np.full(len(states), 1 / len(states))
        return np.random.choice(states, p=w_star)
    else:
        i = S[t - 1]
        k = S[t + 1]
        w = np.zeros(len(states))
        for j in range(len(states)):
            ln_sum = 0
            for c in range(C):
                ln_sum += y[c, t] * np.log(lambdaRate[c, j]) - lambdaRate[c, j]
            w[j] = pi[i, j] * pi[j, k] * np.exp(ln_sum)
        w_star = w / np.sum(w) if np.sum(w) != 0 else np.full(len(states), 1 / len(states))
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
        args = [(y, pi, lambdaRate, states, t, S, C) for t in range(T)]
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