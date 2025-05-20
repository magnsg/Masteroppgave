import cupy as cp
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar


def vectorized_sample(states, w_star):
    """
    Vectorized sampling of states using cumulative probabilities.
    
    Args:
        states (cp.ndarray): Array of possible states.
        w_star (cp.ndarray): Probability matrix of shape (m, T).
        
    Returns:
        new_S (cp.ndarray): Sampled state sequence of shape (T,).
    """
    m, T = w_star.shape
    
    # Generate random numbers for all time points
    rand_samples = cp.random.rand(T)
    
    # Compute cumulative probabilities
    cum_probs = cp.cumsum(w_star, axis=0)
    
    # Find the first state where the cumulative probability exceeds the random sample
    new_S = cp.argmax(cum_probs > rand_samples, axis=0, dtype=cp.int8)
    
    return new_S



def count_transitions(S, m):
    """
    Vectorized counting of state transitions.
    
    Args:
        S (cp.ndarray): State sequence of shape (T,).
        m (int): Number of states.
        
    Returns:
        N (cp.ndarray): Transition count matrix of shape (m, m).
    """
    T = S.shape[0]
    
    # Create indices for the transition matrix
    from_states = S[:-1]  # States at time t
    to_states = S[1:]     # States at time t+1
    
    # Use bincount to count transitions
    flat_indices = from_states * m + to_states
    N_flat = cp.bincount(flat_indices, minlength=m * m)
    
    # Reshape to (m, m)
    N = N_flat.reshape(m, m)
    
    return N



def update_transition_matrix(eta, N):
    """
    Batched Dirichlet sampling of the transition matrix pi on the GPU.
    
    Args:
        eta (cp.ndarray): Dirichlet hyperparameters of shape (m,).
        N (cp.ndarray): Transition count matrix of shape (m, m).
        
    Returns:
        pi (cp.ndarray): Sampled transition matrix of shape (m, m).
    """
    m = N.shape[0]
    
    # Compute concentration parameters for all rows
    concentration_params = eta + N  # Shape: (m, m)
    
    # Sample from gamma distributions in parallel
    gamma_samples = cp.random.gamma(concentration_params, size=concentration_params.shape)
    
    # Normalize gamma samples to get Dirichlet samples
    pi = gamma_samples / cp.sum(gamma_samples, axis=1, keepdims=True)
    
    return pi



def update_lambda_rate(y, S, alpha, beta):
    """
    Vectorized update for lambdaRate using gamma distribution.
    """
    m = S.max().item() + 1  # Get the number of states as a scalar integer
    S_one_hot = cp.eye(m, dtype=cp.float32)[S]  # One-hot encode states (T x m)
    S_counts = cp.sum(S_one_hot, axis=0)  # Sum over time (m,)

    # Sum y over time for each state and cell
    y_sum = cp.sum(y[:, :, cp.newaxis] * S_one_hot, axis=1)  # (C x m)

    # Update lambdaRate using gamma distribution
    lambdaRate = cp.random.gamma(alpha + y_sum, 1 / (beta + S_counts))
    return lambdaRate



def update_state_sequence(y, pi, lambdaRate, states, S):
    """
    Vectorized update for state sequence S.
    """
    C, T = y.shape
    m = len(states)

    # Expand lambdaRate and log_lambdaRate to shape (C x m x T)
    lambdaRate_expanded = cp.tile(lambdaRate[:, :, cp.newaxis], (1, 1, T))  # (C x m x T)
    log_lambdaRate_expanded = cp.log(lambdaRate_expanded)  # (C x m x T)

    # Compute log-sum term for all states and time points
    y_expanded = y[:, cp.newaxis, :]  # (C x 1 x T)
    ln_sum = cp.sum(y_expanded * log_lambdaRate_expanded - lambdaRate_expanded, axis=0)  # (m x T)

    # Compute w for all states and time points
    w = cp.exp(ln_sum)  # (m x T)

    # Apply transition probabilities from previous and next states
    pi_prev = pi[S[:-1], :]  # Transition probabilities from previous states (T-1 x m)
    pi_next = pi[:, S[1:]]   # Transition probabilities to next states (m x T-1)

    # Multiply w by transition probabilities
    w[:, 1:] *= pi_prev.T  # Apply previous state transitions (skip t=0)
    w[:, :-1] *= pi_next   # Apply next state transitions (skip t=T-1)

    # Normalize w to get w_star
    w_star = w / cp.sum(w, axis=0, keepdims=True)  # (m x T)

    # Vectorized sampling of new states
    new_S = vectorized_sample(states, w_star)
    return new_S



def check_convergence(S, S_prev, pi, lambdaRate, numstates, convergence_threshold = 1e-3):
    """
    Check convergence of the MCMC algorithm.
    """
    print(cp.mean(S != S_prev).item())
    return cp.allclose(S, S_prev, atol=convergence_threshold)





def mcmc(y, numstates, N_iter=10):
    """
    MCMC algorithm for state sequence and rate matrix estimation.
    """
    y = cp.array(y, dtype=cp.float32)  # Convert y to CuPy array with float32 precision
    T = y.shape[1]   # Number of time points
    C = y.shape[0]   # Number of cells
    m = numstates    # Number of states
    states = cp.arange(m)  # States

    # Initialize parameters
    pi = cp.full((m, m), 1 / m, dtype=cp.float32)  # Transition matrix
    lambdaRate = cp.random.rand(C, m).astype(cp.float32)  # Rate matrix
    S = cp.random.randint(0, m, T, dtype=cp.int32)  # State sequence
    S_prev = cp.copy(S)  # Previous state sequence

    # Hyperparameters
    alpha = 0.1  # Gamma shape parameter
    beta = 0.1   # Gamma rate parameter
    eta = cp.full(m, 1 / m, dtype=cp.float32)  # Dirichlet hyperparameter

    for l in tqdm(range(N_iter), desc="MCMC Progress"):
        # Update lambdaRate
        lambdaRate = update_lambda_rate(y, S, alpha, beta)

        # Update state sequence
        S = update_state_sequence(y, pi, lambdaRate, states, S)

        # Update transition matrix pi
        N = count_transitions(S, m)
        pi = update_transition_matrix(eta, N)

        """
        # Check convergence
        if l % 200 == 0:
            if check_convergence(S, S_prev, pi, lambdaRate, numstates):
                print(f"Converged after {l} iterations.")
                break
        
        # Update previous state sequence
        S_prev = cp.copy(S)
        """

    # Convert results back to NumPy
    S = cp.asnumpy(S)
    pi = cp.asnumpy(pi)
    lambdaRate = cp.asnumpy(lambdaRate)
    return S, pi, lambdaRate