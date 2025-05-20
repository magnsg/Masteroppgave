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
    # Using int32 instead of int8 to prevent overflow with more states
    new_S = cp.argmax(cum_probs > rand_samples, axis=0).astype(cp.int32)
    
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
    
    # Clip states to valid range to prevent negative indices in bincount
    # This is safer than throwing errors and has minimal performance impact
    S = cp.clip(S, 0, m-1)
    
    # Create indices for the transition matrix
    from_states = S[:-1]  # States at time t
    to_states = S[1:]     # States at time t+1
    
    # Use bincount to count transitions by flattening the transition matrix
    flat_indices = from_states * m + to_states
    N_flat = cp.bincount(flat_indices, minlength=m * m)
    
    # Reshape the flattened counts back to a matrix
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
    
    # Sample from gamma distributions in parallel for each transition
    gamma_samples = cp.random.gamma(concentration_params)
    
    # Normalize gamma samples row-wise to get Dirichlet samples
    pi = gamma_samples / cp.sum(gamma_samples, axis=1, keepdims=True)
    
    return pi


def update_lambda_rate(y, S, alpha, beta):
    """
    Vectorized update for lambdaRate using gamma distribution.
    
    Args:
        y (cp.ndarray): Observation matrix of shape (C, T)
        S (cp.ndarray): Current state sequence of shape (T,)
        alpha (float): Gamma prior shape parameter
        beta (float): Gamma prior rate parameter
        
    Returns:
        lambdaRate (cp.ndarray): Updated rate matrix of shape (C, m)
    """
    # Determine number of states from current sequence
    m = int(S.max()) + 1  # Convert to Python int immediately for indexing
    
    # One-hot encode the state sequence (T x m matrix)
    S_one_hot = cp.eye(m, dtype=cp.float32)[S]
    
    # Count how many times each state appears (sum over time)
    S_counts = cp.sum(S_one_hot, axis=0)  # Shape: (m,)
    
    # Sum observations for each state (C x m matrix)
    y_sum = cp.sum(y[:, :, cp.newaxis] * S_one_hot, axis=1)
    
    # Sample new rates from gamma distribution
    lambdaRate = cp.random.gamma(alpha + y_sum, 1 / (beta + S_counts))
    
    return lambdaRate


def update_state_sequence(y, pi, lambdaRate, states, S):
    """
    Vectorized update for state sequence S with numerical stability improvements.
    
    Args:
        y (cp.ndarray): Observation matrix (C x T)
        pi (cp.ndarray): Transition matrix (m x m)
        lambdaRate (cp.ndarray): Rate matrix (C x m)
        states (cp.ndarray): Array of possible states
        S (cp.ndarray): Current state sequence (T,)
        
    Returns:
        new_S (cp.ndarray): Updated state sequence (T,)
    """
    C, T = y.shape
    m = len(states)
    
    # Compute log-lambda for numerical stability
    log_lambda = cp.log(lambdaRate[:, :, cp.newaxis])  # Shape: (C x m x 1)
    
    # Expand observations for broadcasting (C x 1 x T)
    y_expanded = y[:, cp.newaxis, :]
    
    # Compute log joint probability using log-sum-exp trick
    ln_sum = cp.sum(y_expanded * log_lambda - lambdaRate[:, :, cp.newaxis], axis=0)
    
    # Stabilize exponentials by subtracting maximum
    max_ln = cp.max(ln_sum, axis=0, keepdims=True)
    w = cp.exp(ln_sum - max_ln)  # Shape: (m x T)
    
    # Apply transition probabilities from previous states
    w[:, 1:] *= pi[S[:-1]].T  # Transpose for correct broadcasting
    
    # Apply transition probabilities to next states
    w[:, :-1] *= pi[:, S[1:]]
    
    # Normalize probabilities safely
    w_star = w / cp.sum(w, axis=0, keepdims=True)
    
    # Handle any numerical instabilities (NaN values become uniform probabilities)
    w_star = cp.nan_to_num(w_star, nan=1/m)
    
    # Sample new states
    return vectorized_sample(states, w_star)


def mcmc(y, numstates, N_iter=10):
    """
    MCMC algorithm for state sequence and rate matrix estimation.
    
    Args:
        y (np.ndarray): Input data matrix (cells x time)
        numstates (int): Number of hidden states
        N_iter (int): Number of MCMC iterations
        
    Returns:
        S (np.ndarray): Final state sequence
        pi (np.ndarray): Final transition matrix
        lambdaRate (np.ndarray): Final rate matrix
    """
    # Convert data to GPU array with float32 precision for better performance
    y = cp.array(y, dtype=cp.float32)
    T = y.shape[1]  # Number of time points
    C = y.shape[0]  # Number of cells
    m = numstates   # Number of states
    
    # Initialize transition matrix with uniform probabilities
    pi = cp.full((m, m), 1 / m, dtype=cp.float32)
    
    # Initialize rate matrix with random values
    lambdaRate = cp.random.rand(C, m).astype(cp.float32)
    
    # Initialize state sequence with random states (using int32)
    S = cp.random.randint(0, m, T, dtype=cp.int32)
    
    # Set hyperparameters
    alpha = 0.1  # Gamma shape parameter
    beta = 0.1   # Gamma rate parameter
    eta = cp.full(m, 1 / m, dtype=cp.float32)  # Dirichlet hyperparameter

    # Main MCMC loop
    for l in tqdm(range(N_iter), desc="MCMC Progress"):
        # 1. Update rate matrix lambdaRate
        lambdaRate = update_lambda_rate(y, S, alpha, beta)
        
        # 2. Update state sequence S
        S = update_state_sequence(y, pi, lambdaRate, cp.arange(m), S)
        
        # 3. Update transition matrix pi
        N = count_transitions(S, m)
        pi = update_transition_matrix(eta, N)

    # Convert results back to NumPy arrays before returning
    return cp.asnumpy(S), cp.asnumpy(pi), cp.asnumpy(lambdaRate)