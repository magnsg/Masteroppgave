import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.special as sp


def mcmc(y,numstates):

    #Intialize variables
    T = y.shape[1]#Number of time points
    C = y.shape[0]#Number of cells
    m = numstates#Number of states
    states = np.arange(0,m)#States
    N_iter = 10#Number of iterations


    #Initialize parameters
    pi = np.full((m, m), 1/m)#Transition matrix
    lambdaRate = np.random.rand(C, m)#Rate matrix
    S = np.random.randint(0, m, T)#State sequence
    
    #Initialize hyperparameters
    alpha = 0.1#Gamma hyperparameter
    beta = 0.1#Gamma hyperparameter
    eta = np.array((1/m)*m)#Dirichlet hyperparameter


    #Run MCMC
    for l in range(N_iter):
        print(l)
        #Update Rate matrix
        for i in range(m):
            S_is_i = S == i
            for c in range(C):
                lambdaRate[c,i] = np.random.gamma(alpha + np.sum(y[c,S_is_i]), 1/(beta + np.sum(S_is_i)))


        #Update State sequence
        #Update t = 1
        k = S[1]#S_2
        w = np.zeros(m)
        for j in range(m):
            ln_sum = np.log(pi[j,k])
            for c in range(C):
                ln_sum += y[c,0]*np.log(lambdaRate[c,j])-lambdaRate[c,j]-sp.gammaln(y[c,0]+1)
            w[j] = np.exp(ln_sum)
        if np.sum(w) == 0:
            w_star = np.full(m,1/m)
        else:
            w_star = w/np.sum(w)
        #Sample new state
        S[0] = np.random.choice(states, p=w_star)

        #Update t=2:T-1
        for t in range(1,T-1):
            i = S[t-1]
            k = S[t+1]
            w = np.zeros(m)
            for j in range(m):
                ln_sum = 0
                for c in range(C):
                    ln_sum += y[c,t]*np.log(lambdaRate[c,j])-lambdaRate[c,j]-sp.gammaln(y[c,t]+1)
                w[j] = pi[i,j]*pi[j,k]*np.exp(ln_sum)
            if np.sum(w) == 0:
                w_star = np.full(m,1/m)
            else:
                w_star = w/np.sum(w)

            #Sample new state
            S[t] = np.random.choice(states, p=w_star)
        
        #Update t = T-1
        i = S[T-2]#S_T-1
        w = np.zeros(m)
        for j in range(m):
            ln_sum = 0
            for c in range(C):
                ln_sum += y[c,T-1]*np.log(lambdaRate[c,j])-lambdaRate[c,j]-sp.gammaln(y[c,T-1]+1)
            w[j] = pi[j,k]*np.exp(ln_sum)
        if np.sum(w) == 0:
            w_star = np.full(m,1/m)
        else:
            w_star = w/np.sum(w)
        #Sample new state
        S[T-1] = np.random.choice(states, p=w_star)


    	#Update Transition matrix
        N = np.zeros((m,m))
        for t in range(T-1):
            N[S[t],S[t+1]] += 1

        for i in range(m):
            pi[i] = np.random.dirichlet(eta + N[i])


    return S, pi, lambdaRate