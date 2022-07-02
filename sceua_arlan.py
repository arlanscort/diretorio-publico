'''
Referencia usada na implementacao - Naeini (2019)
'''

#%% 
from ast import Break
import numpy as np
def eggholder(X):
    x1 = X[0]
    x2 = X[1]
    f = -(x2+47)*np.sin(np.sqrt(np.abs(x1/2+(x2+47)))) - x1*np.sin(np.sqrt(np.abs(x1-(x2+47))))
    eggholder.counter += 1
    return f


#%%
def cce(C, m, q, P, alpha, beta):
    
    # L = np.random.choice(m, q, replace=False, p=P)
    
    # Loop de selecao dos sub-complexos
    for b in range(beta):
        L = [1]
        for i in range(q-1):
            for iter in range(1000):
                lpos = 1 + np.floor(m + 0.5 - np.sqrt((m+0.5)**2 - m*(m+1)*np.random.random()))
                if ~np.isin(lpos, L):
                    break
            L.append(lpos)
        L = np.sort(L.copy())
        # L = L - 1
        print(L)
        
        # Loop de evolucao por Nelder e Mead
        


#%%
def sceua(fobj, UB, LB, k, t_max, m=None, q=None, alpha=None, beta=None, seed=None):

    # Ajustes
    fobj.counter = 0
    if seed is not None:
        np.random.seed(seed)
    n = len(UB)
    UB = np.asarray(UB).reshape(1, -1)
    LB = np.asarray(LB).reshape(1, -1)
    if m is None: 
        m = 2*n + 1
    if q is None: 
        q = n + 1
    if alpha is None: 
        alpha = 1
    if beta is None:
        beta = 2*n + 1
    P = [2*(m+1-i)/(m*(m+1)) for i in range(1, m+1)]
    
    # Inicializacao
    t = 0
    s = k*m
    D = np.empty((s, n))
    for i in range(s):
        D[i,:] = (UB-LB)*np.random.rand(1,n) + LB
    F = np.apply_along_axis(fobj, 1, D)
    D = D[np.argsort(F)]
    F = np.sort(F)
    
    # Loop principal
    while (t < t_max):
        t+=1
        
        # Loop nos complexos
        for i in range(1, k+1):
            print(i)
            I = [i+k*(j-1)-1 for j in range(1, m+1)]
            D[I,:] = cce(D[I,:], m, q, P, alpha, beta)
            
            
        break
    return D, F, fobj.counter

#%%
D, F, count = sceua(eggholder, [512, 512], [-512, -512], 4, 10, seed=666)
# %%
