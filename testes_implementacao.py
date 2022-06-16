####
# Testes de implementacao 
# Objetivo - comparar a implementacao em matlab do ALO (https://seyedalimirjalili.com/alo)
# com a implementacao alo_mirjalili (minha) que basicamente traduz o Matlab para o Python 
####

import numpy as np
import alo_mirjalili
import glob as glob
import pandas as pd 

def F1(X):
    return np.sum(X**2)

def F6(X):
    return np.sum(np.abs(X+0.5)**2)
    
def F8(X):
    return np.sum(-X*np.sin(np.sqrt(np.abs(X))))

if __name__ == '__main__':
    
    resultados = []
    for funcao in [F1, F6, F8]:
        comparacoes_t = []
        if (funcao.__name__ == 'F1') or (funcao.__name__ == 'F6'):
            dim = 10
            UB = [100]*dim
            LB = [-100]*dim
        if funcao.__name__ == 'F8':
            dim = 10
            UB = [500]*dim
            LB = [-500]*dim    
        pops = alo_mirjalili.alo(funcao, UB, LB, 40, 500, seed=3320)
        for t in range(500):
            matlab_t = pd.read_csv(f'pops-mirjalili-matlab/{funcao.__name__}/{t+1}.txt', header=None)
            meu_t = pd.DataFrame(pops[t])
            comparacao_t = matlab_t.round(4).equals(meu_t.round(4))
            comparacoes_t.append(comparacao_t)
        resultado = np.any(~np.asarray(comparacoes_t))
        resultados.append(resultado)
        
    for i, funcao in enumerate([F1, F6, F8]):
        print(f'Ha diferencas para {funcao.__name__}?', resultados[i])
