# Funcoes de teste para algoritmos de otimizacao
# Fonte: https://www.sfu.ca/~ssurjano/optimization.html

import numpy as np

# Funcoes 2D
def bukin(X):
    # UB = [-5, 3]
    # LB = [-15, -3]
    # Otimo = [-10, 1]
    # Valor = 0
    x1 = X[0]
    x2 = X[1]
    p1 = 100*np.sqrt(np.abs(x2-0.01*x1**2))
    p2 = 0.01*np.abs(x1+10)
    f = p1 + p2
    return f

def drop_wave(X):
    # UB = [5.12, 5.12]
    # LB = [-5.12, -5.12]
    # Otimo = [0, 0]
    # Valor = -1
    x1 = X[0]
    x2 = X[1]
    p1 = 1 + np.cos(12*np.sqrt(x1**2+x2**2))
    p2 = 0.5*(x1**2+x2**2) + 2
    f = -p1/p2
    return f

def eggholder(X):
    # UB = [512, 512]
    # LB = [-512, -512]
    # Otimo = [512, 404.2319]
    # Valor = -959.6407
    x1 = X[0]
    x2 = X[1]
    p1 = -(x2+47)*np.sin(np.sqrt(np.abs(x1/2+(x2+47))))
    p2 = - x1*np.sin(np.sqrt(np.abs(x1-(x2+47))))
    f = p1 + p2
    eggholder.counter += 1
    return f

def shubert(X):
    # UB = [5.12, 5.12]
    # LB = [-5.12, -5.12]
    # Otimo = multiplos
    # Valor = -186.7309
    x1 = X[0]
    x2 = X[1]
    p1 = 0
    p2 = 0
    for i in range(1, 6):
        p1 += i*np.cos((i+1)*x1+i)
        p2 += i*np.cos((i+1)*x2+i)
    return p1*p2
def six_hump_camel(X):
    # UB = [3, 2]
    # LB = [-3, -2]
    # Otimo = [0.0898, -0.7126] e [-0.0898, 0.7126]
    # Valor = -1.0316
    x1 = X[0]
    x2 = X[1]
    p1 = (4-2.1*x1**2+x1**4/3)*x1**2
    p2 = +x1*x2+(-4+4*x2**2)*x2**2
    f = p1 + p2
    return f
# Funcoes multidimensionais

def ackley(X):
    # UB = [32.768, ..., 32.768]
    # LB = [-32.768, ..., -32.768]
    # Otimo = [0, ..., 0]
    # Valor = 0
    a = 20
    b = 0.2
    c = 2*np.pi
    d = len(X)
    p1 = -a*np.exp(-b*np.sqrt(1/d*np.sum(X**2)))
    p2 = -np.exp(1/d*np.sum(np.cos(c*X))) + a + np.exp(1)
    f = p1+ p2
    return f

def griewank(X):
    # UB = [600, ..., 600]
    # LB = [-600, ..., -600]
    # Otimo = [0, ..., 0]
    # Valor = 0
    soma = 0
    prod = 1
    for i in range(len(X)):
        soma += X[i]**2/4000
        prod *= np.cos(X[i]/np.sqrt(i+1))
    f = soma - prod + 1
    return f

def schwefel(X):
    # UB = [500, ..., 500]
    # LB = [-500, ..., -500]
    # Otimo = [420.9687, ..., 420.9687]
    # Valor = 0
    d = len(X)
    soma = 0
    for i in range(d):
        soma += X[i]*np.sin(np.sqrt(np.abs(X[i])))
    f = 418.9829*d - soma
    return f

# if __name__ == '__main__':
    
#     import alo
#     import matplotlib.pyplot as plt 
    
#     fobj = eggholder
#     _, _, X_geracoes, F_geracoes = alo.alo(fobj, [512,512], [-512,-512], 10, 500)
    
    
#     # Plot
#     for iteracao in range(1,11):
#         X1, X2 = np.meshgrid(np.linspace(-512,512,100), np.linspace(-512,512,100))
#         Z = np.empty((X1.shape[0], X1.shape[1]))
#         for i in range(X1.shape[0]):
#             for j in range(X1.shape[1]):
#                 Z[i,j] = eggholder([X1[i,j], X2[i,j]])
#         fig,ax=plt.subplots(1,1)
#         cp = ax.contourf(X1, X2, Z)
#         otimo = [512, 404.2319]
#         ax.scatter(otimo[0], otimo[1], color='red')
#         ax.set_title(f'Iteracao {iteracao}')
#         ax.scatter(X_geracoes[iteracao-1][:,0], X_geracoes[iteracao-1][:,1], color='black')
#         fig.colorbar(cp)
#         fig.savefig(f'fig_iteracao{iteracao}.png')

if __name__ == '__main__':
    
    # Teste SCE-UA
    import sceua
    k = 10
    D, count, minimos = sceua.executar(eggholder, [512, 512], [-512, -512], k, 10, seed=12)
    import matplotlib.pyplot as plt 
    plt.plot(minimos, color='blue')
    plt.suptitle(f'Complexos = {k}')
    plt.axhline(-959, color='red')
    plt.show()