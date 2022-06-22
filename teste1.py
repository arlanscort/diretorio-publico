####
# Teste 1
# Verificar se a implementacao alo.py eh identica a implementacao em Matlab
# disponivel em https://seyedalimirjalili.com/alo
####

import numpy as np
import alo
import pandas as pd
import plotly.graph_objects as go

def F1(X):
    return np.sum(X**2)

def F5(X):
    d = len(X)
    soma = 0
    for i in range(0, d-1):
        soma += 100*(X[i+1] - X[i]**2)**2 + (X[i]-1)**2
    return soma 

def F10(X):
    d = len(X)
    p1 = -20*np.exp(-0.2*np.sqrt(1/d*np.sum(X**2)))
    p2 = -np.exp(1/d*np.sum(np.cos(2*np.pi*X)))
    return p1 + p2 + 20 + np.exp(1)


if __name__ == '__main__':
    
    ha_diferencas = {} # Ha diferencas?

    # F1 - confs
    fobj = F1
    d = 10
    UB = [+100]*d
    LB = [-100]*d
    n = 15
    seed = 2092
    t_max = 100
    # F1 - roda
    _, _, X_python, F_python = alo.alo(fobj, UB, LB, n, t_max, seed)
    comparacoes = []
    F_matlab = []
    for i in range(1, t_max+1):
        df_matlab = pd.read_csv(f'alo-matlab-resultados/Seed_{seed}_N_{n}_Maxiter_{t_max}_{fobj.__name__}/X_{i}.txt', header=None).round(4)
        df_python = pd.DataFrame(X_python[i-1]).round(4)
        comparacoes.append(df_matlab.equals(df_python))
        F_matlab.append(pd.read_csv(f'alo-matlab-resultados/Seed_{seed}_N_{n}_Maxiter_{t_max}_{fobj.__name__}/f_{i}.txt', header=None).values[0])
    ha_diferencas[fobj.__name__] = np.any(~np.asarray(comparacoes))
    idx = np.arange(1, t_max+1)
    plot_python = [f[0] for f in F_python]
    plot_matlab = [f[0] for f in F_matlab]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=plot_python, name='Python'))
    fig.add_trace(go.Scatter(x=idx, y=plot_matlab, name='Matlab'))
    fig.update_layout(title=fobj.__name__)
    fig.write_html(f'teste1_{fobj.__name__}.html')
    
    # F5 - confs
    fobj = F5
    d = 10
    UB = [+30]*d
    LB = [-30]*d
    n = 50
    seed = 666
    t_max = 500
    # F5 - roda
    _, _, X_python, F_python = alo.alo(fobj, UB, LB, n, t_max, seed)
    comparacoes = []
    F_matlab = []
    for i in range(1, t_max+1):
        df_matlab = pd.read_csv(f'alo-matlab-resultados/Seed_{seed}_N_{n}_Maxiter_{t_max}_{fobj.__name__}/X_{i}.txt', header=None).round(4)
        df_python = pd.DataFrame(X_python[i-1]).round(4)
        comparacoes.append(df_matlab.equals(df_python))
        F_matlab.append(pd.read_csv(f'alo-matlab-resultados/Seed_{seed}_N_{n}_Maxiter_{t_max}_{fobj.__name__}/f_{i}.txt', header=None).values[0])
    ha_diferencas[fobj.__name__] = np.any(~np.asarray(comparacoes))
    idx = np.arange(1, t_max+1)
    plot_python = [f[0] for f in F_python]
    plot_matlab = [f[0] for f in F_matlab]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=plot_python, name='Python'))
    fig.add_trace(go.Scatter(x=idx, y=plot_matlab, name='Matlab'))
    fig.update_layout(title=fobj.__name__)
    fig.write_html(f'teste1_{fobj.__name__}.html')

    # F10 - confs
    fobj = F10
    d = 10
    UB = [+32]*d
    LB = [-32]*d
    n = 40
    seed = 10
    t_max = 200
    # F10 - roda
    _, _, X_python, F_python = alo.alo(fobj, UB, LB, n, t_max, seed)
    comparacoes = []
    F_matlab = []
    for i in range(1, t_max+1):
        df_matlab = pd.read_csv(f'alo-matlab-resultados/Seed_{seed}_N_{n}_Maxiter_{t_max}_{fobj.__name__}/X_{i}.txt', header=None).round(4)
        df_python = pd.DataFrame(X_python[i-1]).round(4)
        comparacoes.append(df_matlab.equals(df_python))
        F_matlab.append(pd.read_csv(f'alo-matlab-resultados/Seed_{seed}_N_{n}_Maxiter_{t_max}_{fobj.__name__}/f_{i}.txt', header=None).values[0])
    ha_diferencas[fobj.__name__] = np.any(~np.asarray(comparacoes))
    idx = np.arange(1, t_max+1)
    plot_python = [f[0] for f in F_python]
    plot_matlab = [f[0] for f in F_matlab]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=plot_python, name='Python'))
    fig.add_trace(go.Scatter(x=idx, y=plot_matlab, name='Matlab'))
    fig.update_layout(title=fobj.__name__)
    fig.write_html(f'teste1_{fobj.__name__}.html')