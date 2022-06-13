'''
Algoritomo ALO - Antlion Optimizer
https://doi.org/10.1016/j.advengsoft.2015.01.010
Autor do algoritmo - Mirjalili (2015) 
Implementacao em python - Scortegagna (2022)
'''

#%% Imports
import sys
import numpy as np
import matplotlib.pyplot as plt

#%%
import fobjs
fobj = fobjs.eggholder

#%% Funcoes
# def roleta_(pesos):
# '''
# A roleta_ com pesos 1/fitnesses esbarra no problema dos valores negativos...
# '''
#     acumulado = np.cumsum(pesos)
#     p = np.random.uniform()*acumulado[-1]
#     indice_selecionado = np.nan
#     for indice in range(len(pesos)):
#         if (acumulado[indice] > p):
#             indice_selecionado = indice
#             break
#     print(f'p={p} e indice={indice_selecionado}')
#     return indice_selecionado
def passeio_aleatorio(dim, Linf, Lsup, antlion, t, t_max):
    # Reduzir progressivamente o espaco de busca
    I = 1
    if t > 0.1*t_max:
        I = (10**2)*(t/t_max)
    elif t > 0.5*t_max:
        I = (10**3)*(t/t_max)
    elif t > 0.75*t_max:
        I = (10**4)*(t/t_max)
    elif t > 0.90*t_max:
        I = (10**5)*(t/t_max)
    elif t > 0.95*t_max:
        I = (10**6)*(t/t_max)
    Linf_t = Linf/I
    Lsup_t = Lsup/I
    # Mover o espaco de busca em torno da formiga-leao e atribuir um quadrante aleatorio - armadilha
    if np.random.rand() < 0.5:
        Linf_t = antlion + Linf
    else:
        Linf_t = antlion - Linf
    if np.random.rand() >= 0.5: #???
        Lsup_t = antlion + Lsup
    else: 
        Lsup_t = antlion - Lsup
    # Limitar ao espaco de busca pre-definido (Scortegagna, 2022)
    Linf_t = np.clip(Linf_t, Linf, Lsup)
    Lsup_t = np.clip(Lsup_t, Linf, Lsup)

    # Executar o passeio aleatorio
    RW = []
    for i in range(dim):
        R = (np.random.rand(t_max,1) > 0.5).astype(int)
        R_ = 2*R-1
        R__ = np.insert(R_, 0, 0, axis=0)
        X = np.cumsum(R__, axis=0)
        # Normalizar 
        a = X.min()
        b = X.max()
        c = Linf[:,i]
        d = Lsup[:,i]
        Xnorm = ((X-a)*(d-c))/(b-a) + c
        RW.append(Xnorm)
    RW = np.concatenate(RW, axis=1)
    return RW

#%% ALO
def alo(Linf, Lsup, n, t_max):

    # Alguns ajustes 
    d = len(Linf)
    Linf = np.asarray(Linf).reshape(1, -1)
    Lsup = np.asarray(Lsup).reshape(1, -1)
    p_sce = np.asarray([2*(n+1-i)/(n*(n+1)) for i in range(1, n+1)])
    f_medias = []
    f_minimos = []

    # Primeira iteracao - inicializacao
    t = 1
    print(f'Iteracao {t}/{t_max}...')
    MX_ant = (Lsup - Linf)*np.random.rand(n, d) + Linf
    MX_antlion = (Lsup - Linf)*np.random.rand(n, d) + Linf
    #MF_ant = np.apply_along_axis(fobj, 1, MX_ant).reshape(-1,1)
    MF_antlion = np.apply_along_axis(fobj, 1, MX_antlion).reshape(-1,1)
    MX_antlion = MX_antlion[np.argsort(MF_antlion, axis=0).flatten()]
    MF_antlion = np.sort(MF_antlion, axis=0)
    X_elite_antlion = MX_antlion[0,:].copy()
    f_elite_antlion = MF_antlion[0].item()
    historico = [[f_elite_antlion, MF_antlion.mean()]]
    print('Concluida.')
    print('Melhor solucao:', np.round(X_elite_antlion,4), np.round(f_elite_antlion,4))

    # Loop principal
    t += 1
    
    pos1 = [MX_ant[-1,:]]

    while(t <= t_max):
        print(f'\nIteracao {t}/{t_max}...')
        # Aplicar os passeios aleatorios

        for i in range(n):
            sel_indice = np.random.choice(n, p=p_sce)
            X_sel_antlion = MX_antlion[sel_indice,:].copy()
            RA = passeio_aleatorio(d, Linf, Lsup, X_sel_antlion, t, t_max)
            RE = passeio_aleatorio(d, Linf, Lsup, X_elite_antlion, t, t_max)
            MX_ant[i,:] = (RA[t,:] + RE[t,:])/2

        pos1.append(MX_ant[-1,:].copy())

        # Atualizar a aptidao das formigas
        MF_ant = np.apply_along_axis(fobj, 1, MX_ant).reshape(-1,1)
        # Concatenar formigas e formigas-leao
        MX_combined = np.concatenate((MX_antlion, MX_ant), axis=0)
        MF_combined = np.concatenate((MF_antlion, MF_ant), axis=0)
        # Ordenar
        MX_combined = MX_combined[np.argsort(MF_combined, axis=0).flatten()]
        MF_combined = np.sort(MF_combined, axis=0)
        # Atualizar a aptidao das formigas-leao
        MX_antlion = MX_combined[:n,:].copy()
        MF_antlion = MF_combined[:n,:].copy()
        # Atualizar a elite
        if MF_antlion[0].item() < f_elite_antlion:
            X_elite_antlion = MX_antlion[0,:].copy()
            f_elite_antlion = MF_antlion[0].item()
        # Mantem a elite na populacao
        MX_antlion[0,:] = X_elite_antlion.copy()
        MF_antlion[0] = f_elite_antlion

        historico.append([f_elite_antlion, MF_antlion.mean()])
        print('Concluida.')
        print('Melhor solucao:', np.round(X_elite_antlion,4), np.round(f_elite_antlion,4))
        t+=1

        if np.any(MX_ant > Lsup) or np.any(MX_ant < Linf) or np.any(MX_antlion > Lsup) or np.any(MX_antlion < Linf):
            sys.exit()    
    return MX_ant, MX_antlion, historico, np.array(pos1)

# %%
# history = alo(Linf, Lsup, n, t_max)
MX_ant, MX_antlion, historico, pos1 = alo([-512, -512], [512, 512], 50, 500)
plt.plot([i[0] for i in historico])
plt.plot([i[1] for i in historico])
plt.show()
# %%
