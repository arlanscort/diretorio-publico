############
# Testes de implementacao 2
# Objetivo - comparar o desempenho da implementacao alo_arlan, alo_valdecy e alo_mirjalili (oficial)
# em um conjunto de funcoes de teste.
# Todos os algoritmos foram setados em 25 formigas e 500 iteracoes.
############

import numpy as np
import matplotlib.pyplot as plt 

def eggholder(X):
    x1 = X[0]
    x2 = X[1]
    f = -(x2+47)*np.sin(np.sqrt(np.abs(x1/2+(x2+47)))) - x1*np.sin(np.sqrt(np.abs(x1-(x2+47))))
    return np.around(f, decimals=4)

if __name__ == '__main__':
    import pandas as pd
    import alo_valdecy
    import alo
    import alo_arlan
    
    repeticoes = 100
    t_max = 500
    n = 25
    UB = [512, 512]
    LB = [-512, -512]
    eggholder_min = -959.6407

    # Experimento 1 - Funcao eggholder
    mins = {'valdecy':[], 'mirjalili':[], 'arlan':[]}
    meds = {'valdecy':[], 'mirjalili':[], 'arlan':[]}
    counts = {'valdecy':0, 'mirjalili':0, 'arlan':0}
    for i in range(repeticoes):
        # ALO - Valdecy
        elite, geracoes = alo_valdecy.ant_lion_optimizer(n, LB, UB, t_max, eggholder)
        mins_valdecy = [np.min(np.array(i)[:,-1]) for i in geracoes][:500]
        mins['valdecy'].append(mins_valdecy)
        meds_valdecy = [np.mean(np.array(i)[:,-1]) for i in geracoes][:500]
        meds['valdecy'].append(meds_valdecy)
        if elite[-1] < -959:
            counts['valdecy'] += 1
        
        # ALO - Mirjalili
        _, f_elite_antlion, _, F_geracoes = alo.alo(eggholder, UB, LB, n, t_max)
        mins_mirjalili = [np.min(i) for i in F_geracoes]
        mins['mirjalili'].append(mins_mirjalili)
        meds_mirjalili = [np.mean(i) for i in F_geracoes]
        meds['mirjalili'].append(meds_mirjalili)
        if f_elite_antlion < -959:
            counts['mirjalili'] += 1

        # ALO - Arlan
        _, f_elite_antlion, _, F_geracoes = alo_arlan.alo(eggholder, UB, LB, n, t_max)
        mins_arlan = [np.min(i) for i in F_geracoes]
        mins['arlan'].append(mins_arlan)
        meds_arlan = [np.mean(i) for i in F_geracoes]
        meds['arlan'].append(meds_arlan)
        if f_elite_antlion < -959:
            counts['arlan'] += 1

        # mins_mirjalili = [np.min(np.array(i)[:,-1]) for i in geracoes][:500]
        # mins['valdecy'].append(mins_valdecy)
        # meds_valdecy = [np.mean(np.array(i)[:,-1]) for i in geracoes][:500]
        # meds['valdecy'].append(meds_valdecy)
        # ALO - Arlan


    fig, axs = plt.subplots(2, 3, figsize=(10,8))
    for i, implementacao in enumerate(['valdecy', 'mirjalili', 'arlan']):
        axs[0,i].set_title(f'{implementacao}\nacertos-{counts[implementacao]}/{repeticoes}')
        for j in range(repeticoes):
            axs[0,i].plot(np.arange(1, 501), mins[implementacao][j], color='red')   
            axs[1,i].plot(np.arange(1, 501), meds[implementacao][j], color='blue')
    axs[0,0].set_ylabel('Mínimos')
    axs[1,0].set_ylabel('Médias')
    [ax.axhline(-959.6407, color='black') for ax in axs.flatten()]
    [ax.set_ylim(-1000, -200) for ax in axs.flatten()]
    plt.tight_layout()
    plt.savefig('eggholder.jpg')
    plt.show()