#%%
import numpy as np

#%%
def eggholder(X):
    x1 = X[0]
    x2 = X[1]
    f = -(x2+47)*np.sin(np.sqrt(np.abs(x1/2+(x2+47)))) - x1*np.sin(np.sqrt(np.abs(x1-(x2+47))))
    return np.around(f, decimals=4)

#%%
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2)
    
    repeticoes = 30
    iteracoes = 100
    idx = np.arange(1, iteracoes+1)
    # Teste SCE - Arlan


    # Teste ALO - Valdecy
    import alo_valdecy
    for i in range(repeticoes):
        
        antlions, minimos, medias, elite = alo_valdecy.ant_lion_optimizer(
            colony_size = 80,
            min_values = [-512,-512],
            max_values = [512,512],
            iterations = iteracoes,
            target_function = eggholder)
        axs[0,0].plot(idx, minimos[-iteracoes:], color='red')
        axs[1,0].plot(idx, medias[-iteracoes:], color='blue')
        
    [i.set_title('ALO Valdecy') for i in axs[:,0]]
    for ax in axs.flatten():
        ax.axhline(y=-959.6407, color='black')
        ax.set_ylim(-1000, 0)
    # Teste ALO - Arlan
    plt.show()
# %%
