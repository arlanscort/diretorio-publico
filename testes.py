#%%
import numpy as np

#%%
def eggholder(X):
    x1 = X[0]
    x2 = X[1]
    f = -(x2+47)*np.sin(np.sqrt(np.abs(x1/2+(x2+47)))) - x1*np.sin(np.sqrt(np.abs(x1-(x2+47))))
    return np.around(f, decimals=4)

#%%
# if __name__ == '__main__':
    
#     import matplotlib.pyplot as plt
#     fig, axs = plt.subplots(2, 2, figsize=(10,8))
    
#     repeticoes = 50
#     iteracoes = 100
#     idx = np.arange(1, iteracoes+1)
#     eggholder_min = -959.6407
    
#     # # Teste SCE - Arlan
#     # import sce
#     # count = 0
#     # for i in range(repeticoes):    
#     #     _, Fs = sce.sce(
#     #         Xmin = [-512, -512],
#     #         Xmax = [512, 512],
#     #         fobj = eggholder,
#     #         itmax = 100,
#     #         p = 20)
#     #     if np.min(Fs[-1]) <= eggholder_min+0.01:
#     #         count += 1
#     #     minimos = [min(i) for i in Fs]
#     #     medias = [np.mean(i) for i in Fs]
#     #     axs[0,1].plot(np.arange(1, len(minimos)+1), minimos, color='red')
#     #     axs[1,1].plot(np.arange(1, len(medias)+1), medias, color='blue')
#     # axs[0,1].set_title(f'SCE Arlan\nAcertos {count}/{repeticoes}')
    
#     # Teste ALO - Valdecy
#     import alo_valdecy
#     count = 0
#     for i in range(repeticoes):    
#         antlions, minimos, medias, elite = alo_valdecy.ant_lion_optimizer(
#             colony_size = 40,
#             min_values = [-512,-512],
#             max_values = [512,512],
#             iterations = 500,
#             target_function = eggholder
#         )
#         print(elite[-1])
        
        
        
#         if elite[-1] <= eggholder_min+0.01:
#             count += 1
#         axs[0,0].plot(idx, minimos[-iteracoes:], color='red')
#         axs[1,0].plot(idx, medias[-iteracoes:], color='blue')
#     axs[0,0].set_title(f'ALO Valdecy\nAcertos {count}/{repeticoes}')

    
#     # Teste ALO - Arlan

    
#     # Ajuste de layout
#     axs[0,0].set_ylabel('Geração - Mínimo')
#     axs[1,0].set_ylabel('Geração - Média')
#     [ax.axhline(-959.6407, color='black') for ax in axs.flatten()]
#     [ax.set_ylim(-1000, -200) for ax in axs.flatten()]
#     plt.tight_layout()
#     plt.savefig('eggholder.jpg')


# %%
# Experimento - Implementacao Valdecy
import alo_valdecy
repeticoes = 50
lista = []
# for formigas in [10, 25, 50, 100]:
    # for iteracoes in [50, 100, 500]:
for formigas in [50]:
    for iteracoes in [500]:
        count = 0
        for i in range(repeticoes):
            antlions, minimos, medias, elite = alo_valdecy.ant_lion_optimizer(
            colony_size = formigas,
            min_values = [-512,-512],
            max_values = [512,512],
            iterations = iteracoes,
            target_function = eggholder
            )
            if elite[-1] <= - 959.63:
                count += 1
            lista.append(elite)
        print(count)
        lista.append([formigas, iteracoes, count])
# df = pd.DataFrame(data=lista, columns=['Formigas', 'Iteracoes', 'Acertos/50'])
# df.to_excel('implementacao_python_valdecy.xlsx')

import pandas as pd 
ex = pd.DataFrame(data=lista, columns=['x1', 'x2', 'f'])
ex.to_excel('ex.xlsx')