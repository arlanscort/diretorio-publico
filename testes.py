#%%
import numpy as np 

def eggholder(X):
    x1 = X[0]
    x2 = X[1]
    f = -(x2+47)*np.sin(np.sqrt(np.abs(x1/2+(x2+47)))) - x1*np.sin(np.sqrt(np.abs(x1-(x2+47))))
    return np.around(f, decimals=4)


#%% Teste SCE
import sce
import plotly.graph_objects as go
fig = go.Figure()
for i in range(3):
    D, Fs = sce.sce([-512, -512], [512, 512], eggholder, itmax=500, p=20)
    
    minimos  = [np.min(i) for i in Fs]
    medias = [np.mean(i) for i in Fs]

    fig.add_trace(go.Scatter(x=np.arange(1, len(Fs)+1), y=minimos, name='Pop mínimos'))
    fig.add_trace(go.Scatter(x=np.arange(1, len(Fs)+1), y=medias, name='Pop médias'))

fig.show()

#%%
