'''
Shuffled Complex Evolution
Implementacao: Arlan Scortegagna, jul/2021
Revisao: Arlan Scortegagna, jul/2021
'''

printa = 30

import numpy as np

def lhs(Xmin, Xmax, s):
    '''
    Latin Hypercube Sampling (LHS) - Amostragem por Hipercubo Latino
    Considera pontos aleatórios nos "slices"
    Entradas
        Xmin - array contendo os limites superiores do espaco factivel
        Xmax - array contendo os limites inferiores do espaco factivel
        s - numero de pontos
    Saidas
        X - array contendo a amostra
    '''

    n = len(Xmin)
    slices = np.linspace(Xmin, Xmax, s+1)
    X = np.random.uniform(low=slices[:-1], high=slices[1:]) # [s x n]
    X = X.T # [n x s]
    for i in range(n):
        np.random.shuffle(X[i])
    X = X.T # [n x s]
    return X

cont = 0

def cce(A, Xmax, Xmin, fobj, m, q, beta, alfa, Probs):
    '''
    Complex Competitive Evolution (CCE)
    Entradas
        Xmin - array contendo os limites inferiores do espaco viavel
        Xmax - array contendo os limites superiores do espaco viavel
        fobj - funcao objetivo
        m - numero de pontos em um complexo
        q - numero de pontos em um subcomplexo
        beta - numero de selecoes de pais do complexo realizadas no CCE
        alfa - numero de aplicacoes de Nelder-Mead no CCE
    Saidas
        D - lista contendo dicionarios X,f com a populacao final
        Fs - lista contendo todas as funcoes objetivo avaliadas
    '''
    global cont

    # Menor hipercubo contendo A
    Hmin = np.array([i['X'] for i in A]).min(axis=0)
    Hmax = np.array([i['X'] for i in A]).max(axis=0)

    # Loop de evolucao dos complexos por selecao aleatoria
    for i in range(beta):

        L = np.random.choice(m, size=q, replace=False, p=Probs)
        B = [A[i] for i in L]

        # Loop de evolucao dos subcomplexos por Nelder-Mead
        for j in range(alfa):

            # Ordena o subcomplexo
            B = sorted(B, key=lambda x : x['f'])
            # Calcula o centroide (g) e o pior ponto (uq)
            Xs = np.array([i['X'] for i in B])
            g = Xs[:-1].mean(axis=0)
            uq = Xs[-1]
            # Reflexao
            r = 2*g-uq
            if np.any(r < Xmin) or np.any(r > Xmax):
                # Mutacao
                r = np.random.uniform(Hmin, Hmax)
            fr = fobj(r)
            cont+=1
            if cont%printa == 0 : print(cont)
            if fr < B[-1]['f']: # fr < fq ?
                B[-1]['X'] = r
                B[-1]['f'] = fr
            else:
                # Contracao
                c = (g+uq)/2
                fc = fobj(c)
                cont+=1
                if cont%printa == 0 : print(cont)
                if fc < B[-1]['f']: # fc < fq ?
                    B[-1]['X'] = c
                    B[-1]['f'] = fc
                else:
                    # Mutacao
                    z = np.random.uniform(Hmin, Hmax)
                    B[-1]['X'] = z
                    B[-1]['f'] = fobj(z)
                    cont+=1
                    if cont%printa == 0 : print(cont)

        # Retorna os subcomplexos B ao complexo A, de acordo com L, e ordena
        for Li, Bi in zip(L, B):
            A[Li] = Bi
        A = sorted(A, key=lambda x : x['f'])

    return A


def sce(Xmin, Xmax, fobj, itmax, p, m=None, q=None, alfa=None, beta=None):
    '''
    Shuffled Complex Evolution (SCE)
    Entradas
        Xmin - array contendo os limites inferiores do espaco viavel
        Xmax - array contendo os limites superiores do espaco viavel
        fobj - funcao objetivo
        itmax - numero maximo de iteracoes
        p - numero de complexos
        m - numero de pontos em um complexo
        q - numero de pontos em um subcomplexo
        beta - numero de selecoes de pais do complexo realizadas no CCE
        alfa - numero de aplicacoes de Nelder-Mead no CCE
    Saidas
        D - lista contendo dicionarios X,f com a populacao final
        Fs - lista contendo todas as funcoes objetivo avaliadas
    '''
    global cont

    # Parametros do SCE (Duan et al., 1994)
    n = len(Xmin)
    if m is None:
        m = 2*n+1
    if q is None:
        q = n+1
    if alfa is None:
        alfa = 1
    if beta is None:
        beta = 2*n+1

    # Definicoes
    s = p*m
    Probs = [2*(m+1-i)/(m*(m+1)) for i in range (1, m+1)] # Probabilidades de selecao para o algoritmo CCE
    IDX = [[k+p*(j-1)-1 for j in range (1, m+1)] for k in range(1, p+1)] # Indices para o particionamento de D (notar que ja esta em 0-based indexing)

    # Convergencia
    DX = np.asarray(Xmax) - np.asarray(Xmin)

    # Populacao inicial (D0)
    print('Gerando populacao inicial...')
    X = lhs(Xmin, Xmax, s)
    D = []
    for x in X:
        D.append(dict(X=x, f=fobj(x)))
        cont+=1
        if cont%printa == 0 : print(cont)

    # Ordena D0
    it = 0
    D = sorted(D, key=lambda x:x['f'])
    Fs = [[i['f'] for i in D]]
    print(f'Iteracao {it}: fmin={Fs[it][0]:.2f}')

    # Itera ate a convergencia
    while True:

        # Particiona D nos complexos A
        for k in range(p):
            A = [D[i] for i in IDX[k]]

            # Evolui A com o CCE
            print(f'Evoluindo complexo {k+1}')
            A = cce(A, Xmax, Xmin, fobj, m, q, beta, alfa, Probs)

            # Substitui A em D
            for i, Ai in zip(IDX[k], A):
                D[i] = Ai

        # Ordena D
        it += 1
        D = sorted(D, key=lambda x:x['f'])
        Fs.append([i['f'] for i in D])
        print(f'Iteracao {it} : fmin={Fs[it][0]:.2f}')

        # Verifica os criterios de convergencia
        # 1 - Numero maximo de iteracoes
        if it == itmax:
            break
        # 2 - Ganho obtido nas ultimas iteracoes

        # 3 - Tamanho do espaco de busca
        X = np.array([i['X'] for i in D])
        DXit = np.max(X, axis=0) - np.min(X, axis=0)
        if np.max(DXit/DX) <= 10**(-3)/100:
            break

    return D, Fs


if __name__ == '__main__':
    # Otimiza a eggholder (com p=10 e np.random.seed(6) atinge o minimo global)
    np.random.seed(6)
    from funcoes_teste import eggholder
    Xmin = [-512, -512]
    Xmax = [ 512,  512]
    fobj = eggholder
    itmax = 1000
    p = 10
    D, Fs = sce(Xmin, Xmax, fobj, itmax, p)
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
                    x=[i for i in range(len(Fs))],
                    y=[i[0] for i in Fs],
                    mode='lines',
                    name='fmin',
                    line=dict(color='red')
                    ))
    fig.add_trace(go.Scatter(
                    x=[i for i in range(len(Fs))],
                    y=[i[-1] for i in Fs],
                    mode='lines',
                    name='fmax',
                    line=dict(color='black')
                    ))
    fig.show()
