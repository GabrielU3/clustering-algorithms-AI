import numpy as np
from src.utils import salvar_particao_clu


def kmeans(dados, k, max_iter, random_state=42):
    np.random.seed(random_state)
    n_samples, n_features = dados.shape

    # Inicialização aleatória dos centroides
    indices = np.random.choice(n_samples, k, replace=False)
    centroides = dados[indices].copy()

    for iteracao in range(max_iter):
        # Atribuição: cada ponto ao centroide mais próximo
        distancias = np.zeros((n_samples, k))
        for i in range(k):
            distancias[:, i] = np.sum((dados - centroides[i])**2, axis=1)

        rotulos = np.argmin(distancias, axis=1)

        # Atualização: recalcular centroides
        centroides_antigos = centroides.copy()
        for i in range(k):
            if np.sum(rotulos == i) > 0:  # se há pontos no cluster
                centroides[i] = np.mean(dados[rotulos == i], axis=0)

        # Critério de parada: centroides não mudaram
        if np.allclose(centroides, centroides_antigos):
            break

    return rotulos


def rodar_kmeans(dados, k, n_iter, ids, caminho_saida):
    # Executa o K-médias e salva a partição gerada em .clu.
    rotulos = kmeans(dados, k, n_iter, random_state=42)
    salvar_particao_clu(ids, rotulos, caminho_saida)
    return rotulos
