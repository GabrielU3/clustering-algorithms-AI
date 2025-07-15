import numpy as np
from src.utils import salvar_particao_clu

def calcular_distancias(dados):
    """Calcula a matriz de distâncias euclidianas entre todos os pares de pontos."""
    n = len(dados)
    distancias = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((dados[i] - dados[j])**2))
            distancias[i, j] = dist
            distancias[j, i] = dist
    return distancias

def linkage_hierarquico_otimizado(dados, metodo='single'):
    """
    Implementação otimizada do agrupamento hierárquico.
    
    Args:
        dados: array numpy com os dados
        metodo: 'single' ou 'complete'
    
    Returns:
        Z: matriz de ligação (linkage matrix)
    """
    n = len(dados)
    
    # Calcular distâncias iniciais entre todos os pares
    distancias = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((dados[i] - dados[j])**2))
            distancias[i, j] = dist
            distancias[j, i] = dist
    
    # Inicializar clusters como pontos individuais
    clusters = [[i] for i in range(n)]
    Z = []
    
    # Para cada nível da hierarquia
    for nivel in range(n-1):
        # Encontrar o par de clusters mais próximo
        min_dist = float('inf')
        i_min, j_min = -1, -1
        
        # Otimização: usar apenas a parte triangular superior da matriz
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                if metodo == 'single':
                    # Single-linkage: distância mínima entre pontos dos clusters
                    dist = float('inf')
                    for p1 in clusters[i]:
                        for p2 in clusters[j]:
                            if p1 < p2:
                                dist = min(dist, distancias[p1, p2])
                            else:
                                dist = min(dist, distancias[p2, p1])
                else:  # complete
                    # Complete-linkage: distância máxima entre pontos dos clusters
                    dist = 0
                    for p1 in clusters[i]:
                        for p2 in clusters[j]:
                            if p1 < p2:
                                dist = max(dist, distancias[p1, p2])
                            else:
                                dist = max(dist, distancias[p2, p1])
                
                if dist < min_dist:
                    min_dist = dist
                    i_min, j_min = i, j
        
        # Mesclar os clusters mais próximos
        cluster_novo = clusters[i_min] + clusters[j_min]
        
        # Adicionar à matriz de ligação
        Z.append([i_min, j_min, min_dist, len(cluster_novo)])
        
        # Atualizar lista de clusters
        clusters = [clusters[k] for k in range(len(clusters)) if k != i_min and k != j_min]
        clusters.append(cluster_novo)
    
    return np.array(Z)


def fcluster_custom(Z, k, criterion='maxclust'):
    """
    Implementação da função fcluster do zero.
    
    Args:
        Z: matriz de ligação
        k: número de clusters desejado
        criterion: critério de corte (sempre 'maxclust')
    
    Returns:
        rotulos: array com os rótulos dos clusters
    """
    n = len(Z) + 1
    rotulos = np.zeros(n, dtype=int)
    
    # Se k == n, cada ponto é um cluster
    if k == n:
        return np.arange(n)
    
    # Se k == 1, todos os pontos estão no mesmo cluster
    if k == 1:
        return np.zeros(n)
    
    # Para outros valores de k, cortar a árvore no nível apropriado
    # O número de clusters diminui de n para 1 conforme a árvore é construída
    # Precisamos cortar quando temos k clusters
    
    # Construir a árvore de clusters
    clusters = [[i] for i in range(n)]
    
    for i, (idx1, idx2, dist, size) in enumerate(Z):
        if len(clusters) == k:
            break
        
        # Converter índices para inteiros
        idx1 = int(idx1)
        idx2 = int(idx2)
        
        # Mesclar clusters
        cluster_novo = clusters[idx1] + clusters[idx2]
        clusters = [clusters[j] for j in range(len(clusters)) if j != idx1 and j != idx2]
        clusters.append(cluster_novo)
    
    # Atribuir rótulos
    for cluster_id, cluster in enumerate(clusters):
        for ponto in cluster:
            rotulos[ponto] = cluster_id
    
    return rotulos

def rodar_linkage(dados, ids, k_min, k_max, modo, pasta_saida, nome_dataset):
    """Executa single-linkage ou complete-linkage para k em [k_min, k_max] e salva as partições."""
    print(f"Executando {modo}-linkage para k de {k_min} a {k_max}...")
    Z = linkage_hierarquico_otimizado(dados, metodo=modo)
    print(f"Matriz de ligação construída. Gerando partições...")
    
    for k in range(k_min, k_max+1):
        print(f"Gerando partição para k={k}...")
        rotulos = fcluster_custom(Z, k, criterion='maxclust')
        caminho = f"{pasta_saida}/{nome_dataset}_linkage_{modo}_k{k}.clu"
        salvar_particao_clu(ids, rotulos, caminho)
        print(f"Partição k={k} salva em {caminho}") 
