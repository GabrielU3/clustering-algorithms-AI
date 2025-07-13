import numpy as np

def calcular_ari(rotulos_reais, rotulos_pred):
    """
    Calcula o Índice Rand Ajustado (ARI) entre duas partições.
    
    Args:
        rotulos_reais: array com os rótulos reais
        rotulos_pred: array com os rótulos preditos
    
    Returns:
        ari: valor do Índice Rand Ajustado
    """
    n = len(rotulos_reais)
    
    # Criar matriz de contingência
    rotulos_reais_unicos = np.unique(rotulos_reais)
    rotulos_pred_unicos = np.unique(rotulos_pred)
    
    # Mapear rótulos para índices consecutivos
    mapa_reais = {rotulo: i for i, rotulo in enumerate(rotulos_reais_unicos)}
    mapa_pred = {rotulo: i for i, rotulo in enumerate(rotulos_pred_unicos)}
    
    # Converter rótulos para índices
    rotulos_reais_idx = np.array([mapa_reais[r] for r in rotulos_reais])
    rotulos_pred_idx = np.array([mapa_pred[r] for r in rotulos_pred])
    
    # Número de clusters
    n_clusters_reais = len(rotulos_reais_unicos)
    n_clusters_pred = len(rotulos_pred_unicos)
    
    # Calcular matriz de contingência
    cont_table = np.zeros((n_clusters_reais, n_clusters_pred))
    for i in range(n):
        cont_table[rotulos_reais_idx[i], rotulos_pred_idx[i]] += 1
    
    # Calcular somas das linhas e colunas
    row_sum = np.sum(cont_table, axis=1)
    col_sum = np.sum(cont_table, axis=0)
    
    # Calcular somas dos quadrados
    sum_cont_table = np.sum(cont_table**2)
    sum_row_sum = np.sum(row_sum**2)
    sum_col_sum = np.sum(col_sum**2)
    
    # Calcular o número esperado de pares concordantes
    expected_index = sum_row_sum * sum_col_sum / float(n**2)
    
    # Calcular o índice máximo
    max_index = (sum_row_sum + sum_col_sum) / 2.0
    
    # Calcular o índice mínimo
    min_index = min(sum_row_sum, sum_col_sum)
    
    # Calcular o índice Rand ajustado
    if max_index == expected_index:
        ari = 0.0
    else:
        ari = (sum_cont_table - expected_index) / (max_index - expected_index)
    
    return ari 