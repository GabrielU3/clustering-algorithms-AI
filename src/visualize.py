import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_clusters(dados, rotulos, ids, caminho_saida, titulo=None, destaque=False):
    """Plota os clusters em 2D e salva o gráfico."""
    plt.figure(figsize=(6,5))
    scatter = plt.scatter(dados[:,0], dados[:,1], c=np.array(rotulos) - 1, cmap='tab10', s=30, edgecolor='k')
    if destaque:
        plt.gca().set_facecolor('#ffe680')
    plt.title(titulo or "Clusters")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.savefig(caminho_saida)
    plt.close() 