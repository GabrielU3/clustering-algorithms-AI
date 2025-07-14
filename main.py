import csv
import os
import random
from src.utils import ler_dados_txt, ler_rotulos_clu
from src.kmeans import rodar_kmeans
from src.linkage import rodar_linkage
from src.rand_index import calcular_ari
from src.visualize import plot_clusters

# Definição dos datasets e parâmetros
DATASETS = [
    {"nome": "c2ds1-2sp", "txt": "datasets/c2ds1-2sp.txt",
        "clu": "datasets/c2ds1-2spReal.clu", "kmin": 2, "kmax": 5},
    {"nome": "c2ds3-2g", "txt": "datasets/c2ds3-2g.txt",
        "clu": "datasets/c2ds3-2gReal.clu", "kmin": 2, "kmax": 5},
    {"nome": "monkey", "txt": "datasets/monkey.txt",
        "clu": "datasets/monkeyReal1.clu", "kmin": 5, "kmax": 12},
]

RESULTS_DIR = "results"
PART_DIR = os.path.join(RESULTS_DIR, "particoes")
GRAPH_DIR = os.path.join(RESULTS_DIR, "graficos")
ARI_PATH = os.path.join(RESULTS_DIR, "ARI.csv")

os.makedirs(PART_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# Estrutura para armazenar resultados ARI
with open(ARI_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "algoritmo", "k", "ARI"])

for ds in DATASETS:
    print(f"\n=== Processando dataset: {ds['nome']} ===")
    print(f"Lendo dados de {ds['txt']}...")
    dados, ids = ler_dados_txt(ds["txt"])
    print(
        f"Dados carregados: {len(dados)} amostras, {dados.shape[1]} features")

    print(f"Lendo rótulos reais de {ds['clu']}...")
    rotulos_reais = ler_rotulos_clu(ds["clu"])
    print(f"Rótulos reais carregados: {len(rotulos_reais)} rótulos")

    # K-means
    print(
        f"\n--- Executando K-means para k entre {ds['kmin']} e {ds['kmax']} ---")
    k = random.randint(ds["kmin"], ds["kmax"])
    print(f"K-means k={k}...")
    part_path = os.path.join(PART_DIR, f"{ds['nome']}_kmeans_k{k}.clu")
    rotulos_pred = rodar_kmeans(dados, k, 20, ids, part_path)
    ari = calcular_ari(rotulos_reais, rotulos_pred)
    print(f"ARI para K-means k={k}: {ari:.4f}")

    with open(ARI_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ds['nome'], 'kmeans', k, ari])

    graph_path = os.path.join(GRAPH_DIR, f"{ds['nome']}_kmeans_k{k}.png")
    plot_clusters(dados, rotulos_pred, ids, graph_path,
                  titulo=f"{ds['nome']} K-means k={k}")
    print(f"Gráfico salvo: {graph_path}")

    # Linkage
    for modo in ['single', 'complete']:
        print(
            f"\n--- Executando {modo}-linkage para k de {ds['kmin']} a {ds['kmax']} ---")
        rodar_linkage(dados, ids, ds["kmin"],
                      ds["kmax"], modo, PART_DIR, ds["nome"])

        for k in range(ds["kmin"], ds["kmax"]+1):
            print(f"Calculando ARI para {modo}-linkage k={k}...")
            part_path = os.path.join(
                PART_DIR, f"{ds['nome']}_linkage_{modo}_k{k}.clu")
            rotulos_pred = ler_rotulos_clu(part_path)
            ari = calcular_ari(rotulos_reais, rotulos_pred)
            print(f"ARI para {modo}-linkage k={k}: {ari:.4f}")

            with open(ARI_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ds['nome'], f'linkage_{modo}', k, ari])

            graph_path = os.path.join(
                GRAPH_DIR, f"{ds['nome']}_linkage_{modo}_k{k}.png")
            plot_clusters(dados, rotulos_pred, ids, graph_path,
                          titulo=f"{ds['nome']} {modo}-linkage k={k}")
            print(f"Gráfico salvo: {graph_path}")

print(f"\n=== Execução concluída! ===")
print(f"Resultados salvos em: {RESULTS_DIR}")
print(f"Tabela ARI: {ARI_PATH}")
