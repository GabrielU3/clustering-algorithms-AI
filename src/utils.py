import numpy as np

# Função para ler dados de um arquivo .txt
# Agora retorna (dados, ids), onde dados é um np.array e ids é uma lista de strings

def ler_dados_txt(caminho):
    """Lê um arquivo .txt de dados, ignora o header e a coluna de ids, retorna (dados, ids)."""
    ids = []
    dados = []
    with open(caminho, 'r') as f:
        next(f)  # pula o header
        for linha in f:
            if linha.strip():
                partes = linha.strip().split('\t')
                ids.append(partes[0])
                dados.append([float(x) for x in partes[1:]])
    return np.array(dados), ids

# Função para ler rótulos reais de um arquivo .clu
# Aceita separador tab ou espaço

def ler_rotulos_clu(caminho):
    """Lê um arquivo .clu de rótulos e retorna uma lista de inteiros (apenas a segunda coluna, separador tab ou espaço)."""
    rotulos = []
    with open(caminho, 'r') as f:
        for linha in f:
            if linha.strip():
                partes = linha.strip().replace('\t', ' ').split()
                rotulos.append(int(partes[-1]))
    return rotulos

# Função para salvar partições no formato .clu
# Usa tabulação como separador

def salvar_particao_clu(ids, rotulos, caminho):
    """Salva os ids e rótulos em um arquivo .clu (separador tab)."""
    with open(caminho, 'w') as f:
        for i, r in zip(ids, rotulos):
            f.write(f"{i}\t{r}\n") 