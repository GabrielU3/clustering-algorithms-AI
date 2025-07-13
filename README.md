# Projeto de Clusterização: K-Médias, Single-Link e Complete-Link

Este projeto implementa e compara três algoritmos de agrupamento (K-médias, Single-Linkage, Complete-Linkage) em três conjuntos de dados, avaliando as partições pelo Índice Rand Ajustado (ARI).

## Estrutura
```
datasets/         # Dados de entrada (.txt, .clu)
src/              # Código-fonte dos algoritmos e utilitários
results/
  particoes/      # Partições geradas (.clu)
  graficos/       # Gráficos gerados (.png)
  ARI.csv         # Tabela de ARI
main.py           # Script principal
README.md         # Este arquivo
```

## Como Executar
1. Instale as dependências:
   ```bash
   pip install numpy scikit-learn scipy matplotlib seaborn
   ```
2. Execute o script principal:
   ```bash
   python main.py
   ```

## Saídas
- Arquivos .clu para cada partição
- Gráficos dos clusters
- Tabela ARI.csv com os resultados

## Observações
- Os dados devem estar na pasta `datasets/`.
- Resultados são salvos em `results/`.
- Compare os resultados com o notebook de referência fornecido. 