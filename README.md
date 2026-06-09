# POLARIS Earth — Sistema de Monitoramento de Riscos Ambientais com Satélites

**FIAP — Global Solution 2026 | Estruturas de Dados e Algoritmos**  
Prof. André Marques | 1º Semestre de 2026

---

## Integrantes

| Nome | RM |
|------|----|
| William Queiroz | 565032 |
| Kauê de Almeida Pena | 564211 |
| Eduardo Delorenzo Moraes | 561749 |
| Lucas Rowlands Abat | 562994 |
| Ronaldo Aparecido Monteiro Almeida | 565017 |

---

## Descrição

O **POLARIS Earth** é um sistema de monitoramento e triagem de riscos ambientais desenvolvido para apoiar equipes da Defesa Civil, Corpo de Bombeiros e órgãos ambientais brasileiros. O sistema integra dados de satélites (NASA FIRMS, GOES-16, ESA Sentinel-2) para calcular automaticamente a melhor sequência de atendimento a municípios em situação de risco, eliminando a tomada de decisão no improviso.

O sistema modela os municípios como um **grafo ponderado** onde os vértices são cidades e as arestas são rodovias com tempo de deslocamento como peso. Sobre esse grafo, uma **Árvore Binária de Busca (BST)** organiza os municípios por índice de risco (0 a 1), permitindo triagem eficiente. Dois algoritmos são executados em paralelo: **Força Bruta** para validar a solução ótima em instâncias pequenas (N ≤ 12) e **algoritmos Gulosos** (Prim, Dijkstra, Kruskal) para resolver instâncias reais com centenas de municípios em milissegundos.

Dois cenários brasileiros reais são implementados:

-  **Cenário A** — Rede de resposta a enchentes na região metropolitana de Porto Alegre (RS, 2024)
-  **Cenário B** — Triagem de risco de seca nos municípios do MATOPIBA (MA, TO, PI, BA), com índice de risco derivado de NDVI MODIS/NASA e precipitação INMET

> Projeto alinhado aos **ODS 2, 9, 11 e 13** da ONU.

## Dependências

O projeto utiliza **Python 3.10 ou superior**. Todas as dependências estão em `requirements.txt`.

| Biblioteca | Uso |
|------------|-----|
| `matplotlib` | Geração dos gráficos e figuras obrigatórias |
| `networkx` | Construção e visualização do grafo de municípios |
| `numpy` | Operações numéricas auxiliares |
| `reportlab` | Geração do relatório técnico em PDF |

>  As estruturas centrais do projeto (Grafo, BST, UnionFind) foram **implementadas do zero** em Python puro, sem bibliotecas externas, conforme exigido pelo enunciado.

---

##  Instalação

**1. Clone o repositório:**

```bash
git clone https://github.com/seu-usuario/global-solution-2026-fund.git
cd global-solution-2026-fund
```

**2. Crie e ative um ambiente virtual (recomendado):**

```bash

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Instale as dependências:**

```bash
pip install -r requirements.txt
```

---

##  Como Executar

Todos os módulos são acessados pelo `main.py` via flags. Execute a partir da raiz do projeto:

```bash
# Executa tudo em sequência (recomendado para entrega)
python src/main.py

# Módulos individuais
python src/main.py --estruturas   # Demonstra grafo, BST, BFS e busca por intervalo
python src/main.py --algoritmos   # Roda Força Bruta + Greedy + gap de otimalidade
python src/main.py --benchmark    # Benchmark N = 5, 8, 10, 12, 20, 50, 100
python src/main.py --figuras      # Gera as 5 figuras obrigatórias em report/
```

### Dashboard interativo

Para acessar o dashboard acesse pelo link abaixo: <br>
 https://williamqbr.github.io/GS_Dynamic_Program/

---

##  Descrição dos Módulos

| Módulo | Responsabilidade |
|--------|-----------------|
| `data_structures.py` | Grafo (lista de adjacência), BST com Node, BFS, DFS, cenários RS e MATOPIBA |
| `brute_force.py` | Backtracking recursivo, cobertura por subconjuntos, análise de explosão combinatória |
| `greedy.py` | Prim, Dijkstra, Kruskal, Union-Find, rota gulosa por razão risco/distância |
| `performance_monitor.py` | `time.perf_counter()`, `tracemalloc`, benchmark com mediana de 3 repetições |
| `visualizations.py` | 5 figuras obrigatórias: grafo MST, BST, desempenho×N, estruturas, gap de otimalidade |

---

##  Fontes de Dados

| Fonte | Dado utilizado | URL |
|-------|---------------|-----|
| NASA FIRMS | Focos de calor (MODIS) | [earthdata.nasa.gov](https://earthdata.nasa.gov) |
| INPE PRODES/DETER | Desmatamento Amazônia | [terrabrasilis.dpi.inpe.br](http://terrabrasilis.dpi.inpe.br) |
| ANA | Pluviometria e hidrologia | [hidroweb.ana.gov.br](https://hidroweb.ana.gov.br) |
| INMET | Dados climáticos históricos | [bdmep.inmet.gov.br](https://bdmep.inmet.gov.br) |
| IBGE | Malha municipal | [ibge.gov.br/geociencias](https://ibge.gov.br/geociencias) |
| DNIT | Malha viária federal | [dnit.gov.br](https://dnit.gov.br) |
| Defesa Civil RS | Dados enchentes 2024 | [defesacivil.rs.gov.br](https://defesacivil.rs.gov.br) |

> Os dados utilizados no projeto são **sintéticos representativos**, construídos a partir dos valores históricos e geográficos reais das fontes acima, conforme justificado no relatório técnico.
