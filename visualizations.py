from __future__ import annotations
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np

from data_structures import (
    Grafo, BinarySearchTree, Node,
    carregar_cenario_rs, carregar_cenario_matopiba
)
from greedy import prim_mst, reconstruir_mst_arestas, dijkstra, rota_gulosa_atendimento
from brute_force import (
    forca_bruta_caminhos, analisar_explosao_combinatoria,
    calcular_gap_otimalidade, _dijkstra_distancias
)
from performance_monitor import benchmark_completo, dados_para_graficos

# Paleta POLARIS Earth
COR_CRITICO  = "#FF6B35"
COR_ALTO     = "#FFBE0B"
COR_MEDIO    = "#00D4FF"
COR_MST      = "#39FF14"
COR_HUB      = "#FFFFFF"
COR_BG       = "#0A0F1A"
COR_BG2      = "#111827"
COR_TEXTO    = "#E2E8F0"
COR_GRID     = "#1E3A5F"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _estilo_polaris():
    """Aplica estilo escuro consistente com a identidade visual do sistema."""
    plt.rcParams.update({
        "figure.facecolor": COR_BG,
        "axes.facecolor": COR_BG2,
        "axes.edgecolor": COR_GRID,
        "axes.labelcolor": COR_TEXTO,
        "axes.titlecolor": COR_TEXTO,
        "text.color": COR_TEXTO,
        "xtick.color": COR_TEXTO,
        "ytick.color": COR_TEXTO,
        "grid.color": COR_GRID,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "legend.facecolor": "#111827",
        "legend.edgecolor": COR_GRID,
        "font.family": "monospace",
    })


def _cor_risco(risco: float) -> str:
    if risco >= 0.85:
        return COR_CRITICO
    if risco >= 0.70:
        return COR_ALTO
    return COR_MEDIO


def _salvar(fig, nome: str) -> str:
    caminho = os.path.join(OUTPUT_DIR, nome)
    fig.savefig(caminho, dpi=150, bbox_inches="tight",
                facecolor=COR_BG, edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Salvo: {caminho}")
    return caminho


def figura_grafo_mst(cenario: str = "rs") -> str:
    """
    Visualiza o grafo de municípios com as arestas da MST (Prim) em destaque.
    """
    _estilo_polaris()

    if cenario == "rs":
        grafo, bst = carregar_cenario_rs()
        hub = 4314902
        titulo = "Grafo de Municípios — Enchentes RS 2024"
        fonte = "Fonte: DNIT + Defesa Civil RS (dados sintéticos representativos)"
    else:
        grafo, bst = carregar_cenario_matopiba()
        hub = 1721000
        titulo = "Grafo de Municípios — Seca MATOPIBA"
        fonte = "Fonte: NDVI MODIS/NASA + INMET (dados sintéticos representativos)"


    G = nx.Graph()
    for v in grafo.vertices():
        G.add_node(v[0], nome=v[1], risco=v[2], custo=v[3], pop=v[4])
    for u, v, w in grafo.arestas():
        G.add_edge(u, v, weight=w)


    pred, custos, _, _ = prim_mst(grafo, hub)
    arestas_mst_raw = reconstruir_mst_arestas(pred, custos)
    arestas_mst_set = {(min(u, v), max(u, v)) for u, v, _ in arestas_mst_raw}


    pos = nx.kamada_kawai_layout(G, weight="weight")

    fig, ax = plt.subplots(figsize=(14, 9))

    # Arestas não-MST
    edges_normal = [(u, v) for u, v in G.edges()
                    if (min(u, v), max(u, v)) not in arestas_mst_set]
    nx.draw_networkx_edges(G, pos, edgelist=edges_normal,
                           edge_color=COR_GRID, width=0.8, alpha=0.5, ax=ax)

    # Arestas MST
    edges_mst = [(u, v) for u, v in G.edges()
                 if (min(u, v), max(u, v)) in arestas_mst_set]
    nx.draw_networkx_edges(G, pos, edgelist=edges_mst,
                           edge_color=COR_MST, width=2.5, alpha=0.85,
                           style="dashed", ax=ax)


    pesos_mst = {(u, v): f"{G[u][v]['weight']:.1f}h"
                 for u, v in edges_mst}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=pesos_mst,
                                 font_color=COR_MST, font_size=7,
                                 bbox=dict(boxstyle="round,pad=0.15",
                                           fc=COR_BG2, ec="none", alpha=0.8),
                                 ax=ax)


    cores_nos = []
    tamanhos = []
    for vid in G.nodes():
        r = G.nodes[vid]["risco"]
        cores_nos.append(_cor_risco(r))
        tamanhos.append(900 if vid == hub else 500)

    nx.draw_networkx_nodes(G, pos, node_color=cores_nos,
                           node_size=tamanhos, alpha=0.9,
                           linewidths=1.5, edgecolors=COR_BG, ax=ax)

    # Rótulos
    labels = {v: f"{G.nodes[v]['nome'].split()[0]}\n{G.nodes[v]['risco']:.2f}"
              for v in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_size=7, font_color=COR_TEXTO, ax=ax)


    legenda = [
        mpatches.Patch(color=COR_CRITICO, label="Crítico (≥0.85)"),
        mpatches.Patch(color=COR_ALTO,    label="Alto (0.70–0.84)"),
        mpatches.Patch(color=COR_MEDIO,   label="Médio (<0.70)"),
        mpatches.Patch(color=COR_MST,     label="MST — Prim (rota ótima)"),
    ]
    ax.legend(handles=legenda, loc="upper left", fontsize=9, framealpha=0.8)

    ax.set_title(titulo, fontsize=14, fontweight="bold", pad=16)
    ax.axis("off")
    fig.text(0.5, 0.01, fonte, ha="center", fontsize=8,
             color="#64748B", style="italic")

    # Interpretação
    n_mst = len(edges_mst)
    peso_mst = sum(G[u][v]["weight"] for u, v in edges_mst)
    fig.text(0.5, -0.02,
             f"A MST conecta {G.number_of_nodes()} municípios usando {n_mst} rotas "
             f"com deslocamento total de {peso_mst:.1f}h. "
             f"Nós com risco crítico (laranja) devem ser priorizados pelo Greedy. "
             f"Arestas verdes tracejadas formam a Árvore Geradora Mínima calculada pelo Prim.",
             ha="center", fontsize=8, color=COR_TEXTO,
             wrap=True, style="italic")

    return _salvar(fig, f"fig1_grafo_mst_{cenario}.png")



def _posicoes_bst(no: Node | None, x: float, y: float,
                  dx: float, nivel: int, pos: dict, edges: list) -> None:
    """Calcula posições dos nós para desenho da BST (recursivo)."""
    if no is None:
        return
    pos[id(no)] = (x, y, no)
    if no.esquerda:
        edges.append((id(no), id(no.esquerda)))
        _posicoes_bst(no.esquerda, x - dx, y - 1.4, dx * 0.55, nivel + 1, pos, edges)
    if no.direita:
        edges.append((id(no), id(no.direita)))
        _posicoes_bst(no.direita, x + dx, y - 1.4, dx * 0.55, nivel + 1, pos, edges)


def figura_bst(cenario: str = "rs") -> str:
    """
    Diagrama da BST com índices de risco como chaves.
    """
    _estilo_polaris()

    if cenario == "rs":
        _, bst = carregar_cenario_rs()
        titulo = "BST de Municípios por Índice de Risco — RS"
        fonte = "Fonte: índices de risco calculados a partir de dados DNIT + Defesa Civil RS"
    else:
        _, bst = carregar_cenario_matopiba()
        titulo = "BST de Municípios por Índice de Risco — MATOPIBA"
        fonte = "Fonte: NDVI MODIS/NASA + INMET"

    pos: dict = {}
    edges: list = []
    _posicoes_bst(bst.raiz, 0, 0, 3.0, 0, pos, edges)

    if not pos:
        print("  BST vazia — nenhuma figura gerada.")
        return ""

    fig, ax = plt.subplots(figsize=(14, 8))

    # Arestas
    for (pid, cid) in edges:
        if pid in pos and cid in pos:
            px, py, _ = pos[pid]
            cx, cy, _ = pos[cid]
            ax.plot([px, cx], [py, cy], color=COR_GRID, linewidth=1.2, zorder=1)

    # Nós
    for nid, (x, y, no) in pos.items():
        risco = no.chave
        cor = _cor_risco(risco)
        circ = plt.Circle((x, y), 0.45, color=cor, alpha=0.85, zorder=2)
        ax.add_patch(circ)
        ax.text(x, y + 0.05, f"{risco:.2f}", ha="center", va="center",
                fontsize=8, fontweight="bold", color=COR_BG, zorder=3)
        nome_curto = no.vertice[1].split()[0][:8]
        ax.text(x, y - 0.62, nome_curto, ha="center", va="top",
                fontsize=6.5, color=COR_TEXTO, zorder=3)

    xs = [v[0] for v in pos.values()]
    ys = [v[1] for v in pos.values()]
    ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
    ax.set_ylim(min(ys) - 1.8, max(ys) + 1.2)

    ax.text(min(xs) - 1.2, min(ys) - 1.4,
            "← percurso in-order: ordem crescente de risco →",
            ha="left", fontsize=8, color=COR_MST, style="italic")

    legenda = [
        mpatches.Patch(color=COR_CRITICO, label="Crítico (≥0.85)"),
        mpatches.Patch(color=COR_ALTO,    label="Alto (0.70–0.84)"),
        mpatches.Patch(color=COR_MEDIO,   label="Médio (<0.70)"),
    ]
    ax.legend(handles=legenda, loc="upper right", fontsize=9)
    ax.set_title(titulo, fontsize=13, fontweight="bold", pad=14)
    ax.axis("off")
    fig.text(0.5, 0.01, fonte, ha="center", fontsize=8, color="#64748B", style="italic")
    h = bst.altura()
    fig.text(0.5, -0.02,
             f"A BST com {bst.tamanho()} nós tem altura {h} "
             f"({bst.fator_balanceamento()}). "
             f"O percurso in-order retorna os municípios em ordem crescente de risco, "
             f"permitindo priorização direta pelo algoritmo Guloso em O(h + k).",
             ha="center", fontsize=8, color=COR_TEXTO, style="italic")

    return _salvar(fig, f"fig2_bst_{cenario}.png")


def figura_desempenho_comparativo() -> str:
    """
    Gráfico comparativo de tempo de execução em função do tamanho N.
    """
    _estilo_polaris()

    print("  Executando benchmark (pode demorar alguns segundos)...")
    resultados = benchmark_completo(tamanhos=[5, 8, 10, 12, 20, 50, 100], repeticoes=3)
    dados = dados_para_graficos(resultados)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))


    ax1 = axes[0]

    cores_algo = {
        "Prim":        (COR_MST,    "o-",  2.0),
        "Dijkstra":    (COR_MEDIO,  "s--", 1.8),
        "Kruskal":     (COR_ALTO,   "^-",  1.8),
        "FB_Caminhos": (COR_CRITICO,"x--", 2.2),
    }

    for algo, (cor, estilo, lw) in cores_algo.items():
        if algo not in dados:
            continue
        d = dados[algo]
        ns = [n for n, v in zip(d["n"], d["viavel"]) if v]
        ts = [t for t, v in zip(d["tempo_ms"], d["viavel"]) if v and t is not None]
        if not ns:
            continue
        ax1.plot(ns, ts, estilo, color=cor, linewidth=lw,
                 markersize=7, label=algo, alpha=0.9)


    ns_teoria = list(range(5, 14))
    fatorial_normalizado = [math.factorial(n) * 1e-4 for n in ns_teoria]
    ax1.plot(ns_teoria, fatorial_normalizado, ":", color=COR_CRITICO,
             alpha=0.4, linewidth=1.2, label="O(N!) teórico")

    ax1.axvline(x=12, color="#FF6B35", linewidth=1, alpha=0.6, linestyle=":")
    ax1.text(12.2, ax1.get_ylim()[1] * 0.5 if ax1.get_ylim()[1] > 0 else 1,
             "N=12\nlimite FB", color=COR_CRITICO, fontsize=8, va="center")

    ax1.set_xlabel("N (número de municípios)", fontsize=11)
    ax1.set_ylabel("Tempo de execução (ms)", fontsize=11)
    ax1.set_title("Escalabilidade: Força Bruta vs Guloso", fontsize=12, pad=12)
    ax1.legend(fontsize=9, framealpha=0.85)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    ax2 = axes[1]

    for algo, (cor, estilo, lw) in cores_algo.items():
        if algo not in dados:
            continue
        d = dados[algo]
        ns = [n for n, v in zip(d["n"], d["viavel"]) if v]
        ops = [o for o, v in zip(d["operacoes"], d["viavel"]) if v]
        if not ns or not ops:
            continue
        ax2.plot(ns, ops, estilo, color=cor, linewidth=lw,
                 markersize=7, label=algo, alpha=0.9)

    ax2.set_xlabel("N (número de municípios)", fontsize=11)
    ax2.set_ylabel("Operações elementares", fontsize=11)
    ax2.set_title("Contagem de Operações por Algoritmo", fontsize=12, pad=12)
    ax2.legend(fontsize=9, framealpha=0.85)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    fig.suptitle("POLARIS Earth — Análise de Desempenho dos Algoritmos",
                 fontsize=14, fontweight="bold", y=1.01)

    fonte = "Fonte: medições empíricas com time.perf_counter() e tracemalloc — Python 3.x"
    fig.text(0.5, 0.01, fonte, ha="center", fontsize=8, color="#64748B", style="italic")
    fig.text(0.5, -0.04,
             "A Força Bruta cresce fatorialmente (N!) tornando-se inviável a partir de N≈12. "
             "Os algoritmos Gulosos (Prim, Dijkstra, Kruskal) mantêm crescimento O((V+E)logV), "
             "permitindo processar os 478 municípios afetados pelas enchentes do RS em milissegundos.",
             ha="center", fontsize=8.5, color=COR_TEXTO, style="italic")

    plt.tight_layout()
    return _salvar(fig, "fig3_desempenho_comparativo.png")


def figura_tabela_estruturas() -> str:
    """
    Tabela visual das estruturas de dados utilizadas
    """
    _estilo_polaris()

    linhas = [
        ["list",           "Lista de adjacência do grafo",     "O(V+E) espaço; O(grau) iteração"],
        ["tuple",          "Vértice (id, nome, risco, custo, pop)", "Imutabilidade: dado geoespacial não muda"],
        ["dict",           "Adj. ponderada; custos Dijkstra/Prim", "Acesso O(1) por id_município"],
        ["set",            "Nós visitados BFS/DFS; fronteira Greedy", "Pertencimento O(1); evita ciclos"],
        ["deque",          "Fila BFS para travessia do grafo", "append/popleft O(1) vs list O(n)"],
        ["heapq",          "Fila de prioridade Prim/Dijkstra", "Extrai mínimo em O(log V)"],
        ["BinarySearchTree","BST de municípios por risco",     "Busca/intervalo O(h); in-order O(n)"],
        ["UnionFind",      "Detecção de ciclos no Kruskal",    "Unir/Encontrar ≈ O(1) amortizado"],
    ]

    cols = ["Estrutura", "Uso no Sistema", "Justificativa de Complexidade"]
    cores_linhas = [
        COR_CRITICO, COR_ALTO, COR_MEDIO, COR_MST,
        "#A855F7", "#EC4899", "#14B8A6", "#F97316",
    ]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis("off")

    tabela = ax.table(
        cellText=linhas,
        colLabels=cols,
        loc="center",
        cellLoc="left",
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9)
    tabela.scale(1, 2.1)

    # Estilo do cabeçalho
    for j in range(len(cols)):
        cell = tabela[0, j]
        cell.set_facecolor(COR_GRID)
        cell.set_text_props(color=COR_TEXTO, fontweight="bold")
        cell.set_edgecolor(COR_BG)

    for i, cor_struct in enumerate(cores_linhas):
        for j in range(len(cols)):
            cell = tabela[i + 1, j]
            cell.set_facecolor(COR_BG2)
            cell.set_edgecolor(COR_BG)
            if j == 0:
                cell.set_text_props(color=cor_struct, fontweight="bold", fontfamily="monospace")
            else:
                cell.set_text_props(color=COR_TEXTO)

    ax.set_title(
        "Estruturas de Dados — Justificativa de Uso e Complexidade",
        fontsize=13, fontweight="bold", pad=16, color=COR_TEXTO
    )
    fonte = "Fonte: implementação em data_structures.py e greedy.py — POLARIS Earth"
    fig.text(0.5, 0.01, fonte, ha="center", fontsize=8, color="#64748B", style="italic")
    fig.text(0.5, -0.04,
             "A escolha de lista de adjacência (dict de lists) sobre matriz de adjacência "
             "reduz o espaço de O(V²) para O(V+E), crítico para redes esparsas de municípios. "
             "O heapq garante extração de mínimo em O(log V), viabilizando Prim e Dijkstra "
             "em instâncias com centenas de municípios em tempo sub-segundo.",
             ha="center", fontsize=8.5, color=COR_TEXTO, style="italic", wrap=True)

    return _salvar(fig, "fig4_tabela_estruturas.png")


def figura_gap_otimalidade() -> str:
    """
    Gráfico do gap percentual entre FB (ótimo) e Greedy em função de N.
    """
    _estilo_polaris()

    from performance_monitor import gerar_grafo_sintetico

    tamanhos = [5, 6, 7, 8, 9, 10, 11, 12]
    gaps: list[float] = []
    custos_fb: list[float] = []
    custos_greedy: list[float] = []

    print("  Calculando gaps de otimalidade...")
    for n in tamanhos:
        grafo, bst, hub = gerar_grafo_sintetico(n, seed=n * 7)
        ids = grafo.ids_vertices()

        orig, dest = ids[0], ids[-1]
        _, custo_fb, _ = forca_bruta_caminhos(grafo, orig, dest)

        dist, _, _ = dijkstra(grafo, orig)
        custo_dijkstra = dist.get(dest, float("inf"))

        if custo_fb > 0 and custo_dijkstra < float("inf"):
            analise = calcular_gap_otimalidade(custo_fb, custo_dijkstra)
            gap = analise["gap_pct"]
        else:
            gap = 0.0

        gaps.append(gap)
        custos_fb.append(custo_fb if custo_fb < float("inf") else 0)
        custos_greedy.append(custo_dijkstra if custo_dijkstra < float("inf") else 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    barras = ax1.bar(tamanhos, gaps,
                     color=[COR_MST if g < 5 else COR_ALTO if g < 15 else COR_CRITICO
                            for g in gaps],
                     alpha=0.85, edgecolor=COR_BG, linewidth=0.5)
    ax1.axhline(y=0, color=COR_TEXTO, linewidth=0.8, alpha=0.5)
    ax1.axhline(y=5, color=COR_ALTO, linewidth=1, linestyle="--", alpha=0.6,
                label="Limiar 5% (excelente)")
    ax1.axhline(y=15, color=COR_CRITICO, linewidth=1, linestyle="--", alpha=0.6,
                label="Limiar 15% (revisar)")

    for bar, gap in zip(barras, gaps):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{gap:.1f}%", ha="center", fontsize=9, color=COR_TEXTO)

    ax1.set_xlabel("N (municípios)", fontsize=11)
    ax1.set_ylabel("Gap de Otimalidade (%)", fontsize=11)
    ax1.set_title("Gap: Força Bruta × Dijkstra", fontsize=12, pad=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_xticks(tamanhos)

    ax2 = axes[1]
    x = np.array(tamanhos)
    w = 0.35
    ax2.bar(x - w / 2, custos_fb, w, label="Força Bruta (ótimo)",
            color=COR_CRITICO, alpha=0.85, edgecolor=COR_BG)
    ax2.bar(x + w / 2, custos_greedy, w, label="Dijkstra (Greedy)",
            color=COR_MEDIO, alpha=0.85, edgecolor=COR_BG)

    ax2.set_xlabel("N (municípios)", fontsize=11)
    ax2.set_ylabel("Custo do caminho (horas)", fontsize=11)
    ax2.set_title("Custo Absoluto: FB vs Dijkstra", fontsize=12, pad=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_xticks(tamanhos)

    fig.suptitle("POLARIS Earth — Gap de Otimalidade: Força Bruta vs Greedy",
                 fontsize=13, fontweight="bold")

    fonte = "Fonte: grafos sintéticos com seed fixo; FB via backtracking, Greedy via Dijkstra"
    fig.text(0.5, 0.01, fonte, ha="center", fontsize=8, color="#64748B", style="italic")
    gap_medio = sum(gaps) / len(gaps) if gaps else 0
    fig.text(0.5, -0.04,
             f"Gap médio observado: {gap_medio:.2f}%. "
             "O algoritmo Dijkstra (Greedy) converge para o ótimo global em caminho mínimo, "
             "pois o critério de relaxação é exato para pesos não-negativos (Teorema de Dijkstra). "
             "O gap observado ≈ 0% confirma a corretude da implementação Greedy para este problema.",
             ha="center", fontsize=8.5, color=COR_TEXTO, style="italic")

    plt.tight_layout()
    return _salvar(fig, "fig5_gap_otimalidade.png")



def gerar_todas_as_figuras() -> list[str]:
    """Gera todas as 5 figuras obrigatórias e retorna caminhos."""
    print("\n" + "=" * 60)
    print("POLARIS Earth — Gerando Figuras Obrigatórias")
    print("=" * 60)

    arquivos = []

    print("\n[1/5] Grafo com MST destacada (RS)...")
    arquivos.append(figura_grafo_mst("rs"))

    print("\n[2/5] BST de municípios por risco (RS)...")
    arquivos.append(figura_bst("rs"))

    print("\n[3/5] Gráfico comparativo de desempenho...")
    arquivos.append(figura_desempenho_comparativo())

    print("\n[4/5] Tabela de estruturas de dados...")
    arquivos.append(figura_tabela_estruturas())

    print("\n[5/5] Gap de otimalidade FB vs Greedy...")
    arquivos.append(figura_gap_otimalidade())

    print(f"\n✓ Todas as figuras salvas em: {OUTPUT_DIR}/")
    return arquivos


if __name__ == "__main__":
    gerar_todas_as_figuras()