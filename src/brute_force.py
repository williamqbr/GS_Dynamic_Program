from __future__ import annotations
from data_structures import Grafo, BinarySearchTree, carregar_cenario_rs
import itertools


class Contadores:
    """Registra métricas de execução da Força Bruta."""

    def __init__(self):
        self.chamadas_recursivas: int = 0
        self.caminhos_avaliados: int = 0
        self.subconjuntos_avaliados: int = 0
        self.melhor_custo: float = float("inf")
        self.melhor_solucao: list = []

    def resetar(self):
        self.__init__()

    def __repr__(self):
        return (
            f"Contadores(\n"
            f"  chamadas_recursivas    = {self.chamadas_recursivas:,}\n"
            f"  caminhos_avaliados     = {self.caminhos_avaliados:,}\n"
            f"  subconjuntos_avaliados = {self.subconjuntos_avaliados:,}\n"
            f"  melhor_custo           = {self.melhor_custo:.4f}\n"
            f")"
        )


_cnt = Contadores()


def _fb_caminhos_rec(grafo: Grafo, atual: int, destino: int,
                     visitados: set[int], caminho: list[int],
                     custo_atual: float, todos_caminhos: list) -> None:
    """
    Backtracking recursivo: gera todos os caminhos simples de 'atual' a 'destino'.
    """
    _cnt.chamadas_recursivas += 1

    if atual == destino:
        _cnt.caminhos_avaliados += 1
        todos_caminhos.append((list(caminho), custo_atual))
        if custo_atual < _cnt.melhor_custo:
            _cnt.melhor_custo = custo_atual
            _cnt.melhor_solucao = list(caminho)
        return

    for vizinho, peso in grafo.vizinhos(atual):
        if vizinho not in visitados:          # evita ciclos
            visitados.add(vizinho)
            caminho.append(vizinho)
            _fb_caminhos_rec(grafo, vizinho, destino, visitados,
                             caminho, custo_atual + peso, todos_caminhos)
            caminho.pop()
            visitados.discard(vizinho)


def forca_bruta_caminhos(grafo: Grafo, origem: int, destino: int
                          ) -> tuple[list[int], float, list[tuple]]:
    """
    Encontra TODOS os caminhos entre origem e destino por enumeração completa.
    """
    _cnt.resetar()

    visitados: set[int] = {origem}
    caminho: list[int] = [origem]
    todos_caminhos: list[tuple] = []

    _fb_caminhos_rec(grafo, origem, destino, visitados, caminho, 0.0, todos_caminhos)

    # Ordena por custo para facilitar análise
    todos_caminhos.sort(key=lambda x: x[1])

    return _cnt.melhor_solucao, _cnt.melhor_custo, todos_caminhos


def forca_bruta_cobertura(grafo: Grafo, bst: BinarySearchTree,
                           hub: int, orcamento_horas: float,
                           limiar_risco: float = 0.0
                           ) -> tuple[list[int], float, float]:
    """
    Enumeração exaustiva de subconjuntos de municípios para maximizar
    a soma de riscos atendidos dentro de um orçamento de horas.
    """
    # Candidatos: municípios com risco acima do limiar (via BST)
    candidatos_vertices = bst.buscar_intervalo(limiar_risco, 1.0)
    candidatos_ids = [v[0] for v in candidatos_vertices if v[0] != hub]

    # Pré-calcula distâncias mínimas do hub (Dijkstra simples)
    dist_hub = _dijkstra_distancias(grafo, hub)

    melhor_subconjunto: list[int] = []
    melhor_soma_risco: float = 0.0
    melhor_custo: float = 0.0

    _cnt.resetar()

    # Enumera todos os 2^N subconjuntos
    n = len(candidatos_ids)
    for r in range(1, n + 1):
        for subconj in itertools.combinations(candidatos_ids, r):
            _cnt.subconjuntos_avaliados += 1

            custo = sum(dist_hub.get(v, float("inf")) for v in subconj)

            if custo <= orcamento_horas:
                soma_risco = sum(
                    grafo.obter_vertice(v)[2]
                    for v in subconj
                    if grafo.obter_vertice(v) is not None
                )
                if soma_risco > melhor_soma_risco:
                    melhor_soma_risco = soma_risco
                    melhor_subconjunto = list(subconj)
                    melhor_custo = custo

    return melhor_subconjunto, melhor_soma_risco, melhor_custo


def _dijkstra_distancias(grafo: Grafo, origem: int) -> dict[int, float]:
    """
    Dijkstra para calcular distâncias mínimas do hub a todos os vértices.
    """
    import heapq
    dist: dict[int, float] = {v: float("inf") for v in grafo.ids_vertices()}
    dist[origem] = 0.0
    heap = [(0.0, origem)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in grafo.vizinhos(u):
            nd = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return dist


def analisar_explosao_combinatoria(tamanhos_n: list[int] | None = None
                                    ) -> list[dict]:
    """
    Calcula o número teórico de operações da Força Bruta em função de N.
    Retorna lista de dicts com métricas por N.
    """
    import math

    if tamanhos_n is None:
        tamanhos_n = [5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 20, 25, 30]

    resultados = []
    for n in tamanhos_n:
        fatorial = math.factorial(n)
        subconjuntos = 2 ** n
        # Estimativa de tempo: ~10^-7 s por operação elementar (Python moderno)
        tempo_fb_caminhos_s = fatorial * 1e-7
        tempo_fb_subconj_s = subconjuntos * 1e-7

        resultados.append({
            "n": n,
            "fat_n": fatorial,
            "2_n": subconjuntos,
            "tempo_caminhos_s": tempo_fb_caminhos_s,
            "tempo_subconj_s": tempo_fb_subconj_s,
            "viavel_caminhos": tempo_fb_caminhos_s < 60,     # < 1 min
            "viavel_subconj": tempo_fb_subconj_s < 60,
        })

    return resultados

def imprimir_tabela_explosao() -> None:
    """Imprime tabela formatada da explosão combinatória."""
    dados = analisar_explosao_combinatoria()
    print("\n" + "=" * 75)
    print(f"{'N':>4} | {'N! (caminhos)':>18} | {'2^N (subconj)':>14} | "
          f"{'T caminhos':>12} | {'T subconj':>11} | {'Viável?':>7}")
    print("-" * 75)
    for d in dados:
        tc = d["tempo_caminhos_s"]
        ts = d["tempo_subconj_s"]
        tc_str = f"{tc:.2e}s" if tc < 3600 else ">1h"
        ts_str = f"{ts:.2e}s" if ts < 3600 else ">1h"
        viavel = "✓" if d["viavel_subconj"] else "✗ INVIÁVEL"
        print(f"{d['n']:>4} | {d['fat_n']:>18,} | {d['2_n']:>14,} | "
              f"{tc_str:>12} | {ts_str:>11} | {viavel:>9}")
    print("=" * 75)
    print("→ Força Bruta torna-se inviável empiricamente a partir de N ≈ 20")
    print("  (tempo > 1 min para subconjuntos; > 1h para N! de caminhos)")


def calcular_gap_otimalidade(custo_fb: float, custo_greedy: float) -> dict:
    """
    Calcula o gap percentual entre a solução ótima e a Greedy.
    """
    if custo_fb == 0:
        return {"gap_pct": 0.0, "interpretacao": "Ambos retornaram custo zero"}
    gap = (custo_greedy - custo_fb) / custo_fb * 100
    if abs(gap) < 0.01:
        interpretacao = "Greedy encontrou solução ÓTIMA (gap ≈ 0%)"
    elif abs(gap) < 5:
        interpretacao = f"Greedy ficou {gap:+.2f}% do ótimo — EXCELENTE"
    elif abs(gap) < 15:
        interpretacao = f"Greedy ficou {gap:+.2f}% do ótimo — BOM"
    else:
        interpretacao = f"Greedy ficou {gap:+.2f}% do ótimo — REVISAR CRITÉRIO LOCAL"
    return {
        "custo_fb": custo_fb,
        "custo_greedy": custo_greedy,
        "gap_pct": round(gap, 4),
        "interpretacao": interpretacao,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("POLARIS Earth — Força Bruta")
    print("=" * 60)

    grafo, bst = carregar_cenario_rs()

    # ── Teste 1: todos os caminhos Porto Alegre → Pelotas
    origem = 4314902   # Porto Alegre
    destino = 4313409  # Pelotas

    print(f"\n1. Todos os caminhos: Porto Alegre → Pelotas")
    melhor, custo, todos = forca_bruta_caminhos(grafo, origem, destino)

    nomes = []
    for vid in melhor:
        v = grafo.obter_vertice(vid)
        nomes.append(v[1] if v else str(vid))

    print(f"   Melhor caminho: {' → '.join(nomes)}")
    print(f"   Custo ótimo:    {custo:.2f}h")
    print(f"   Caminhos encontrados: {len(todos)}")
    print(f"\n{_cnt}")

    # ── Teste 2: cobertura por subconjunto
    print("\n2. Cobertura por subconjunto (orçamento = 5h, limiar_risco=0.80)")
    subconj, soma_risco, custo_total = forca_bruta_cobertura(
        grafo, bst, hub=origem, orcamento_horas=5.0, limiar_risco=0.80
    )
    nomes_subconj = [grafo.obter_vertice(v)[1] for v in subconj]
    print(f"   Municípios: {nomes_subconj}")
    print(f"   Soma de risco: {soma_risco:.4f}")
    print(f"   Custo total:   {custo_total:.2f}h")
    print(f"   Subconjuntos avaliados: {_cnt.subconjuntos_avaliados:,}")

    imprimir_tabela_explosao()