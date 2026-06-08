from __future__ import annotations
import time
import tracemalloc
import random
import math
from dataclasses import dataclass, field
from data_structures import Grafo, BinarySearchTree, criar_vertice, criar_aresta
from brute_force import forca_bruta_caminhos, analisar_explosao_combinatoria
from greedy import prim_mst, dijkstra, kruskal_mst



@dataclass
class ResultadoMedicao:
    algoritmo: str
    n_vertices: int
    tempo_ms: float
    memoria_kb: float
    operacoes: int
    peso_solucao: float
    viavel: bool = True
    observacao: str = ""

    def __str__(self) -> str:
        status = "OK" if self.viavel else "INVIÁVEL"
        return (
            f"[{self.algoritmo:12s}] N={self.n_vertices:4d} | "
            f"tempo={self.tempo_ms:9.3f}ms | "
            f"mem={self.memoria_kb:7.1f}KB | "
            f"ops={self.operacoes:8,} | "
            f"peso={self.peso_solucao:7.3f} | {status}"
        )


def gerar_grafo_sintetico(n: int, seed: int = 42,
                           densidade: float = 0.4) -> tuple[Grafo, BinarySearchTree, int]:
    """
    Gera grafo conexo aleatório com N municípios sintéticos para benchmarking.
    """
    random.seed(seed)
    grafo = Grafo()
    bst = BinarySearchTree()
    ids = list(range(1000, 1000 + n))

    # Gera vértices
    nomes = [f"Município_{i}" for i in range(n)]
    for i, vid in enumerate(ids):
        risco = round(random.uniform(0.3, 0.99), 4)
        custo = round(random.uniform(100, 2000), 0)
        pop = random.randint(5000, 500000)
        v = criar_vertice(vid, nomes[i], risco, custo, pop)
        grafo.adicionar_vertice(v)
        bst.inserir(v)

    # Garante conectividade: cadeia
    for i in range(n - 1):
        peso = round(random.uniform(0.2, 5.0), 2)
        grafo.adicionar_aresta(ids[i], ids[i + 1], peso)

    # Arestas extras
    max_extras = int(n * (n - 1) / 2 * densidade)
    tentativas = 0
    extras = 0
    while extras < max_extras and tentativas < max_extras * 5:
        u = random.choice(ids)
        v = random.choice(ids)
        if u != v:
            peso = round(random.uniform(0.2, 5.0), 2)
            try:
                grafo.adicionar_aresta(u, v, peso)
                extras += 1
            except Exception:
                pass
        tentativas += 1

    return grafo, bst, ids[0]


def medir(algoritmo_fn, *args, nome: str = "algo", n: int = 0,
          timeout_s: float = 30.0) -> ResultadoMedicao:
    """
    Executa função e mede tempo (perf_counter) e memória (tracemalloc).
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    try:
        resultado = algoritmo_fn(*args)
    except RecursionError:
        tracemalloc.stop()
        return ResultadoMedicao(
            algoritmo=nome, n_vertices=n,
            tempo_ms=float("inf"), memoria_kb=0,
            operacoes=0, peso_solucao=0,
            viavel=False, observacao="RecursionError: N muito grande para FB"
        )

    t1 = time.perf_counter()
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tempo_ms = (t1 - t0) * 1000
    memoria_kb = pico / 1024

    # Extrai métricas do resultado conforme o algoritmo
    operacoes = 0
    peso = 0.0

    if nome.startswith("Prim"):
        _, custo_mst, operacoes, peso = resultado
    elif nome.startswith("Dijkstra"):
        _, _, operacoes = resultado
    elif nome.startswith("Kruskal"):
        _, peso, operacoes = resultado
    elif nome.startswith("FB_Caminhos"):
        _, custo, todos = resultado
        peso = custo if custo < float("inf") else 0
        operacoes = len(todos)
    elif nome.startswith("FB_Cobertura"):
        _, soma, _ = resultado
        peso = soma

    viavel = tempo_ms < timeout_s * 1000

    return ResultadoMedicao(
        algoritmo=nome, n_vertices=n,
        tempo_ms=round(tempo_ms, 4),
        memoria_kb=round(memoria_kb, 2),
        operacoes=operacoes,
        peso_solucao=round(peso, 4),
        viavel=viavel,
    )


def benchmark_completo(tamanhos: list[int] | None = None,
                        repeticoes: int = 3) -> list[ResultadoMedicao]:
    """
    Executa benchmark para todos os tamanhos e algoritmos.
    """
    if tamanhos is None:
        tamanhos = [5, 8, 10, 12, 20, 50, 100]

    resultados: list[ResultadoMedicao] = []
    LIMITE_FB = 12

    print(f"\n{'─' * 80}")
    print(f"{'ALGORITMO':12s} {'N':>5} {'TEMPO (ms)':>12} {'MEM (KB)':>10} "
          f"{'OPERAÇÕES':>12} {'VIÁVEL':>8}")
    print(f"{'─' * 80}")

    for n in tamanhos:
        grafo, bst, hub = gerar_grafo_sintetico(n, seed=42 + n)


        medicoes_prim = []
        for _ in range(repeticoes):
            m = medir(prim_mst, grafo, hub, nome="Prim", n=n)
            medicoes_prim.append(m)
        med_prim = _mediana(medicoes_prim)
        resultados.append(med_prim)
        print(med_prim)


        medicoes_dijk = []
        for _ in range(repeticoes):
            m = medir(dijkstra, grafo, hub, nome="Dijkstra", n=n)
            medicoes_dijk.append(m)
        med_dijk = _mediana(medicoes_dijk)
        resultados.append(med_dijk)
        print(med_dijk)


        medicoes_kru = []
        for _ in range(repeticoes):
            m = medir(kruskal_mst, grafo, nome="Kruskal", n=n)
            medicoes_kru.append(m)
        med_kru = _mediana(medicoes_kru)
        resultados.append(med_kru)
        print(med_kru)


        if n <= LIMITE_FB:
            ids = grafo.ids_vertices()
            if len(ids) >= 2:
                orig, dest = ids[0], ids[-1]
                medicoes_fb = []
                for _ in range(repeticoes):
                    m = medir(forca_bruta_caminhos, grafo, orig, dest,
                              nome="FB_Caminhos", n=n, timeout_s=30.0)
                    medicoes_fb.append(m)
                med_fb = _mediana(medicoes_fb)
                resultados.append(med_fb)
                print(med_fb)
        else:
            print(f"{'FB_Caminhos':12s} N={n:4d} | PULADO — N > {LIMITE_FB} (inviável)")

        print()

    return resultados


def _mediana(medicoes: list[ResultadoMedicao]) -> ResultadoMedicao:
    """Retorna a medição com tempo mediano (reduz ruído de JIT/GC)."""
    medicoes_ord = sorted(medicoes, key=lambda m: m.tempo_ms)
    return medicoes_ord[len(medicoes_ord) // 2]


def dados_para_graficos(resultados: list[ResultadoMedicao]) -> dict:
    """
    Organiza resultados em estrutura pronta para matplotlib.
    """
    algos = {}
    for r in resultados:
        if r.algoritmo not in algos:
            algos[r.algoritmo] = {"n": [], "tempo_ms": [], "memoria_kb": [],
                                   "operacoes": [], "viavel": []}
        algos[r.algoritmo]["n"].append(r.n_vertices)
        algos[r.algoritmo]["tempo_ms"].append(r.tempo_ms if r.viavel else None)
        algos[r.algoritmo]["memoria_kb"].append(r.memoria_kb)
        algos[r.algoritmo]["operacoes"].append(r.operacoes)
        algos[r.algoritmo]["viavel"].append(r.viavel)
    return algos


def resumo_escalabilidade(resultados: list[ResultadoMedicao]) -> None:
    """Imprime tabela resumo e identifica cruzamento FB × Greedy."""
    dados = dados_para_graficos(resultados)

    print("\n" + "=" * 60)
    print("RESUMO DE ESCALABILIDADE")
    print("=" * 60)

    for algo, d in dados.items():
        ns_viaveis = [n for n, v in zip(d["n"], d["viavel"]) if v]
        tempos = [t for t in d["tempo_ms"] if t is not None]
        if not tempos:
            print(f"  {algo}: sem medições viáveis")
            continue
        print(f"\n  {algo}:")
        print(f"    N viáveis:     {ns_viaveis}")
        print(f"    Tempo mín:     {min(tempos):.4f}ms")
        print(f"    Tempo máx:     {max(tempos):.4f}ms")
        if len(tempos) >= 2:
            fator = tempos[-1] / tempos[0] if tempos[0] > 0 else float("inf")
            print(f"    Fator crescim: {fator:.1f}×")

    # Identifica onde FB se torna inviável
    fb_data = dados.get("FB_Caminhos", {})
    if fb_data:
        ultimo_viavel = max(
            (n for n, v in zip(fb_data["n"], fb_data["viavel"]) if v),
            default=None
        )
        print(f"\n  → Força Bruta torna-se inviável a partir de N ≈ {ultimo_viavel or 12}+")
        print(f"    (explosão combinatória: N! chamadas recursivas)")


if __name__ == "__main__":
    print("=" * 60)
    print("POLARIS Earth — Monitor de Desempenho")
    print("=" * 60)

    resultados = benchmark_completo(
        tamanhos=[5, 8, 10, 12, 20, 50, 100],
        repeticoes=3
    )

    resumo_escalabilidade(resultados)

    from brute_force import imprimir_tabela_explosao
    imprimir_tabela_explosao()

    dados = dados_para_graficos(resultados)
    print(f"\nDados exportados para visualizations.py: {list(dados.keys())}")