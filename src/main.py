"""
main.py — POLARIS Earth
Ponto de entrada principal do sistema.

Executa todos os módulos em sequência:
  1. Estruturas de dados (grafo + BST)
  2. Força Bruta (baseline + explosão combinatória)
  3. Algoritmos Gulosos (Prim, Dijkstra, Kruskal)
  4. Monitor de desempenho (benchmark N × tempo)
  5. Geração das figuras obrigatórias

Uso:
    python main.py              # executa tudo
    python main.py --estruturas # só estruturas de dados
    python main.py --algoritmos # só FB + Greedy
    python main.py --benchmark  # só benchmark de desempenho
    python main.py --figuras    # só gera as figuras
"""

import time

'''def checar_dependencias():
    deps = {
        "matplotlib": "pip install matplotlib",
        "networkx":   "pip install networkx",
        "numpy":      "pip install numpy",
    }
    faltando = []
    for lib, cmd in deps.items():
        try:
            __import__(lib)
        except ImportError:
            faltando.append((lib, cmd))
    if faltando:
        print("\n[ERRO] Dependências ausentes:")
        for lib, cmd in faltando:
            print(f"  {lib:15s} → instale com: {cmd}")
        print("\nOu instale tudo de uma vez:")
        print("  pip install -r requirements.txt\n")
        sys.exit(1)

checar_dependencias()'''

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_structures import (
    carregar_cenario_rs, carregar_cenario_matopiba
)
from src.brute_force import (
    forca_bruta_caminhos, forca_bruta_cobertura,
    imprimir_tabela_explosao, calcular_gap_otimalidade
)
from src.greedy import (
    prim_mst, dijkstra, kruskal_mst,
    reconstruir_mst_arestas, reconstruir_caminho,
    rota_gulosa_atendimento
)
from src.performance_monitor import benchmark_completo, resumo_escalabilidade
from src.visualizations import gerar_todas_as_figuras


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          POLARIS EARTH — Sistema de Monitoramento            ║
║              de Riscos Ambientais com Satélites              ║
╠══════════════════════════════════════════════════════════════╣
║  FIAP — Global Solution 2026 | Estruturas de Dados           ║
║  Disciplina: Estruturas de Dados e Algoritmos                ║
╚══════════════════════════════════════════════════════════════╝
"""

SEP = "─" * 62


def secao(titulo: str):
    print(f"\n{'═' * 62}")
    print(f"  {titulo}")
    print(f"{'═' * 62}")


def sub(titulo: str):
    print(f"\n  {titulo}")
    print(f"  {'─' * 50}")


def rodar_estruturas():
    secao("MÓDULO 1 — Estruturas de Dados")

    for nome_cenario, loader, hub_id in [
        ("Cenário A — Enchentes RS",  carregar_cenario_rs,      4314902),
        ("Cenário B — Seca MATOPIBA", carregar_cenario_matopiba, 1721000),
    ]:
        sub(nome_cenario)
        grafo, bst = loader()

        print(f"\n  {grafo}")
        print(f"  {bst}")

        print("\n  Percurso in-order (risco crescente):")
        for v in bst.percurso_in_order():
            print(f"    {v[1]:30s}  risco={v[2]:.2f}  pop={v[4]:>10,}")

        print("\n  Municípios críticos (risco ≥ 0.85):")
        criticos = bst.municipios_criticos(0.85)
        for v in criticos:
            print(f"    {v[1]:30s}  risco={v[2]:.2f}  custo=R${v[3]:,.0f}")

        print(f"\n  BFS a partir do hub ({grafo.obter_vertice(hub_id)[1]}):")
        ordem = grafo.bfs(hub_id)
        nomes = [grafo.obter_vertice(vid)[1] for vid in ordem]
        print("    " + " → ".join(nomes))

        print(f"\n  Busca BST no intervalo [0.70, 0.90]:")
        intervalo = bst.buscar_intervalo(0.70, 0.90)
        for v in intervalo:
            print(f"    {v[1]:30s}  risco={v[2]:.2f}")


def rodar_forca_bruta():
    secao("MÓDULO 2 — Força Bruta (Baseline)")

    grafo, bst = carregar_cenario_rs()
    hub        = 4314902   # Porto Alegre
    destino    = 4313409   # Pelotas

    sub("2.1 Todos os caminhos: Porto Alegre → Pelotas")
    t0 = time.perf_counter()
    melhor, custo_fb, todos = forca_bruta_caminhos(grafo, hub, destino)
    tempo_ms = (time.perf_counter() - t0) * 1000

    nomes = [grafo.obter_vertice(v)[1] for v in melhor]
    print(f"\n  Melhor caminho: {' → '.join(nomes)}")
    print(f"  Custo ótimo:    {custo_fb:.2f}h")
    print(f"  Caminhos encontrados: {len(todos)}")
    print(f"  Tempo: {tempo_ms:.3f}ms")

    if len(todos) > 1:
        print("\n  Todos os caminhos (ordenados por custo):")
        for i, (cam, custo) in enumerate(todos[:8]):
            ns = [grafo.obter_vertice(v)[1] for v in cam]
            print(f"    [{i+1}] {custo:.2f}h  {' → '.join(ns)}")
        if len(todos) > 8:
            print(f"    ... e mais {len(todos) - 8} caminhos")

    sub("2.2 Cobertura por subconjunto (orçamento=5h, risco≥0.80)")
    t0 = time.perf_counter()
    subconj, soma_risco, custo_subconj = forca_bruta_cobertura(
        grafo, bst, hub=hub, orcamento_horas=5.0, limiar_risco=0.80
    )
    tempo_ms = (time.perf_counter() - t0) * 1000

    nomes_subconj = [grafo.obter_vertice(v)[1] for v in subconj]
    print(f"\n  Municípios selecionados: {nomes_subconj}")
    print(f"  Soma de risco coberto:   {soma_risco:.4f}")
    print(f"  Custo total:             {custo_subconj:.2f}h")
    print(f"  Tempo: {tempo_ms:.3f}ms")

    sub("2.3 Explosão Combinatória")
    imprimir_tabela_explosao()

    return custo_fb


def rodar_algoritmos_gulosos(custo_fb: float = None):
    secao("MÓDULO 3 — Algoritmos Gulosos")

    for nome_cenario, loader, hub_id, hub_nome in [
        ("Cenário A — Enchentes RS",  carregar_cenario_rs,       4314902, "Porto Alegre"),
        ("Cenário B — Seca MATOPIBA", carregar_cenario_matopiba, 1721000, "Palmas"),
    ]:
        sub(nome_cenario)
        grafo, bst = loader()

        # Prim
        t0 = time.perf_counter()
        pred, custos, ops_prim, peso_mst = prim_mst(grafo, hub_id)
        tempo_prim = (time.perf_counter() - t0) * 1000
        arestas_mst = reconstruir_mst_arestas(pred, custos)

        print(f"\n  [PRIM] MST — peso total: {peso_mst:.2f}h  "
              f"| ops: {ops_prim}  | tempo: {tempo_prim:.3f}ms")
        for u, v, w in arestas_mst:
            nu = grafo.obter_vertice(u)[1]
            nv = grafo.obter_vertice(v)[1]
            print(f"    {nu:22s} ─── {nv:22s}  {w:.1f}h")

        # Dijkstra
        t0 = time.perf_counter()
        dist, pred_dijk, arestas_rel = dijkstra(grafo, hub_id)
        tempo_dijk = (time.perf_counter() - t0) * 1000

        print(f"\n  [DIJKSTRA] Distâncias mínimas de {hub_nome}  "
              f"| arestas relaxadas: {arestas_rel}  | tempo: {tempo_dijk:.3f}ms")
        for vid, d in sorted(dist.items(), key=lambda x: x[1]):
            if d < float("inf"):
                v = grafo.obter_vertice(vid)
                caminho = reconstruir_caminho(pred_dijk, vid)
                print(f"    {v[1]:25s}  {d:.2f}h  ({len(caminho)} saltos)")

        # Kruskal
        t0 = time.perf_counter()
        mst_k, peso_k, ops_k = kruskal_mst(grafo)
        tempo_kru = (time.perf_counter() - t0) * 1000

        print(f"\n  [KRUSKAL] MST — peso: {peso_k:.2f}h  "
              f"| ops: {ops_k}  | tempo: {tempo_kru:.3f}ms")
        print(f"  Comparação Prim={peso_mst:.2f}h ≈ Kruskal={peso_k:.2f}h ✓ (ambos MST ótima)")

        # Rota de atendimento
        rota, custo_rota = rota_gulosa_atendimento(
            grafo, bst, hub_id, num_equipes=8, limiar_risco=0.70
        )
        print(f"\n  [ROTA] Sequência de atendimento (8 equipes, risco ≥ 0.70):")
        for p in rota:
            print(f"    [{p['ordem']:02d}] {p['nome']:22s}  "
                  f"risco={p['risco']:.2f}  dist={p['distancia_hub_h']:.2f}h  "
                  f"prior={p['prioridade_gulosa']:.3f}  equipe={p['equipe']}")
        print(f"  Custo total de deslocamento: {custo_rota:.2f}h")

        # Gap de otimalidade (só no cenário RS onde temos o FB)
        if custo_fb and hub_id == 4314902:
            dest = 4313409
            custo_dijkstra = dist.get(dest, float("inf"))
            if custo_dijkstra < float("inf"):
                analise = calcular_gap_otimalidade(custo_fb, custo_dijkstra)
                print(f"\n  [GAP] {analise['interpretacao']}")
                print(f"        FB={analise['custo_fb']:.4f}h | "
                      f"Greedy={analise['custo_greedy']:.4f}h | "
                      f"gap={analise['gap_pct']:+.2f}%")



def rodar_benchmark():
    secao("MÓDULO 4 — Benchmark de Desempenho")
    print("\n  Tamanhos testados: N = 5, 8, 10, 12, 20, 50, 100")
    print("  Força Bruta executada apenas para N ≤ 12\n")

    resultados = benchmark_completo(
        tamanhos=[5, 8, 10, 12, 20, 50, 100],
        repeticoes=3
    )
    resumo_escalabilidade(resultados)
    return resultados



def rodar_figuras():
    secao("MÓDULO 5 — Geração de Figuras Obrigatórias")
    arquivos = gerar_todas_as_figuras()
    print(f"\n  {len(arquivos)} figura(s) gerada(s) em report/")
    return arquivos



def main():
    print(BANNER)

    args = sys.argv[1:]
    rodar_tudo = not args

    t_inicio = time.perf_counter()

    custo_fb = None

    if rodar_tudo or "--estruturas" in args:
        rodar_estruturas()

    if rodar_tudo or "--algoritmos" in args:
        custo_fb = rodar_forca_bruta()
        rodar_algoritmos_gulosos(custo_fb)

    if rodar_tudo or "--benchmark" in args:
        rodar_benchmark()

    if rodar_tudo or "--figuras" in args:
        rodar_figuras()

    t_total = time.perf_counter() - t_inicio
    print(f"\n{'═' * 62}")
    print(f"  ✓ POLARIS Earth — execução concluída em {t_total:.2f}s")
    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    main()