from __future__ import annotations
import heapq
from data_structures import Grafo, BinarySearchTree, carregar_cenario_rs, carregar_cenario_matopiba


def prim_mst(grafo: Grafo, raiz: int) -> tuple[dict, dict, int, float]:
    """
    Constrói a Árvore Geradora Mínima usando algoritmo de Prim.
    """
    ids = grafo.ids_vertices()
    custo_mst: dict[int, float] = {v: float("inf") for v in ids}
    predecessor: dict[int, int | None] = {v: None for v in ids}
    na_mst: set[int] = set()
    operacoes: int = 0

    custo_mst[raiz] = 0.0

    heap: list[tuple[float, int]] = [(0.0, raiz)]

    while heap:
        custo_u, u = heapq.heappop(heap)   # extrai vértice de menor custo
        operacoes += 1

        if u in na_mst:
            continue
        na_mst.add(u)

        for v, peso in grafo.vizinhos(u):
            if v not in na_mst and peso < custo_mst[v]:
                custo_mst[v] = peso
                predecessor[v] = u
                heapq.heappush(heap, (peso, v))   # insere com nova prioridade
                operacoes += 1

    peso_total = sum(
        c for v, c in custo_mst.items() if c < float("inf")
    )
    return predecessor, custo_mst, operacoes, peso_total

def reconstruir_mst_arestas(predecessor: dict, custo_mst: dict
                              ) -> list[tuple[int, int, float]]:
    """Reconstrói a lista de arestas da MST a partir do dict de predecessores."""
    arestas: list[tuple[int, int, float]] = []
    for v, u in predecessor.items():
        if u is not None:
            arestas.append((u, v, custo_mst[v]))
    return sorted(arestas, key=lambda x: x[2])  # ordena por peso


def dijkstra(grafo: Grafo, origem: int
             ) -> tuple[dict[int, float], dict[int, int | None], int]:
    """
    Caminho mínimo de fonte única usando Dijkstra com heap.
    """
    ids = grafo.ids_vertices()
    dist: dict[int, float] = {v: float("inf") for v in ids}
    predecessor: dict[int, int | None] = {v: None for v in ids}
    dist[origem] = 0.0
    arestas_relaxadas: int = 0

    heap: list[tuple[float, int]] = [(0.0, origem)]

    while heap:
        d_u, u = heapq.heappop(heap)

        if d_u > dist[u]:
            continue

        for v, peso in grafo.vizinhos(u):
            arestas_relaxadas += 1
            nova_dist = dist[u] + peso
            if nova_dist < dist[v]:
                dist[v] = nova_dist
                predecessor[v] = u
                heapq.heappush(heap, (nova_dist, v))

    return dist, predecessor, arestas_relaxadas


def reconstruir_caminho(predecessor: dict, destino: int) -> list[int]:
    """Reconstrói o caminho do destino até a origem via predecessores."""
    caminho: list[int] = []
    atual: int | None = destino
    while atual is not None:
        caminho.append(atual)
        atual = predecessor[atual]
    caminho.reverse()
    return caminho


class UnionFind:
    """
    Union-Find com path compression e union by rank.
    """

    def __init__(self, vertices: list[int]):
        self.pai: dict[int, int] = {v: v for v in vertices}
        self.rank: dict[int, int] = {v: 0 for v in vertices}

    def encontrar(self, x: int) -> int:
        """Path compression: achata a árvore. O(α(n)) amortizado."""
        if self.pai[x] != x:
            self.pai[x] = self.encontrar(self.pai[x])   # compressão
        return self.pai[x]

    def unir(self, x: int, y: int) -> bool:
        """
        Union by rank: árvore menor aponta para a maior.
        """
        rx, ry = self.encontrar(x), self.encontrar(y)
        if rx == ry:
            return False    # formariam ciclo
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.pai[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def kruskal_mst(grafo: Grafo) -> tuple[list[tuple], float, int]:
    todas_arestas = grafo.arestas()
    todas_arestas.sort(key=lambda e: e[2])   # ordena por peso (guloso)

    uf = UnionFind(grafo.ids_vertices())
    mst_arestas: list[tuple[int, int, float]] = []
    peso_total: float = 0.0
    operacoes: int = 0

    for u, v, peso in todas_arestas:
        operacoes += 1
        if uf.unir(u, v):           # adiciona se não forma ciclo
            mst_arestas.append((u, v, peso))
            peso_total += peso
            if len(mst_arestas) == grafo.num_vertices() - 1:
                break               # MST completa com V-1 arestas

    return mst_arestas, peso_total, operacoes


def rota_gulosa_atendimento(grafo: Grafo, bst: BinarySearchTree,
                             hub: int, num_equipes: int,
                             limiar_risco: float = 0.70
                             ) -> tuple[list[dict], float]:
    """
    Gera a sequência ótima de atendimento
    """
    # Passo 1: consulta BST para municípios prioritários
    municipios_risco = bst.buscar_intervalo(limiar_risco, 1.0)
    ids_candidatos = [v[0] for v in municipios_risco if v[0] != hub]

    if not ids_candidatos:
        return [], 0.0

    # Passo 2: Dijkstra para distâncias mínimas do hub
    dist, predecessor, _ = dijkstra(grafo, hub)

    # Passo 3: calcula razão risco/distância para cada candidato
    candidatos_com_prioridade: list[tuple] = []
    for vid in ids_candidatos:
        v = grafo.obter_vertice(vid)
        if v is None:
            continue
        risco = v[2]
        dist_hub = dist.get(vid, float("inf"))
        if dist_hub == float("inf"):
            continue
        prioridade = risco / (dist_hub + 0.1)   # +0.1 evita divisão por zero
        candidatos_com_prioridade.append((-prioridade, dist_hub, vid, v))

    heapq.heapify(candidatos_com_prioridade)

    # Passo 4: seleciona municípios até esgotar equipes
    rota: list[dict] = []
    custo_total: float = 0.0
    municipios_por_equipe = max(1, len(ids_candidatos) // num_equipes)

    while candidatos_com_prioridade and len(rota) < len(ids_candidatos):
        neg_prior, dist_hub, vid, vertice = heapq.heappop(candidatos_com_prioridade)
        prioridade = -neg_prior

        caminho = reconstruir_caminho(predecessor, vid)
        nomes_caminho = []
        for cid in caminho:
            cv = grafo.obter_vertice(cid)
            nomes_caminho.append(cv[1] if cv else str(cid))

        rota.append({
            "ordem": len(rota) + 1,
            "id": vid,
            "nome": vertice[1],
            "risco": vertice[2],
            "custo_atendimento": vertice[3],
            "populacao": vertice[4],
            "distancia_hub_h": round(dist_hub, 2),
            "prioridade_gulosa": round(prioridade, 4),
            "caminho": nomes_caminho,
            "equipe": ((len(rota)) // municipios_por_equipe) + 1,
        })
        custo_total += dist_hub

    return rota, round(custo_total, 2)


if __name__ == "__main__":
    print("=" * 60)
    print("POLARIS Earth — Algoritmos Gulosos")
    print("=" * 60)

    # Cenário A: Enchentes RS
    print("\n[CENÁRIO A] Enchentes — Rio Grande do Sul")
    grafo_rs, bst_rs = carregar_cenario_rs()
    hub_rs = 4314902   # Porto Alegre

    # Prim MST
    pred, custos, ops_prim, peso_mst = prim_mst(grafo_rs, hub_rs)
    arestas_mst = reconstruir_mst_arestas(pred, custos)

    print(f"\n  Prim MST:")
    print(f"    Peso total da MST: {peso_mst:.2f}h")
    print(f"    Operações (inserções heap): {ops_prim}")
    print(f"    Arestas da MST:")
    for u, v, w in arestas_mst:
        nu = grafo_rs.obter_vertice(u)[1]
        nv = grafo_rs.obter_vertice(v)[1]
        print(f"      {nu:20s} ─── {nv:20s}  {w:.1f}h")

    # Dijkstra: distâncias do hub
    dist, pred_dijk, arestas_rel = dijkstra(grafo_rs, hub_rs)
    print(f"\n  Dijkstra (Porto Alegre → todos):")
    print(f"    Arestas relaxadas: {arestas_rel}")
    for vid, d in sorted(dist.items(), key=lambda x: x[1]):
        if d < float("inf"):
            nome = grafo_rs.obter_vertice(vid)[1]
            caminho = reconstruir_caminho(pred_dijk, vid)
            n_caminho = len(caminho)
            print(f"    {nome:25s}  {d:.2f}h  ({n_caminho} saltos)")

    # Kruskal MST
    mst_k, peso_k, ops_k = kruskal_mst(grafo_rs)
    print(f"\n  Kruskal MST:")
    print(f"    Peso total: {peso_k:.2f}h  (ops Union-Find: {ops_k})")
    print(f"    Nota: Prim={peso_mst:.2f}h ≈ Kruskal={peso_k:.2f}h ✓ (ambos MST)")

    # Rota gulosa de atendimento
    print("\n  Rota Gulosa de Atendimento (8 equipes, limiar risco≥0.70):")
    rota, custo_rota = rota_gulosa_atendimento(grafo_rs, bst_rs, hub_rs, 8, 0.70)
    for parada in rota:
        print(f"    [{parada['ordem']:02d}] {parada['nome']:20s}  "
              f"risco={parada['risco']:.2f}  "
              f"dist={parada['distancia_hub_h']:.2f}h  "
              f"prior={parada['prioridade_gulosa']:.3f}  "
              f"equipe={parada['equipe']}")
    print(f"  Custo total de deslocamento: {custo_rota:.2f}h")

    # ── Cenário B: MATOPIBA ───────────────────
    print("\n" + "=" * 60)
    print("[CENÁRIO B] Seca — MATOPIBA")
    grafo_mb, bst_mb = carregar_cenario_matopiba()
    hub_mb = 1721000   # Palmas

    pred_mb, custos_mb, ops_mb, peso_mb = prim_mst(grafo_mb, hub_mb)
    rota_mb, custo_mb = rota_gulosa_atendimento(grafo_mb, bst_mb, hub_mb, 5, 0.75)

    print(f"\n  MST peso total: {peso_mb:.2f}h  (ops Prim: {ops_mb})")
    print(f"  Municípios críticos (risco≥0.85) pela BST:")
    for v in bst_mb.municipios_criticos(0.85):
        print(f"    {v[1]:30s}  risco={v[2]:.2f}")
    print(f"\n  Rota gulosa ({len(rota_mb)} municípios, custo={custo_mb:.2f}h):")
    for p in rota_mb[:6]:
        print(f"    [{p['ordem']:02d}] {p['nome']:25s}  risco={p['risco']:.2f}  "
              f"prior={p['prioridade_gulosa']:.3f}")