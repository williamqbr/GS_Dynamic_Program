from __future__ import annotations
from collections import deque


#  TUPLA: representação de vértice e aresta
# Vértice: (id_municipio, nome, indice_risco, custo_atendimento, populacao)
# Aresta:  (id_origem, id_destino, peso_km_ou_horas)

def criar_vertice(id_municipio: int, nome: str, indice_risco: float,
                  custo_atendimento: float, populacao: int) -> tuple:
    """Retorna tupla imutável representando um município."""
    if not (0.0 <= indice_risco <= 1.0):
        raise ValueError(f"indice_risco deve estar em [0,1]. Recebido: {indice_risco}")
    return (id_municipio, nome, round(indice_risco, 4), custo_atendimento, populacao)


def criar_aresta(origem: int, destino: int, peso: float) -> tuple:
    """Retorna tupla representando conexão entre municípios."""
    return (origem, destino, round(peso, 2))


class Grafo:

    def __init__(self):
        self._vertices: dict[int, tuple] = {}
        self._adj: dict[int, list[tuple]] = {}
        self._num_arestas: int = 0

    def adicionar_vertice(self, vertice: tuple) -> None:
        """Adiciona município ao grafo. """
        vid = vertice[0]
        if vid in self._vertices:
            raise ValueError(f"Vértice {vid} já existe.")
        self._vertices[vid] = vertice
        self._adj[vid] = []

    def obter_vertice(self, vid: int) -> tuple | None:
        return self._vertices.get(vid)

    def vertices(self) -> list[tuple]:
        """Retorna lista de todos os vértices."""
        return list(self._vertices.values())

    def ids_vertices(self) -> list[int]:
        return list(self._vertices.keys())

    def adicionar_aresta(self, origem: int, destino: int, peso: float) -> None:
        """Adiciona aresta não-direcionada."""
        if origem not in self._adj or destino not in self._adj:
            raise ValueError("Um ou ambos os vértices não existem no grafo.")
        self._adj[origem].append((destino, peso))
        self._adj[destino].append((origem, peso))
        self._num_arestas += 1

    def vizinhos(self, vid: int) -> list[tuple]:
        """Retorna lista [(vizinho_id, peso)] do vértice. O(grau(v))."""
        return self._adj.get(vid, [])

    def arestas(self) -> list[tuple]:
        """Retorna lista de todas as arestas (sem duplicatas)."""
        vistas: set[tuple] = set()   # set para verificação rápida O(1)
        resultado = []
        for u, vizs in self._adj.items():
            for v, w in vizs:
                edge = (min(u, v), max(u, v), w)
                if edge not in vistas:
                    vistas.add(edge)
                    resultado.append(edge)
        return resultado

    # BFS e DFS usando deque e set
    def bfs(self, inicio: int) -> list[int]:
        """
        Busca em largura a partir de 'inicio'.
        """
        if inicio not in self._adj:
            return []
        visitados: set[int] = {inicio}
        fila: deque[int] = deque([inicio])
        ordem: list[int] = []

        while fila:
            atual = fila.popleft()
            ordem.append(atual)
            for vizinho, _ in self._adj[atual]:
                if vizinho not in visitados:
                    visitados.add(vizinho)
                    fila.append(vizinho)
        return ordem

    def dfs(self, inicio: int, visitados: set[int] | None = None) -> list[int]:
        """
        Busca em profundidade recursiva.
        """
        if visitados is None:
            visitados = set()
        visitados.add(inicio)
        ordem = [inicio]
        for vizinho, _ in self._adj.get(inicio, []):
            if vizinho not in visitados:
                ordem.extend(self.dfs(vizinho, visitados))
        return ordem

    def eh_conexo(self) -> bool:
        """Verifica conectividade via BFS"""
        if not self._vertices:
            return True
        inicio = next(iter(self._adj))
        return len(self.bfs(inicio)) == len(self._vertices)

    def num_vertices(self) -> int:
        return len(self._vertices)

    def num_arestas(self) -> int:
        return self._num_arestas

    def __repr__(self) -> str:
        return (f"Grafo(V={self.num_vertices()}, E={self.num_arestas()}, "
                f"conexo={self.eh_conexo()})")


#  BST — Árvore Binária de Busca por risco

class Node:

    def __init__(self, vertice: tuple):
        self.chave: float = vertice[2]
        self.vertice: tuple = vertice
        self.esquerda: Node | None = None
        self.direita: Node | None = None

    def __repr__(self) -> str:
        return f"Node(risco={self.chave:.4f}, nome='{self.vertice[1]}')"


class BinarySearchTree:
    """
    ordenada por índice de risco.
    """

    def __init__(self):
        self.raiz: Node | None = None
        self._tamanho: int = 0

    # ── Inserção ─────────────────────────────
    def inserir(self, vertice: tuple) -> None:
        """Insere município mantendo propriedade BST. O(h)."""
        self.raiz = self._inserir_rec(self.raiz, vertice)
        self._tamanho += 1

    def _inserir_rec(self, no: Node | None, vertice: tuple) -> Node:
        if no is None:
            return Node(vertice)
        risco = vertice[2]
        if risco < no.chave:
            no.esquerda = self._inserir_rec(no.esquerda, vertice)
        else:
            no.direita = self._inserir_rec(no.direita, vertice)
        return no

    # ── Busca exata ──────────────────────────
    def buscar(self, risco: float) -> tuple | None:
        """Retorna vértice com índice de risco exato, ou None. O(h)."""
        no = self._buscar_rec(self.raiz, risco)
        return no.vertice if no else None

    def _buscar_rec(self, no: Node | None, risco: float) -> Node | None:
        if no is None or abs(no.chave - risco) < 1e-9:
            return no
        if risco < no.chave:
            return self._buscar_rec(no.esquerda, risco)
        return self._buscar_rec(no.direita, risco)

    def buscar_intervalo(self, r_min: float, r_max: float) -> list[tuple]:
        """
        Retorna todos os municípios com risco em [r_min, r_max].
        """
        resultado: list[tuple] = []
        self._buscar_intervalo_rec(self.raiz, r_min, r_max, resultado)
        return resultado

    def _buscar_intervalo_rec(self, no: Node | None, r_min: float,
                               r_max: float, resultado: list) -> None:
        if no is None:
            return
        if no.chave > r_min:
            self._buscar_intervalo_rec(no.esquerda, r_min, r_max, resultado)
        if r_min <= no.chave <= r_max:
            resultado.append(no.vertice)
        if no.chave < r_max:
            self._buscar_intervalo_rec(no.direita, r_min, r_max, resultado)

    def percurso_in_order(self) -> list[tuple]:
        """Retorna municípios em ordem crescente de risco. O(n)."""
        resultado: list[tuple] = []
        self._in_order_rec(self.raiz, resultado)
        return resultado

    def _in_order_rec(self, no: Node | None, resultado: list) -> None:
        if no is None:
            return
        self._in_order_rec(no.esquerda, resultado)
        resultado.append(no.vertice)
        self._in_order_rec(no.direita, resultado)

    def altura(self) -> int:
        """Calcula altura da árvore. O(n)."""
        return self._altura_rec(self.raiz)

    def _altura_rec(self, no: Node | None) -> int:
        if no is None:
            return -1
        return 1 + max(self._altura_rec(no.esquerda), self._altura_rec(no.direita))

    def fator_balanceamento(self) -> str:
        """Avalia quão balanceada está a árvore."""
        h = self.altura()
        import math
        h_ideal = math.log2(self._tamanho + 1) if self._tamanho > 0 else 0
        razao = h / h_ideal if h_ideal > 0 else 1.0
        if razao < 1.5:
            status = "BEM BALANCEADA"
        elif razao < 2.5:
            status = "MODERADAMENTE DESBALANCEADA"
        else:
            status = "DESBALANCEADA (considere AVL/RB)"
        return f"altura={h}, h_ideal≈{h_ideal:.1f}, razão={razao:.2f} → {status}"

    def remover(self, risco: float) -> bool:
        """Remove nó com dado índice de risco. Retorna True se removido. O(h)."""
        nova_raiz, removido = self._remover_rec(self.raiz, risco)
        if removido:
            self.raiz = nova_raiz
            self._tamanho -= 1
        return removido

    def _remover_rec(self, no: Node | None, risco: float) -> tuple[Node | None, bool]:
        if no is None:
            return None, False
        removido = False
        if risco < no.chave:
            no.esquerda, removido = self._remover_rec(no.esquerda, risco)
        elif risco > no.chave:
            no.direita, removido = self._remover_rec(no.direita, risco)
        else:
            removido = True
            if no.esquerda is None:
                return no.direita, True
            if no.direita is None:
                return no.esquerda, True
            # Dois filhos: substitui pelo sucessor in-order
            sucessor = self._minimo(no.direita)
            no.chave = sucessor.chave
            no.vertice = sucessor.vertice
            no.direita, _ = self._remover_rec(no.direita, sucessor.chave)
        return no, removido

    def _minimo(self, no: Node) -> Node:
        while no.esquerda:
            no = no.esquerda
        return no

    def tamanho(self) -> int:
        return self._tamanho

    def municipios_criticos(self, limiar: float = 0.8) -> list[tuple]:
        """Retorna municípios de alto risco (acima do limiar) em ordem. O(h+k)."""
        return self.buscar_intervalo(limiar, 1.0)

    def __repr__(self) -> str:
        return (f"BinarySearchTree(n={self._tamanho}, "
                f"altura={self.altura()}, "
                f"balanceamento={self.fator_balanceamento()})")


def carregar_cenario_rs() -> tuple[Grafo, BinarySearchTree]:
    """
    Cenário A — Subgrafo representativo de 12 municípios da região metropolitana de POA.
    """
    municipios = [
        criar_vertice(4314902, "Porto Alegre",   0.89, 1850.0, 1400000),
        criar_vertice(4304606, "Canoas",          0.95,  980.0,  356000),
        criar_vertice(4313409, "Pelotas",         0.72,  620.0,  328000),
        criar_vertice(4305108, "Caxias do Sul",   0.61,  840.0,  510000),
        criar_vertice(4316808, "Santa Maria",     0.78,  520.0,  277000),
        criar_vertice(4307708, "Gravataí",        0.91,  760.0,  255000),
        criar_vertice(4322400, "Viamão",          0.85,  580.0,  239000),
        criar_vertice(4318705, "São Leopoldo",    0.77,  490.0,  214000),
        criar_vertice(4315602, "Rio Grande",      0.68,  410.0,  208000),
        criar_vertice(4300604, "Alvorada",        0.93,  440.0,  196000),
        criar_vertice(4313375, "Novo Hamburgo",   0.74,  460.0,  247000),
        criar_vertice(4304614, "Charqueadas",     0.87,  280.0,   38000),
    ]

    # Arestas: (origem_id, destino_id, tempo_horas)
    arestas = [
        (4314902, 4304606, 0.6),
        (4314902, 4307708, 0.4),
        (4314902, 4322400, 0.3),
        (4314902, 4300604, 0.2),
        (4304606, 4307708, 0.3),
        (4304606, 4318705, 0.5),
        (4307708, 4300604, 0.3),
        (4322400, 4300604, 0.4),
        (4318705, 4313375, 0.3),
        (4305108, 4313375, 1.8),
        (4313409, 4315602, 1.2),
        (4316808, 4304614, 0.8),
        (4304614, 4314902, 0.7),
        (4304614, 4316808, 0.8),
        (4313375, 4307708, 0.6),
    ]

    grafo = Grafo()
    bst = BinarySearchTree()

    for m in municipios:
        grafo.adicionar_vertice(m)
        bst.inserir(m)

    for orig, dest, peso in arestas:
        grafo.adicionar_aresta(orig, dest, peso)

    return grafo, bst


def carregar_cenario_matopiba() -> tuple[Grafo, BinarySearchTree]:
    """
    Cenário B — Dados sintéticos baseados em valores históricos de 2023-2024.
    """
    municipios = [
        criar_vertice(1721000, "Palmas",              0.68, 1500.0, 310000),
        criar_vertice(2903201, "Barreiras",            0.91,  870.0, 160000),
        criar_vertice(2105302, "Imperatriz",           0.77,  780.0, 258000),
        criar_vertice(2911501, "Luís Eduardo Magalhães", 0.96, 430.0, 82000),
        criar_vertice(2100956, "Balsas",               0.88,  390.0, 86000),
        criar_vertice(2211100, "Uruçuí",               0.84,  320.0, 22000),
        criar_vertice(2904753, "Formosa do Rio Preto", 0.79,  260.0, 14000),
        criar_vertice(2907806, "Correntina",           0.73,  240.0, 30000),
        criar_vertice(1712702, "Wanderlândia",         0.62,  180.0, 12000),
        criar_vertice(1702109, "Araguaína",            0.70,  210.0, 181000),
        criar_vertice(1716604, "Paraíso do Tocantins", 0.74,  195.0, 47000),
        criar_vertice(1712405, "Dianópolis",           0.81,  170.0, 22000),
    ]

    arestas = [
        (1721000, 1716604, 0.9),
        (1721000, 2211100, 1.1),
        (2105302, 1702109, 0.5),
        (2105302, 1712702, 0.4),
        (1702109, 1712702, 0.3),
        (1702109, 1716604, 0.7),
        (1716604, 2211100, 0.6),
        (2211100, 2904753, 0.4),
        (2904753, 2903201, 0.8),
        (2904753, 2907806, 0.5),
        (2903201, 2911501, 0.3),
        (2907806, 2911501, 0.7),
        (1712405, 2211100, 0.5),
        (1712405, 2904753, 0.3),
        (2100956, 1712702, 0.6),
    ]

    grafo = Grafo()
    bst = BinarySearchTree()

    for m in municipios:
        grafo.adicionar_vertice(m)
        bst.inserir(m)

    for orig, dest, peso in arestas:
        grafo.adicionar_aresta(orig, dest, peso)

    return grafo, bst


#  DEMONSTRAÇÃO DAS ESTRUTURAS
if __name__ == "__main__":
    print("=" * 60)
    print("POLARIS Earth — Estruturas de Dados")
    print("=" * 60)

    grafo, bst = carregar_cenario_rs()

    print(f"\n{grafo}")
    print(f"\nBST: {bst}")

    print("\n— Percurso in-order (risco crescente):")
    for v in bst.percurso_in_order():
        print(f"  {v[1]:25s}  risco={v[2]:.2f}  pop={v[4]:>10,}")

    print(f"\n— Municípios críticos (risco > 0.85):")
    for v in bst.municipios_criticos(0.85):
        print(f"  {v[1]:25s}  risco={v[2]:.2f}")

    print(f"\n— BFS a partir de Porto Alegre:")
    ordem_bfs = grafo.bfs(4314902)
    nomes = [grafo.obter_vertice(vid)[1] for vid in ordem_bfs]
    print("  " + " → ".join(nomes))

    print(f"\n— Vizinhos de Porto Alegre:")
    for vid, peso in grafo.vizinhos(4314902):
        nome = grafo.obter_vertice(vid)[1]
        print(f"  {nome:25s}  {peso}h")