import random
import heapq
import time
from collections import deque

#  Constants

ROAD     = 0
BUILDING = 1
TRAFFIC  = 2
DELIVERY = 3
BASE     = 4

SYM = {ROAD: '.', BUILDING: '#', TRAFFIC: 'T', DELIVERY: 'D', BASE: 'B'}
SIZE = 15

#  Grid Construction

BUILDINGS = [
    (0,3),(0,4),(0,5),(1,8),(1,9),(2,2),(2,3),(3,11),(3,12),(3,13),
    (4,5),(4,6),(5,1),(5,2),(5,3),(6,9),(6,10),(7,4),(7,5),(7,6),
    (8,12),(8,13),(9,2),(9,3),(10,7),(10,8),(10,9),(11,0),(11,1),
    (12,5),(12,6),(13,11),(13,12),(13,13),(14,3),(14,4),
]

TRAFFIC_CELLS = [
    (0,7),(0,8),(2,6),(2,7),(4,10),(4,11),(6,3),(6,4),
    (8,6),(8,7),(10,2),(10,3),(12,9),(12,10),(14,7),(14,8),
]

def make_grid():
    g = [[ROAD]*SIZE for _ in range(SIZE)]
    for r, c in BUILDINGS:
        g[r][c] = BUILDING
    for r, c in TRAFFIC_CELLS:
        if g[r][c] == ROAD:
            g[r][c] = TRAFFIC
    return g

def make_costs(g, rng):
    costs = {}
    for r in range(SIZE):
        for c in range(SIZE):
            t = g[r][c]
            if t == BUILDING:
                costs[(r,c)] = 0
            elif t == TRAFFIC:
                costs[(r,c)] = rng.randint(10, 20)
            else:
                costs[(r,c)] = rng.randint(1, 5)
    return costs

def place_deliveries(g, base, rng, n=5):
    pool = [(r,c) for r in range(SIZE) for c in range(SIZE)
            if g[r][c] in (ROAD, TRAFFIC) and (r,c) != base]
    rng.shuffle(pool)
    chosen = pool[:n]
    for r, c in chosen:
        g[r][c] = DELIVERY
    return chosen

#  Graph Helpers

def adj(pos, g):
    r, c = pos
    out = []
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr, nc = r+dr, c+dc
        if 0 <= nr < SIZE and 0 <= nc < SIZE and g[nr][nc] != BUILDING:
            out.append((nr, nc))
    return out

def trace(parent, goal):
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

def total_cost(path, costs):
    return sum(costs[p] for p in path)

#  Search Algorithms

def bfs(g, costs, start, goal):
    # BFS – shortest path by cell count
    visited = {start}
    queue   = deque([start])
    parent  = {start: None}
    exp     = 0
    while queue:
        cur = queue.popleft()
        exp += 1
        if cur == goal:
            p = trace(parent, goal)
            return p, total_cost(p, costs), exp
        for nb in adj(cur, g):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = cur
                queue.append(nb)
    return None, 0, exp


def dfs(g, costs, start, goal):
    # DFS – mark visited ON PUSH (guarantees termination, O(V+E))
    visited = {start}
    stack   = [start]
    parent  = {start: None}
    exp     = 0
    while stack:
        cur = stack.pop()
        exp += 1
        if cur == goal:
            p = trace(parent, goal)
            return p, total_cost(p, costs), exp
        for nb in adj(cur, g):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = cur
                stack.append(nb)
    return None, 0, exp


def ucs(g, costs, start, goal):
    # UCS – minimum cost path
    heap   = [(0, start)]
    parent = {start: None}
    gcost  = {start: 0}
    exp    = 0
    while heap:
        cost, cur = heapq.heappop(heap)
        exp += 1
        if cur == goal:
            p = trace(parent, goal)
            return p, cost, exp
        if cost > gcost.get(cur, float('inf')):
            continue
        for nb in adj(cur, g):
            nc = gcost[cur] + costs[nb]
            if nc < gcost.get(nb, float('inf')):
                gcost[nb]  = nc
                parent[nb] = cur
                heapq.heappush(heap, (nc, nb))
    return None, 0, exp


ALGOS = {
    'BFS': bfs,
    'DFS': dfs,
    'UCS': ucs,
}

#  Text Visualisation

def show_grid(g, path=None, s=None, goal=None):
    ps = set(path) if path else set()
    print('      ' + ''.join(f'{c:<3}' for c in range(SIZE)))
    print('     ' + '─' * (SIZE * 3))
    for r in range(SIZE):
        row = f'{r:2} |  '
        for c in range(SIZE):
            p = (r, c)
            if   p == s:                                        row += 'S  '
            elif p == goal:                                     row += 'G  '
            elif p in ps and g[r][c] not in (BASE, DELIVERY):  row += '*  '
            else:                                               row += SYM.get(g[r][c], '?') + '  '
        print(row)
    print()

def legend():
    print("Legend: B=Base  D=Delivery  #=Building  T=Traffic(costly)  .=Road  *=Path  S=Start  G=Goal")
    print()

#  Delivery Simulation

def simulate(g, costs, deliveries, base, algo_name='UCS'):
    fn  = ALGOS[algo_name]
    pos = base
    tc = tt = tn = 0

    print('=' * 60)
    print(f'  DELIVERY SIMULATION  |  Algorithm: {algo_name}')
    print('=' * 60)

    for i, dest in enumerate(deliveries, 1):
        print(f'\n[Delivery {i}/{len(deliveries)}]  Start={pos}  -->  Goal={dest}')
        t0 = time.perf_counter()
        path, cost, nodes = fn(g, costs, pos, dest)
        elapsed = time.perf_counter() - t0

        if path is None:
            print('  !! No path found – skipping.')
            continue

        print(f'  Path length    : {len(path)} cells')
        print(f'  Travel cost    : {cost}')
        print(f'  Nodes explored : {nodes}')
        print(f'  Time (s)       : {elapsed:.6f}')
        print(f'  Route          : {path}')
        show_grid(g, path=path, s=pos, goal=dest)

        tc += cost;  tt += elapsed;  tn += nodes
        pos = dest   # robot moves to delivery point

    print('=' * 60)
    print('  TOTALS FOR THIS RUN')
    print(f'  Total cost     : {tc}')
    print(f'  Total time (s) : {tt:.6f}')
    print(f'  Nodes explored : {tn}')
    print('=' * 60)

#  Algorithm Comparison Table

def compare(g, costs, deliveries, base):
    print('\n' + '=' * 66)
    print('  PERFORMANCE COMPARISON  –  5 sequential deliveries')
    print('=' * 66)
    print(f"{'Algorithm':<14}  {'Total Cost':>11}  {'Time (s)':>10}  {'Nodes Exp':>10}")
    print('-' * 66)

    for name, fn in ALGOS.items():
        pos = base
        tc = tt = tn = 0
        ok = True
        for dest in deliveries:
            t0 = time.perf_counter()
            path, cost, nodes = fn(g, costs, pos, dest)
            tt += time.perf_counter() - t0
            if path is None:
                ok = False; break
            tc += cost;  tn += nodes;  pos = dest
        if ok:
            print(f'{name:<14}  {tc:>11}  {tt:>10.6f}  {tn:>10}')
        else:
            print(f'{name:<14}  {"N/A":>11}  {"N/A":>10}  {"N/A":>10}')

    print('=' * 66)
    print()
    print('  Metrics:')
    print('  Path Optimality  – lower total cost is better')
    print('  Execution Time   – lower is better')
    print('  Search Efficiency– fewer nodes explored is better')
    print()
    print('  UCS is guaranteed optimal (lowest cost).')
    print('  BFS finds the shortest path by cell count.')
    print('  DFS may find suboptimal paths but uses less memory.')
    print('=' * 66)

#  Main Entry Point

def main():
    rng = random.Random(42)   # fixed seed → reproducible environment

    g      = make_grid()
    costs  = make_costs(g, rng)

    BASE_POS = (7, 0)
    g[BASE_POS[0]][BASE_POS[1]] = BASE
    costs[BASE_POS] = 1

    deliveries = place_deliveries(g, BASE_POS, rng, n=5)

    print('\n' + '=' * 60)
    print('  URBAN DELIVERY ROBOT')
    print(f'  Path Planning on {SIZE} \u00d7 {SIZE} Grid')
    print('=' * 60)
    print(f'\n  Base Station : {BASE_POS}')
    print(f'  Deliveries   : {deliveries}')
    print()
    legend()

    print('Initial Grid:\n')
    show_grid(g)

    print('Cell traversal costs:')
    for label, pos in [('Base', BASE_POS)] + [(f'D{i}', d) for i, d in enumerate(deliveries, 1)]:
        print(f'  {label} {pos} = {costs[pos]}')

    # UCS step-by-step delivery with grid visuals
    print()
    simulate(g, costs, deliveries, BASE_POS, algo_name='UCS')

    # Comparison of BFS, DFS, and UCS
    compare(g, costs, deliveries, BASE_POS)


if __name__ == '__main__':
    main()
