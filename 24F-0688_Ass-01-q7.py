# AI Pathfinder - Assignment 1, Question 7
# Student IDs: 24F-0688, 24F-0660
# Run: pip install pygame

import pygame
import collections
import heapq
import random
import sys

# ── Window & grid settings ────────────────────────────────────────────────────
CELL_SIZE    = 64
MAP_SIZE     = 10
PANEL_WIDTH  = 220
WINDOW_W     = MAP_SIZE * CELL_SIZE + PANEL_WIDTH
WINDOW_H     = MAP_SIZE * CELL_SIZE
FPS          = 60
STEP_DELAY   = 60   # Slightly faster for IDDFS visualization

# ── Colours ───────────────────────────────────────────────────────────────────
WHITE        = (255, 255, 255)
GREY         = (200, 200, 200)
DARK_GREY    = (100, 100, 100)
PANEL_BG     = ( 30,  30,  45)
STATIC_WALL  = (100,   0,   0)
DYNAMIC_WALL = (210, 100,   0)
EXPLORED     = (255, 220,  80)
FRONTIER     = (255, 140,   0)
PATH_COLOR   = ( 50, 220,  80)
START_COLOR  = ( 20, 180,  20)
TARGET_COLOR = ( 30,  80, 220)
CURRENT_CLR  = (  0, 220, 220)
TEXT_LIGHT   = (220, 220, 220)
BTN_NORMAL   = ( 60,  60,  90)
BTN_HOVER    = ( 90,  90, 130)
BTN_ACTIVE   = ( 40, 160,  90)

EMPTY   =  0
STATIC  = -1
DYNAMIC = -2

class AIPathfinder:
    def __init__(self):
        self.map_size    = MAP_SIZE
        self.map         = [[EMPTY] * MAP_SIZE for _ in range(MAP_SIZE)]
        self.start       = (7, 7)
        self.target      = (8, 1)
        self.spawn_chance = 0.03
        self.random_walls = []

        # [cite_start]STRICT VISIBLE MOVEMENT ORDER [cite: 31-37]
        # 1. Up, 2. Right, 3. Bottom, 4. Bottom-Right, 5. Left, 6. Top-Left
        self.directions = [
            (-1,  0),   # 1. Up
            ( 0,  1),   # 2. Right
            ( 1,  0),   # 3. Bottom
            ( 1,  1),   # 4. Bottom-Right (Diagonal)
            ( 0, -1),   # 5. Left
            (-1, -1),   # 6. Top-Left (Diagonal)
        ]

        # Static wall setup
        for row in range(2, 8):
            self.map[row][5] = STATIC

    def clear_random_walls(self):
        for row, col in self.random_walls:
            if self.map[row][col] == DYNAMIC:
                self.map[row][col] = EMPTY
        self.random_walls.clear()

    def is_walkable(self, cell):
        row, col = cell
        return (0 <= row < self.map_size and 0 <= col < self.map_size and
                self.map[row][col] != STATIC and self.map[row][col] != DYNAMIC)

    def maybe_spawn_wall(self, safe_cells=None):
        if safe_cells is None: safe_cells = set()
        safe_cells.add(self.start)
        safe_cells.add(self.target)

        if random.random() < self.spawn_chance:
            open_cells = [(r, c) for r in range(self.map_size) for c in range(self.map_size)
                          if self.map[r][c] == EMPTY and (r, c) not in safe_cells]
            if open_cells:
                chosen = random.choice(open_cells)
                self.map[chosen[0]][chosen[1]] = DYNAMIC
                self.random_walls.append(chosen)
                return chosen
        return None

    def trace_back_path(self, came_from, end_node):
        route = []
        current = end_node
        while current is not None:
            route.append(current)
            current = came_from.get(current)
        return route[::-1]

    # --- Search Generators ---
    
    def bfs_gen(self):
        self.clear_random_walls()
        to_visit  = collections.deque([self.start])
        came_from = {self.start: None}
        visited   = []

        while to_visit:
            current = to_visit.popleft()
            if current == self.target:
                yield {"current": current, "visited": visited, "frontier": list(to_visit), 
                       "path": self.trace_back_path(came_from, self.target), "done": True}
                return
            if current not in visited:
                visited.append(current)
                for dr, dc in self.directions:
                    neighbor = (current[0] + dr, current[1] + dc)
                    if self.is_walkable(neighbor) and neighbor not in came_from:
                        came_from[neighbor] = current
                        to_visit.append(neighbor)
                self.maybe_spawn_wall(safe_cells=set(visited) | set(to_visit))
            yield {"current": current, "visited": list(visited), "frontier": list(to_visit), "path": None, "done": False}
        yield {"done": True, "failed": True}

    def dfs_gen(self, depth_limit=None):
        # NOTE: We do NOT clear walls here for IDDFS, otherwise walls vanish every depth increase.
        if depth_limit is None: 
            self.clear_random_walls()
            
        to_visit  = [(self.start, 0)]
        came_from = {self.start: None}
        visited   = []

        while to_visit:
            current, current_depth = to_visit.pop()
            if current == self.target:
                yield {"current": current, "visited": visited, "frontier": [n for n,_ in to_visit], 
                       "path": self.trace_back_path(came_from, self.target), "done": True}
                return
            if current not in visited:
                if depth_limit is None or current_depth < depth_limit:
                    visited.append(current)
                    for dr, dc in reversed(self.directions):
                        neighbor = (current[0] + dr, current[1] + dc)
                        if self.is_walkable(neighbor) and neighbor not in came_from:
                            came_from[neighbor] = current
                            to_visit.append((neighbor, current_depth + 1))
                    self.maybe_spawn_wall(safe_cells={n for n,_ in to_visit})
            yield {"current": current, "visited": list(visited), "frontier": [n for n,_ in to_visit], "path": None, "done": False}
        yield {"done": True, "failed": True}

    def ucs_gen(self):
        self.clear_random_walls()
        pq = [(0, self.start)]
        came_from = {self.start: None}
        cost_so_far = {self.start: 0}
        visited = []

        while pq:
            current_cost, current = heapq.heappop(pq)
            if current == self.target:
                yield {"current": current, "visited": visited, "frontier": [n for _,n in pq], 
                       "path": self.trace_back_path(came_from, self.target), "done": True}
                return
            if current not in visited:
                visited.append(current)
                for dr, dc in self.directions:
                    neighbor = (current[0] + dr, current[1] + dc)
                    new_cost = current_cost + 1
                    if self.is_walkable(neighbor):
                        if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                            cost_so_far[neighbor] = new_cost
                            came_from[neighbor] = current
                            heapq.heappush(pq, (new_cost, neighbor))
                self.maybe_spawn_wall(safe_cells={n for _,n in pq})
            yield {"current": current, "visited": list(visited), "frontier": [n for _,n in pq], "path": None, "done": False}
        yield {"done": True, "failed": True}

    def iddfs_gen(self):
        self.clear_random_walls()
        # 
        # Iterate through depths 1 to 100
        for limit in range(1, self.map_size * self.map_size):
            # 1. Reset for new depth
            yield {"current": self.start, "visited": [], "frontier": [], "path": None, "done": False}
            
            # 2. Run DFS with current limit
            gen = self.dfs_gen(depth_limit=limit)
            
            for state in gen:
                # 3. FIX: Only yield "Done" if we actually found the target (Success)
                #    If DFS returns "Done + Failed", we IGNORE it so loop continues to next depth.
                if state.get("done"):
                    if not state.get("failed"):
                        yield state # Success!
                        return
                    # Else: It failed this depth, do nothing (continue to next depth limit)
                else:
                    yield state # Yield intermediate steps
        
        # If we exit the loop, we really failed
        yield {"done": True, "failed": True}

    def bidirectional_gen(self):
        self.clear_random_walls()
        f_q = collections.deque([self.start]); f_came = {self.start: None}; f_vis = set()
        b_q = collections.deque([self.target]); b_came = {self.target: None}; b_vis = set()

        def get_bidir_path(meet):
            p1 = self.trace_back_path(f_came, meet)
            p2 = []
            curr = meet
            while curr:
                curr = b_came.get(curr)
                if curr: p2.append(curr)
            return p1 + p2

        while f_q and b_q:
            # Forward
            curr_f = f_q.popleft()
            f_vis.add(curr_f)
            for dr, dc in self.directions:
                n = (curr_f[0] + dr, curr_f[1] + dc)
                if self.is_walkable(n) and n not in f_came:
                    f_came[n] = curr_f
                    f_q.append(n)
                    if n in b_came:
                        yield {"current": n, "visited": list(f_vis|b_vis), "frontier": list(f_q)+list(b_q), 
                               "path": get_bidir_path(n), "done": True}; return
            # Backward
            curr_b = b_q.popleft()
            b_vis.add(curr_b)
            for dr, dc in self.directions:
                n = (curr_b[0] + dr, curr_b[1] + dc)
                if self.is_walkable(n) and n not in b_came:
                    b_came[n] = curr_b
                    b_q.append(n)
                    if n in f_came:
                        yield {"current": n, "visited": list(f_vis|b_vis), "frontier": list(f_q)+list(b_q), 
                               "path": get_bidir_path(n), "done": True}; return
            
            self.maybe_spawn_wall(safe_cells=set(f_q)|set(b_q))
            yield {"current": curr_f, "visited": list(f_vis|b_vis), "frontier": list(f_q)+list(b_q), "path": None, "done": False}
        yield {"done": True, "failed": True}

# ── Pygame UI ─────────────────────────────────────────────────────────────────

class Button:
    def __init__(self, x, y, w, h, text, key):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text; self.algo_key = key
        self.hovered = False; self.active = False
    def draw(self, surface, font):
        color = BTN_ACTIVE if self.active else (BTN_HOVER if self.hovered else BTN_NORMAL)
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        label = font.render(self.text, True, WHITE)
        surface.blit(label, label.get_rect(center=self.rect.center))
    def check_hover(self, pos): self.hovered = self.rect.collidepoint(pos)
    def is_clicked(self, pos): return self.rect.collidepoint(pos)

class PygameApp:
    def __init__(self):
        pygame.init()
        self.screen  = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("AI Pathfinder")
        self.clock   = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("consolas", 13)
        self.pf      = AIPathfinder()
        self.state   = None
        self.gen     = None
        self.running_algo = None
        self.paused  = False
        self.step_timer = 0

        # UI Layout
        panel_x = MAP_SIZE * CELL_SIZE + 10
        btn_w, btn_h = PANEL_WIDTH - 20, 34
        self.buttons = []
        labels = [("BFS", "bfs"), ("DFS", "dfs"), ("UCS", "ucs"), 
                  ("DLS (d=15)", "dls"), ("IDDFS", "iddfs"), ("Bidirectional", "bidir")]
        for i, (text, key) in enumerate(labels):
            self.buttons.append(Button(panel_x, 110 + i * 42, btn_w, btn_h, text, key))
        self.pause_btn = Button(panel_x, 110 + len(labels) * 42 + 10, btn_w, btn_h, "Pause / Resume", "pause")
        self.reset_btn = Button(panel_x, 110 + len(labels) * 42 + 54, btn_w, btn_h, "Reset", "reset")

    def start_algo(self, key):
        self.pf = AIPathfinder()
        self.running_algo = key
        self.paused = False
        self.state = None
        if key == "bfs":   self.gen = self.pf.bfs_gen()
        elif key == "dfs": self.gen = self.pf.dfs_gen()
        elif key == "ucs": self.gen = self.pf.ucs_gen()
        elif key == "dls": self.gen = self.pf.dfs_gen(depth_limit=15)
        elif key == "iddfs": self.gen = self.pf.iddfs_gen()
        elif key == "bidir": self.gen = self.pf.bidirectional_gen()
        for btn in self.buttons: btn.active = (btn.algo_key == key)

    def draw_grid(self):
        visited = set(self.state["visited"]) if self.state else set()
        frontier = set(self.state["frontier"]) if self.state else set()
        path = set(self.state["path"] or []) if self.state else set()
        current = self.state["current"] if self.state else None
        
        for r in range(MAP_SIZE):
            for c in range(MAP_SIZE):
                x, y = c * CELL_SIZE, r * CELL_SIZE
                color = WHITE
                val = self.pf.map[r][c]
                if val == STATIC: color = STATIC_WALL
                elif val == DYNAMIC: color = DYNAMIC_WALL
                elif (r,c) == self.pf.start: color = START_COLOR
                elif (r,c) == self.pf.target: color = TARGET_COLOR
                elif (r,c) in path: color = PATH_COLOR
                elif (r,c) == current: color = CURRENT_CLR
                elif (r,c) in visited: color = EXPLORED
                elif (r,c) in frontier: color = FRONTIER
                
                pygame.draw.rect(self.screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self.screen, GREY, (x, y, CELL_SIZE, CELL_SIZE), 1)
                
                if val == DYNAMIC:
                    lbl = self.font_sm.render("!", True, WHITE)
                    self.screen.blit(lbl, lbl.get_rect(center=(x+32, y+32)))

    def draw_panel(self):
        panel_rect = pygame.Rect(MAP_SIZE * CELL_SIZE, 0, PANEL_WIDTH, WINDOW_H)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        panel_x = MAP_SIZE * CELL_SIZE + 10
        
        # CORRECT STUDENT IDs
        ids = self.font_sm.render("24F-0688 & 24F-0660", True, TEXT_LIGHT)
        self.screen.blit(ids, (panel_x, 48))

        mouse_pos = pygame.mouse.get_pos()
        for btn in self.buttons + [self.pause_btn, self.reset_btn]:
            btn.check_hover(mouse_pos)
            btn.draw(self.screen, self.font_sm)

    def tick_algo(self):
        if self.gen is None or self.paused: return
        if self.state and self.state.get("done"): return

        # Dynamic Obstacle Check
        if self.state and self.state.get("path"):
            if any(not self.pf.is_walkable(node) for node in self.state["path"]):
                self.start_algo(self.running_algo) # Re-plan
                return

        now = pygame.time.get_ticks()
        if now - self.step_timer < STEP_DELAY: return
        self.step_timer = now
        try:
            self.state = next(self.gen)
        except StopIteration: pass

    def run(self):
        while True:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for btn in self.buttons:
                        if btn.is_clicked(event.pos): self.start_algo(btn.algo_key)
                    if self.pause_btn.is_clicked(event.pos): self.paused = not self.paused
                    if self.reset_btn.is_clicked(event.pos): self.pf = AIPathfinder(); self.gen = None; self.state = None
            
            self.tick_algo()
            self.screen.fill(WHITE)
            self.draw_grid()
            self.draw_panel()
            pygame.display.flip()

if __name__ == "__main__":
    PygameApp().run()