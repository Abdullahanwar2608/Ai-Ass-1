import pygame
import math
import heapq
import random
import time
import sys

pygame.init()
pygame.font.init()

SCREEN_W = 1100
SCREEN_H = 800
BOARD_W = 800
MENU_W = 300
NUM_ROWS = 20

MAIN_WINDOW = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Advanced Dynamic Pathfinding Agent")

C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)
C_GREY = (200, 200, 200)
C_DARK_GREY = (50, 50, 50)
C_RED = (255, 60, 60)
C_GREEN = (46, 204, 113)
C_BLUE = (52, 152, 219)
C_YELLOW = (241, 196, 15)
C_PURPLE = (155, 89, 182)
C_ORANGE = (230, 126, 34)
C_CYAN = (0, 255, 255)

try:
    TXT_TITLE = pygame.font.SysFont('arial', 24, bold=True)
    TXT_NORMAL = pygame.font.SysFont('arial', 18)
    TXT_SMALL = pygame.font.SysFont('arial', 14)
except:
    TXT_TITLE = pygame.font.SysFont('comicsans', 24)
    TXT_NORMAL = pygame.font.SysFont('comicsans', 18)
    TXT_SMALL = pygame.font.SysFont('comicsans', 14)

class MapNode:
    def __init__(self, r, c, width, total_r):
        self.row = r
        self.col = c
        self.x = r * width
        self.y = c * width
        self.size = width
        self.color = C_WHITE
        self.adjacent_nodes = []
        self.max_rows = total_r
        self.prior_node = None

    def get_coords(self):
        return self.row, self.col

    def is_wall(self): return self.color == C_BLACK
    def is_start_point(self): return self.color == C_GREEN
    def is_target(self): return self.color == C_BLUE

    def clear_node(self): self.color = C_WHITE
    def set_start(self): self.color = C_GREEN
    def set_visited(self): self.color = C_RED
    def set_frontier(self): self.color = C_YELLOW
    def set_wall(self): self.color = C_BLACK
    def set_target(self): self.color = C_BLUE
    def set_route(self): self.color = C_PURPLE
    def set_entity(self): self.color = C_CYAN

    def render(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.size, self.size))

    def refresh_adjacent(self, board):
        self.adjacent_nodes = []
        if self.row < self.max_rows - 1 and not board[self.row + 1][self.col].is_wall():
            self.adjacent_nodes.append(board[self.row + 1][self.col])
        if self.row > 0 and not board[self.row - 1][self.col].is_wall():
            self.adjacent_nodes.append(board[self.row - 1][self.col])
        if self.col < self.max_rows - 1 and not board[self.row][self.col + 1].is_wall():
            self.adjacent_nodes.append(board[self.row][self.col + 1])
        if self.col > 0 and not board[self.row][self.col - 1].is_wall():
            self.adjacent_nodes.append(board[self.row][self.col - 1])

class PanelButton:
    def __init__(self, px, py, pw, ph, label, cmd_code):
        self.hitbox = pygame.Rect(px, py, pw, ph)
        self.label = label
        self.cmd_code = cmd_code
        self.base_color = C_GREY
        self.active_color = C_WHITE

    def render(self, surface):
        cursor_pos = pygame.mouse.get_pos()
        fill = self.active_color if self.hitbox.collidepoint(cursor_pos) else self.base_color
        pygame.draw.rect(surface, fill, self.hitbox)
        pygame.draw.rect(surface, C_BLACK, self.hitbox, 2)
        
        lbl_surface = TXT_NORMAL.render(self.label, True, C_BLACK)
        lbl_rect = lbl_surface.get_rect(center=self.hitbox.center)
        surface.blit(lbl_surface, lbl_rect)

    def check_click(self, pos):
        return self.hitbox.collidepoint(pos)

def calc_manhattan(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2)

def calc_euclidean(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def build_route(history, curr_node):
    route = []
    while curr_node in history:
        curr_node = history[curr_node]
        route.append(curr_node)
    route.reverse()
    return route

def run_pathfinder(board, start_pos, target_pos, heur_func, use_greedy, mult_weight):
    tie_breaker = 0
    frontier = []
    heapq.heappush(frontier, (0, tie_breaker, start_pos))
    history = {}
    
    cost_from_start = {cell: float("inf") for r in board for cell in r}
    cost_from_start[start_pos] = 0
    
    total_est_cost = {cell: float("inf") for r in board for cell in r}
    total_est_cost[start_pos] = heur_func(start_pos.get_coords(), target_pos.get_coords())

    frontier_tracker = {start_pos}
    explored_count = 0
    t_start = time.time()

    while not len(frontier) == 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        curr_node = heapq.heappop(frontier)[2]
        frontier_tracker.remove(curr_node)

        if curr_node == target_pos:
            t_end = time.time()
            final_route = build_route(history, target_pos)
            return final_route, explored_count, (t_end - t_start) * 1000

        explored_count += 1
        
        for adj in curr_node.adjacent_nodes:
            temp_cost = cost_from_start[curr_node] + 1

            if temp_cost < cost_from_start[adj]:
                history[adj] = curr_node
                cost_from_start[adj] = temp_cost
                h_val = heur_func(adj.get_coords(), target_pos.get_coords())
                
                if use_greedy:
                    total_est_cost[adj] = h_val * mult_weight
                else:
                    total_est_cost[adj] = temp_cost + (h_val * mult_weight)

                if adj not in frontier_tracker:
                    tie_breaker += 1
                    heapq.heappush(frontier, (total_est_cost[adj], tie_breaker, adj))
                    frontier_tracker.add(adj)
                    if adj != target_pos and adj != start_pos:
                        adj.set_frontier()

        if curr_node != start_pos:
            curr_node.set_visited()
            
    return None, explored_count, 0

def create_board(total_r, width):
    board = []
    cell_w = width // total_r
    for i in range(total_r):
        board.append([])
        for j in range(total_r):
            cell = MapNode(i, j, cell_w, total_r)
            board[i].append(cell)
    return board

def draw_board_lines(surface, total_r, width):
    cell_w = width // total_r
    for i in range(total_r):
        pygame.draw.line(surface, C_GREY, (0, i * cell_w), (width, i * cell_w))
        for j in range(total_r):
            pygame.draw.line(surface, C_GREY, (j * cell_w, 0), (j * cell_w, width))

def spawn_random_maze(board, total_r):
    for r in board:
        for cell in r:
            if not cell.is_start_point() and not cell.is_target():
                if random.random() < 0.3:
                    cell.set_wall()

def render_ui_panel(surface, ui_btns, stats, config):
    pygame.draw.rect(surface, C_DARK_GREY, (BOARD_W, 0, MENU_W, SCREEN_H))
    
    header = TXT_TITLE.render("Control Panel", True, C_WHITE)
    surface.blit(header, (BOARD_W + 70, 20))
    
    for btn in ui_btns:
        btn.render(surface)
        
    line_y = 450
    pygame.draw.line(surface, C_WHITE, (BOARD_W+20, line_y), (SCREEN_W-20, line_y), 2)
    
    stat_labels = ["Nodes Visited:", "Path Cost:", "Time (ms):"]
    stat_vals = [str(stats[0]), str(stats[1]), f"{stats[2]:.2f}"]
    
    for idx, (lbl, val) in enumerate(zip(stat_labels, stat_vals)):
        t_lbl = TXT_NORMAL.render(lbl, True, C_YELLOW)
        t_val = TXT_NORMAL.render(val, True, C_WHITE)
        surface.blit(t_lbl, (BOARD_W + 30, line_y + 20 + (idx*60)))
        surface.blit(t_val, (BOARD_W + 30, line_y + 45 + (idx*60)))

    cfg_y = 650
    pygame.draw.line(surface, C_WHITE, (BOARD_W+20, cfg_y), (SCREEN_W-20, cfg_y), 2)
    
    txt_algo = f"Algorithm: {'GBFS' if config['greedy'] else 'A*'}"
    txt_heur = f"Heuristic: {'Euclidean' if config['euclid'] else 'Manhattan'}"
    txt_wt = f"Weight: {config['weight']}x"
    txt_dyn = f"Dynamic Mode: {'ON' if config['dynamic'] else 'OFF'}"
    
    surface.blit(TXT_SMALL.render(txt_algo, True, C_WHITE), (BOARD_W + 30, cfg_y + 20))
    surface.blit(TXT_SMALL.render(txt_heur, True, C_WHITE), (BOARD_W + 30, cfg_y + 45))
    surface.blit(TXT_SMALL.render(txt_wt, True, C_WHITE), (BOARD_W + 30, cfg_y + 70))
    surface.blit(TXT_SMALL.render(txt_dyn, True, C_ORANGE if config['dynamic'] else C_GREY), (BOARD_W + 30, cfg_y + 95))

def main():
    node_board = create_board(NUM_ROWS, BOARD_W)
    
    start_pos = None
    target_pos = None
    
    btn_x = BOARD_W + 50
    ui_btns = [
        PanelButton(btn_x, 80, 200, 40, "Start Search", "CMD_START"),
        PanelButton(btn_x, 130, 200, 40, "Reset Path", "CMD_RESET"),
        PanelButton(btn_x, 180, 200, 40, "Clear Walls", "CMD_CLEAR"),
        PanelButton(btn_x, 230, 200, 40, "Random Maze", "CMD_MAZE"),
        PanelButton(btn_x, 290, 200, 30, "Toggle Algo", "CMD_ALGO"),
        PanelButton(btn_x, 330, 200, 30, "Toggle Heuristic", "CMD_HEUR"),
        PanelButton(btn_x, 370, 200, 30, "Toggle Weight", "CMD_WEIGHT"),
        PanelButton(btn_x, 410, 200, 30, "Dynamic Mode", "CMD_DYN"),
    ]
    
    config = {
        "greedy": False,
        "euclid": False,
        "weight": 1.0,
        "dynamic": False
    }
    stats = (0, 0, 0)
    
    active_route = []
    entity_loc = None
    is_running = True
    
    while is_running:
        MAIN_WINDOW.fill(C_WHITE)
        
        for r in node_board:
            for cell in r:
                cell.render(MAIN_WINDOW)
        draw_board_lines(MAIN_WINDOW, NUM_ROWS, BOARD_W)
        
        render_ui_panel(MAIN_WINDOW, ui_btns, stats, config)
        
        if active_route:
            for cell in active_route:
                if cell != target_pos and cell != start_pos and cell != entity_loc:
                    cell.set_route()
                    
        pygame.display.update()
        
        if active_route and entity_loc:
            time.sleep(0.15)
            
            upcoming_step = active_route.pop(0)
            
            if entity_loc != start_pos:
                entity_loc.set_visited()
                
            entity_loc = upcoming_step
            entity_loc.set_entity()
            
            if entity_loc == target_pos:
                print("Goal Reached")
                active_route = []
                entity_loc = None
                continue

        if config['dynamic']:
            if random.random() < 0.1: 
                rand_x = random.randint(0, NUM_ROWS-1)
                rand_y = random.randint(0, NUM_ROWS-1)
                mutated_cell = node_board[rand_x][rand_y]
                if not mutated_cell.is_start_point() and not mutated_cell.is_target() and mutated_cell != entity_loc:
                    mutated_cell.set_wall()
                    
            path_obstructed = False
            for cell in active_route:
                if cell.is_wall():
                    path_obstructed = True
                    break
                    
            if path_obstructed:
                print("Path Blocked! Re-planning...")
                for r in node_board:
                    for cell in r:
                        cell.refresh_adjacent(node_board)
                        
                calc_func = calc_euclidean if config['euclid'] else calc_manhattan
                fresh_route, v, t = run_pathfinder(node_board, entity_loc, target_pos, calc_func, config['greedy'], config['weight'])
                
                if fresh_route:
                    active_route = fresh_route
                    stats = (stats[0] + v, len(active_route), stats[2] + t)
                else:
                    print("No Path Possible!")
                    active_route = []

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False

            if pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                if mouse_pos[0] < BOARD_W:
                    grid_r, grid_c = mouse_pos[0] // (BOARD_W // NUM_ROWS), mouse_pos[1] // (BOARD_W // NUM_ROWS)
                    clicked_cell = node_board[grid_r][grid_c]
                    
                    if not start_pos and clicked_cell != target_pos:
                        start_pos = clicked_cell
                        start_pos.set_start()
                    elif not target_pos and clicked_cell != start_pos:
                        target_pos = clicked_cell
                        target_pos.set_target()
                    elif clicked_cell != target_pos and clicked_cell != start_pos:
                        clicked_cell.set_wall()
                else:
                    for btn in ui_btns:
                        if btn.check_click(mouse_pos):
                            if btn.cmd_code == "CMD_START" and start_pos and target_pos:
                                for r in node_board:
                                    for cell in r:
                                        cell.refresh_adjacent(node_board)
                                calc_func = calc_euclidean if config['euclid'] else calc_manhattan
                                route, v, t = run_pathfinder(node_board, start_pos, target_pos, calc_func, config['greedy'], config['weight'])
                                if route:
                                    active_route = route
                                    stats = (v, len(route), t)
                                    entity_loc = start_pos
                                    
                            elif btn.cmd_code == "CMD_RESET":
                                active_route = []
                                entity_loc = None
                                stats = (0,0,0)
                                for r in node_board:
                                    for cell in r:
                                        if cell.color in [C_RED, C_YELLOW, C_PURPLE, C_CYAN]:
                                            cell.clear_node()
                                        if cell == start_pos: cell.set_start()
                                        if cell == target_pos: cell.set_target()

                            elif btn.cmd_code == "CMD_CLEAR":
                                start_pos = None
                                target_pos = None
                                active_route = []
                                node_board = create_board(NUM_ROWS, BOARD_W)
                                stats = (0,0,0)

                            elif btn.cmd_code == "CMD_MAZE":
                                spawn_random_maze(node_board, NUM_ROWS)

                            elif btn.cmd_code == "CMD_ALGO":
                                config['greedy'] = not config['greedy']

                            elif btn.cmd_code == "CMD_HEUR":
                                config['euclid'] = not config['euclid']

                            elif btn.cmd_code == "CMD_WEIGHT":
                                config['weight'] = 2.5 if config['weight'] == 1.0 else 1.0

                            elif btn.cmd_code == "CMD_DYN":
                                config['dynamic'] = not config['dynamic']
                                
            elif pygame.mouse.get_pressed()[2]:
                mouse_pos = pygame.mouse.get_pos()
                if mouse_pos[0] < BOARD_W:
                    grid_r, grid_c = mouse_pos[0] // (BOARD_W // NUM_ROWS), mouse_pos[1] // (BOARD_W // NUM_ROWS)
                    clicked_cell = node_board[grid_r][grid_c]
                    clicked_cell.clear_node()
                    if clicked_cell == start_pos: start_pos = None
                    elif clicked_cell == target_pos: target_pos = None

    pygame.quit()

if __name__ == "__main__":
    main()