import numpy as np
import random
import matplotlib.pyplot as plt
import datetime
import os
import math

width, height = 100, 100
n_drones = 2
n_points = 5
obstacles = [(40, 40, 5), (70, 70, 10), (10, 20, 5)]
goal = (100, 100)
SAFE_DISTANCE = 3.0
EPS = 1e-6

W_LEN = 1.0
W_COLL = 1e2 * 5.0
W_CLEAR = 5.0
W_SMOOTH = 5.0
W_INTER = 1e2 * 7.0

output_dir = "/Users/mjl/python_code/YHXX/output_interaction"


def point_to_segment_distance(obs, a, b):
    (x, y), (x1, y1), (x2, y2) = obs, a, b
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(x - x1, y - y1)
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    pro_x = x1 + t * dx
    pro_y = y1 + t * dy
    return math.hypot(x - pro_x, y - pro_y)

def in_circle(a, b, obs):
    x, y, r = obs
    d = point_to_segment_distance((x, y), a, b)
    return d <= r + 1e-9

def angle_between(a, b, c):
    ux, uy = a[0] - b[0], a[1] - b[1]
    vx, vy = c[0] - b[0], c[1] - b[1]
    nu = math.hypot(ux, uy)
    nv = math.hypot(vx, vy)
    if nu < EPS or nv < EPS:
        return 0.0
    cosang = (ux * vx + uy * vy) / (nu * nv)
    cosang = max(-1.0, min(1.0, cosang))
    return math.acos(cosang)

def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    
def segments_intersect(a1, a2, b1, b2):
    return (ccw(a1,b1,b2) != ccw(a2, b1, b2)) and (ccw(a1, a2, b1)!=ccw(a1, a2, b2))

class Path:
    def __init__(self, drone_id = None):
        self.drone_id = drone_id

        self.path = [(0,0)]
        self.fitness = float("inf")

        for _ in range(n_points - 2):
            new_x = random.randint(0, width)
            new_y = random.randint(0, height)
            self.path.append((new_x, new_y))
        self.path.append((width, height))
    
    def evaluate(self, other_path = None):
        pts = self.path

        collisions = 0
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            for obs in obstacles:
                if in_circle(a, b, obs):
                    collisions += 1

        clear_pen = 0
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            for obs in obstacles:
                d = point_to_segment_distance((obs[0], obs[1]), a, b) - obs[2]
                if d < SAFE_DISTANCE:
                    clear_pen += (SAFE_DISTANCE - d) ** 2

        smooth_index = 0.0
        for i in range(1, len(pts) - 1):
            ang = angle_between(pts[i - 1], pts[i], pts[i + 1])
            smooth_index += (math.pi - ang) ** 2

        total_distance = 0
        for i in range(1, len(pts)):
            total_distance += self.distance(self.path[i - 1], self.path[i])

        interaction_pen = self.calculate_drone_interaction(other_path)

        self.fitness = W_LEN * total_distance + \
                    W_COLL * collisions + \
                    W_CLEAR * clear_pen + \
                    W_SMOOTH * smooth_index +\
                    W_INTER * interaction_pen
        
        return self.fitness

    def distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    
    def calculate_drone_interaction(self, other_path=None):
        if other_path == None:
            return 0
        collision = 0
        p0 = self.path
        p1 = other_path.path

        for i in range(len(p0)-1):
            a1, a2 = p0[i], p0[i+1]
            for j in range(len(p1)-1):
                b1, b2 = p1[j], p1[j+1]
                if segments_intersect(a1, a2, b1, b2):
                    collision += 1
        return collision

def initialize_population(pop_size, drone_id):
    return [Path(drone_id) for _ in range(pop_size)]


def select(population):
    population.sort(key=lambda x: x.fitness)
    chosen = population[:len(population) // 2]
    if len(chosen) < 2:
        chosen = population[:2]
    return chosen


def crossover(parent1, parent2):
    cp = random.randint(1, n_points - 2)
    child1 = Path(parent1.drone_id)
    child2 = Path(parent1.drone_id)

    child1.path = parent1.path[:cp] + parent2.path[cp:]
    child2.path = parent2.path[:cp] + parent1.path[cp:]

    return child1, child2


def mutate(path):
    mp = random.randint(1, len(path.path) - 2)
    path.path[mp] = (random.randint(0, width), random.randint(0, height))
    path.evaluate()

def plot_paths(paths, save_dir, filename):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.3)
        ax.add_artist(circle)
    
    for path_obj in paths:
        pts = np.array(path_obj.path)
        ax.plot(pts[:, 0], pts[:, 1], marker="o", linestyle="-", label=f"Drone {path_obj.drone_id} Path")
    
    ax.plot(goal[0], goal[1], "go", markersize=10, label="Goal")

    fitness_info = ', '.join([f"Drone {p.drone_id}: {p.fitness:.4f}" for p in paths])
    ax.set_title(f"Paths with Fitness Values: {fitness_info}")
    
    ax.legend()

    save_path = os.path.join(save_dir, f"{filename}")
    fig.savefig(save_path, dpi=200)
    print(f"[Saved] {save_path}")

    plt.close(fig)


def main():
    pop_size = 20
    generations = 1000
    save_interval = 50
    random.seed(42)

    time_dir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(output_dir, time_dir)
    os.makedirs(save_dir, exist_ok=True)

    populations = [initialize_population(pop_size, d)for d in range(n_drones)]
    for d in range(n_drones):
        for ind in populations[d]:
            ind.evaluate(other_path=None)
    
    best_overall = [min(populations[d], key=lambda x: x.fitness)for d in range(n_drones)]
    print(f"初始化最优路径适应度为{best_overall[0].fitness}、{best_overall[1].fitness}")

    for generation in range(1, generations+1):
        new_population = [None]*n_drones
        best_current = [None]*n_drones

        for d in range(n_drones):
            other_id = 1 - d
            if generation == 1:
                other_path = None
            else:
                other_path = best_overall[other_id]

            parents = select(populations[d])
            next_generation = []
            while len(next_generation) < len(populations[d]):
                p1, p2 = random.sample(parents, 2)
                c1, c2 = crossover(p1, p2)

                if random.random() < 0.1:
                    mutate(c1)
                if random.random() < 0.1:
                    mutate(c2)
                c1.evaluate(other_path)
                c2.evaluate(other_path)

                next_generation.append(c1)
                if len(next_generation) < len(populations[d]):
                    next_generation.append(c2)

            new_population[d] = next_generation
            best_current[d] = min(next_generation, key=lambda x: x.fitness)

        for d in range(n_drones):
            if best_overall[d].fitness > best_current[d].fitness:
                best_overall[d] = best_current[d]
        
        populations = new_population

        if generation % 20 == 0:
            print(f"Gen {generation}:best_current = {best_current[0].fitness:4f}、{best_current[1].fitness:4f}")
        
        if generation % save_interval == 0:
            plot_paths([best_current[0], best_current[1]],save_dir,filename=f"{generation}")
   
    plot_paths([best_overall[0], best_overall[1]], save_dir, filename="best")
    print(f"实验结束，最优适应度为{best_overall[0].fitness:4f}、{best_overall[1].fitness:4f}")

if __name__ == "__main__":
    main()
