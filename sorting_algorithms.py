# algorithm.py

from collections import deque
import random

# ── IMPORT THE NEW ML MODULE ────────────────────────────────────────────────────
from ml_sorting import OutfeedML

# ────────────────────────────────────────────────────────────────────────────────

def greedy(p, loads, outfeeds):
    """
    Initial assignment: choose feasible outfeed with minimal tracked load.

    Parameters:
    - p: Parcel instance with attributes `feasible_outfeeds` and `length`.
    - loads: dict mapping outfeed index to current tracked load.
    - outfeeds: list of Outfeed instances with attribute `max_length`.

    Returns:
    - index of chosen outfeed, or None if no feasible outfeed.
    """
    feas = [
        k
        for k in p.feasible_outfeeds
        if loads[k] + p.length <= outfeeds[k].max_length
    ]
    if not feas:
        return None
    return min(feas, key=lambda k: loads[k])


def imbalance(loads):
    """
    Compute objective: spread between heaviest and lightest load.
    """
    return max(loads.values()) - min(loads.values())


def run_local_search(window, loads, outfeeds, assignment, max_iters=100):
    """
    Hill-climbing to refine assignments in a sliding window.

    Parameters:
    - window: iterable of recent Parcel instances.
    - loads: dict mapping outfeed index to current tracked load.
    - outfeeds: list of Outfeed instances.
    - assignment: dict mapping parcel.id to its current assigned outfeed (or None).
    - max_iters: maximum hill-climbing iterations.

    Updates `assignment` in-place to the improved assignments.
    """
    # Copy loads for local search
    local_loads = loads.copy()
    # Initialize assignment for window
    assign_w = {}
    for p in window:
        k = greedy(p, local_loads, outfeeds)
        assign_w[p.id] = k
        if k is not None:
            local_loads[k] += p.length

    # Hill-climbing loop
    for _ in range(max_iters):
        improved = False
        for p in window:
            current = assign_w[p.id]
            for k in p.feasible_outfeeds:
                if k == current or local_loads[k] + p.length > outfeeds[k].max_length:
                    continue
                # simulate move
                new_loads = local_loads.copy()
                if current is not None:
                    new_loads[current] -= p.length
                new_loads[k] += p.length
                if imbalance(new_loads) < imbalance(local_loads):
                    # accept move
                    local_loads = new_loads
                    assign_w[p.id] = k
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Commit improved assignments back to system-wide assignment
    for pid, new_k in assign_w.items():
        assignment[pid] = new_k


def fcfs(parcel) -> deque[int]:
    """
    First-Come-First-Serve algorithm for assigning parcels to outfeeds.
    :param parcel: parcel object.
    :return: a double-ended queue containing the ordered outfeeds ID's.
    """
    return deque(parcel.feasible_outfeeds.copy())


def genetic(parcel, population_size=50, generations=100, mutation_rate=0.1) -> int:
    """
    Genetic algorithm to choose the optimal outfeed for a parcel.
    :param parcel: parcel object.
    :return: ID of optimal outfeed.
    """

    feasible_outfeeds = parcel.feasible_outfeeds

    def fitness(outfeed_id) -> float:
        """
        Prioritize closer outfeeds (outfeeds with lower index)
        :param outfeed_id: ID of the outfeed
        :return: float of representation of ID with higher fitness.
        """
        return 1 / (1 + outfeed_id)  # Lower index gets higher fitness

    population = [random.choice(feasible_outfeeds) for _ in range(population_size)]

    for _ in range(generations):
        fitness_scores = [fitness(individual) for individual in population]

        selected = []
        for _ in range(population_size):
            i1, i2 = random.sample(range(population_size), 2)
            selected.append(population[i1] if fitness_scores[i1] > fitness_scores[i2] else population[i2])

        children = []
        for _ in range(population_size):
            p1, p2 = random.sample(selected, 2)
            child = random.choice([p1, p2])
            children.append(child)

        for i in range(population_size):
            if random.random() < mutation_rate:
                children[i] = random.choice(feasible_outfeeds)

        population = children

    best_individual = max(population, key=fitness)
    return best_individual


# ── NEW: MACHINE-LEARNING-BASED “FIRST SERVE” ──────────────────────────────────────

# We’ll create a single global instance of OutfeedML. In practice, you might
# want to load a saved model from disk (e.g. “outfeed_model.pkl”). For now, 
# we leave `ml_model` un-trained until `main.py` calls its `.fit(...)`.
ml_model: OutfeedML = None

def initialize_ml_model(model_path: str = None):
    """
    Call this once at program start. If model_path is provided, we load a pre-trained model.
    Otherwise, we instantiate a fresh OutfeedML for training.
    """
    global ml_model
    if model_path:
        ml_model = OutfeedML.load(model_path)
    else:
        ml_model = OutfeedML()  # fresh, un-trained
    return ml_model


def mlfs(parcel) -> deque[int]:
    """
    “Machine-Learning First Serve”:
      1. We assume `ml_model` has already been `.fit(...)` or loaded.
      2. We compute one predicted outfeed ID via `ml_model.predict(parcel)`.
      3. We return a deque with exactly that one ID. 
         If that outfeed is full, your simulation logic will pop it,
         see it's full, then attempt next from `parcel.outfeed_attempts` (which
         came from parcel.feasible_outfeeds), and eventually recirculate if needed.
    """
    if ml_model is None or not ml_model.is_trained:
        raise RuntimeError("ml_model not initialized or not trained. Call initialize_ml_model(...) and then .fit(...) before using mlfs().")

    chosen = ml_model.predict(parcel)  # single int
    return deque([chosen])
