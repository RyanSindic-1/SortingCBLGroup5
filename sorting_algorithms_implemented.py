# algorithm.py

from collections import deque
import random
from datetime import timedelta
import numpy as np

# ── IMPORT THE NEW ML MODULE ────────────────────────────────────────────────────
from ml_sorting import OutfeedML

# ────────────────────────────────────────────────────────────────────────────────
def greedy_time(self, p):
    """
    Initial assignment: choose feasible outfeed with minimal tracked load.

    Returns the index of the chosen outfeed, or None if no feasible outfeed can accept the parcel right now.
    """
    feas = [k for k in p.feasible_outfeeds if (self.outfeeds[k].can_accept(p))]
    if not feas:
        return None
    return min(feas, key=lambda k: self.loads[k])


def imbalance_time(self, loads):
    """
    Compute load imbalance = (max load) - (min load) across all outfeeds.

    Parameters:
    - loads: sum of service_time per outfeed
    """
    return max(loads.values()) - min(loads.values())


def run_local_search_time(self, max_iters=100):
    """
    Hill climbing to refine assignments of parcels in the window, 
    trying to minimize the imbalance between the outfeeds.
    max_iters is the maximum hill climbing iterations that will be performed.
    """
    # Copy current loads and build a temporary assignment for the window
    loads = self.loads.copy()
    assign_w = {}

    # Assign parcel to the outfeed with the minimal tracked load (greedy_time)
    for (_, p) in self.window:
        k = greedy_time(self, p)
        assign_w[p.id] = k
        if k is not None:
            loads[k] += self.service_times.get(p.id, 0.0)

    # Hill‐climb: try moving one parcel at a time to reduce imbalance
    for _ in range(max_iters):
        improved = False
        for (_, p) in self.window:
            cur = assign_w[p.id]
            for k in p.feasible_outfeeds:
                if k == cur or not self.outfeeds[k].can_accept(p):
                    continue
                st = self.service_times.get(p.id, 0.0)
                new_loads = loads.copy()
                if cur is not None:
                    new_loads[cur] -= st
                new_loads[k] += st
                if imbalance_time(self, new_loads) < imbalance_time(self, loads):
                    # accept this move
                    loads, assign_w[p.id] = new_loads, k
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Commit assignments back into self.assignment
    for pid, k in assign_w.items():
        self.assignment[pid] = k


def handle_enter_scanner_time(self, evt, fes):
    """
    This function handles the ENTER_SCANNER event.
    It assigns the parcel to an outfeed based on a greedy strategy,
    afthet that it updates the sliding window of recent parcels and performs local search
    for load balancing every REBALANCE_INTERVAL arrivals. What this means is that every certain number of parcels,
    we try to reassign parcels in the sliding window to minimize the load imbalance across outfeeds. The decision point of assigning a parcel is dynamic,
    based on the current state of the outfeeds and the parcels in the sliding window.
    """

    from DES_GoodCode_implemented import Event

    p = evt.parcel

    # 1) Greedy pick among currently‐free channels
    k0 = greedy_time(self, p)
    self.assignment[p.id] = k0
    if k0 is None:
        # It could physically fit on some outfeed, but none are free right now
        self.first_pass_failures.add(p.id)

    # 2) Append (arrival_time, p) to sliding window and evict old entries
    self.window.append((evt.time, p))
    while self.window and (evt.time - self.window[0][0] > self.WINDOW_DURATION):
        self.window.popleft()

    # 3) Every REBALANCE_INTERVAL arrivals, run local search
    self.rebal_ctr += 1
    if self.rebal_ctr >= self.REBALANCE_INTERVAL:
        run_local_search_time(self)
        self.rebal_ctr = 0

    # 4) Use (possibly rebalanced) assignment to schedule next step
    t = evt.time
    final_k = self.assignment[p.id]
    if final_k is None:
        # Recirculate: schedule ENTER_SCANNER again after full loop back time
        self.recirculated_count += 1
        dt = timedelta(seconds=(self.dist_outfeeds_to_infeeds / self.belt_speed))
        fes.add(Event(Event.RECIRCULATE, t + dt, p))
    else:
        # Schedule ENTER_OUTFEED: travel time from scanner to gate final_k
        dt = (self.dist_scanner_to_outfeeds + final_k * self.dist_between_outfeeds) / self.belt_speed
        dt_gate = timedelta(seconds=dt)
        fes.add(Event(Event.ENTER_OUTFEED, t + dt_gate, p, outfeed_id=final_k))

def load_balance_time(parcel):
    """
    Marker function: when main.py sees sorting_algorithm == load_balance, it delegates
    to handle_enter_scanner(...) instead of FCFS/genetic/MLFS.
    The body can be empty or return None.
    """
    return None

def greedy_length(self, p):
    """
    Initial assignment: choose feasible outfeed with minimal tracked load.

    Returns the index of the chosen outfeed, or None if no feasible outfeed can accept the parcel right now.
    """
    feas = [k for k in p.feasible_outfeeds if (self.outfeeds[k].can_accept(p))]
    if not feas:
        return None
    return min(feas, key=lambda k: self.loads_l[k])


def imbalance_length(self, loads_l):
    """
    Compute objective: spread between heaviest and lightest load.
    """
    return max(loads_l.values()) - min(loads_l.values())


def run_local_search_length(self, max_iters=100):
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
    # Hill-climbing to refine assignments in the sliding window
    loads_l = self.loads_l.copy()
    assign_wl = {}
    # Baseline: assign any unassigned parcels in window greedily
    for (_, p) in self.window:
        k = greedy_length(self, p)
        assign_wl[p.id] = k
        if k is not None:
            loads_l[k] += p.length

    # Iteratively try moves that reduce imbalance
    for _ in range(max_iters):
        improved = False
        for (_, p) in self.window:
            cur = assign_wl[p.id]
            for k in p.feasible_outfeeds:
                if k == cur or loads_l[k] + p.length > self.outfeeds[k].max_length:
                    continue
                new_loads_l = loads_l.copy()
                if cur is not None:
                    new_loads_l[cur] -= p.length
                new_loads_l[k] += p.length
                if imbalance_length(self, new_loads_l) < imbalance_length(self, loads_l):
                    loads_l, assign_wl[p.id] = new_loads_l, k
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    # Commit improved assignments
    for pid, k in assign_wl.items():
        self.assignment_l[pid] = k


def handle_enter_scanner_length(self, evt, fes):
        
    from DES_GoodCode_implemented import Event

    p = evt.parcel
    # 1) greedy assign
    k0 = greedy_length(self, p)
    self.assignment_l[p.id] = k0
    # track failure on first pass
    if k0 is None:
        self.first_pass_failures.add(p.id)
    # 2) Append (arrival_time, p) to sliding window and evict old entries
    self.window.append((evt.time, p))
    while self.window and (evt.time - self.window[0][0] > self.WINDOW_DURATION):
        self.window.popleft()
    
    # 3) Every REBALANCE_INTERVAL arrivals, run local search
    self.rebal_ctr += 1
    if self.rebal_ctr >= self.REBALANCE_INTERVAL:
        run_local_search_length(self)
        self.rebal_ctr = 0

    # 4) schedule
    t = evt.time
    final_k = self.assignment_l[p.id]
    if final_k is None:
        # Recirculate: schedule ENTER_SCANNER again after full loop back time
        self.recirculated_count += 1
        dt = timedelta(seconds=(self.dist_outfeeds_to_infeeds / self.belt_speed))
        fes.add(Event(Event.RECIRCULATE, t + dt, p))
    else:
        # Schedule ENTER_OUTFEED: travel time from scanner to gate final_k
        dt = timedelta(seconds=(self.dist_scanner_to_outfeeds + final_k * self.dist_between_outfeeds) / self.belt_speed)
    
        fes.add(Event(Event.ENTER_OUTFEED, t + dt, p, outfeed_id=final_k))
        
def load_balance_length(parcel):
    """
    Marker function: when main.py sees sorting_algorithm == load_balance, it delegates
    to handle_enter_scanner(...) instead of FCFS/genetic/MLFS.
    The body can be empty or return None.
    """
    return None


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

    # Initial population
    population = [random.choice(feasible_outfeeds) for _ in range(population_size)]
    
    # Evaluate fitness
    for _ in range(generations):
        fitness_scores = [fitness(individual) for individual in population]

    # Selection 
        selected = []
        for _ in range(population_size):
            i1, i2 = random.sample(range(population_size), 2)
            selected.append(population[i1] if fitness_scores[i1] > fitness_scores[i2] else population[i2])

    # Crossover
        children = []
        for _ in range(population_size):
            p1, p2 = random.sample(selected, 2)
            child = random.choice([p1, p2])
            children.append(child)

    # Mutation 
        for i in range(population_size):
            if random.random() < mutation_rate:
                children[i] = random.choice(feasible_outfeeds)

        population = children

    # Best solution 
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
    ML-First-Serve with confidence threshold:
      1. If top prediction p>=threshold and feasible, try it first.
      2. Else if parcel.feasible_outfeeds nonempty, fall back to that list.
      3. Else rank all gates by ML probability, update feasible_outfeeds, and return.
    """
    if ml_model is None or not ml_model.is_trained:
        raise RuntimeError(
            "ml_model not initialized or not trained. "
            "Call initialize_ml_model(...) and then .fit(...) before using mlfs()."
        )

    # Extract probabilities and class labels
    feats   = ml_model.parcel_to_features(parcel).reshape(1, -1)
    proba   = ml_model.clf.predict_proba(feats)[0]   # e.g. [0.1,0.7,0.2]
    classes = ml_model.clf.classes_                 # e.g. [0,1,2]

    # Identify top class and its confidence
    idx_max = int(np.argmax(proba))
    p_max   = proba[idx_max]
    feas    = list(parcel.feasible_outfeeds)

    # 1) High-confidence & feasible → prioritized first
    if p_max >= ml_model.threshold and idx_max in feas:
        rest = [f for f in feas if f != idx_max]
        return deque([idx_max] + rest)

    # 2) Fallback to mechanical feasibility list
    if feas:
        return deque(feas)

    # 3) No mechanical options → rank all gates by descending prob
    ranked = [c for _, c in sorted(zip(proba, classes), key=lambda x: -x[0])]
    parcel.feasible_outfeeds = ranked
    return deque(ranked)