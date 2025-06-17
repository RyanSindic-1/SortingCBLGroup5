# algorithm.py

from collections import deque
import random
from datetime import timedelta
import numpy as np

# ── IMPORT THE NEW ML MODULE ────────────────────────────────────────────────────
from ml_sorting import OutfeedML

# ────────────────────────────────────────────────────────────────────────────────

def LBSL_with_logging(parcel, system):
    """
    Use genetic() to pick the best gate *and* record the feature→label pair
    for later ML training.
    """
    # 1) extract features
    feats = ml_model.extract_features(parcel, system)
    # 2) get the “ground-truth” label
    label = load_balance_length(parcel)
    # 3) record it
    system.training_data.append((feats, label))
    # 4) return exactly what LBSL() would
    return label
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
    from DES_GoodCode_implemented import compute_outfeed_time

    # 1) Start from current global loads (excluding future actuals)
    time_loads = self.loads.copy()
    length_loads = self.loads_l.copy()
    assign_w = {}

    # 2) Initial greedy assignment using predicted service times
    for (_, p) in self.window:
        st = compute_outfeed_time(p)
        k = greedy_time(self, p)
        assign_w[p.id] = k
        if k is not None:
            time_loads[k]   += st
            length_loads[k] += p.length

    # 3) Hill-climb: try moving parcels to reduce time imbalance
    for _ in range(max_iters):
        improved = False
        for (_, p) in self.window:
            cur = assign_w[p.id]
            st = compute_outfeed_time(p)
            for k in p.feasible_outfeeds:
                if k == cur:
                    continue
                # enforce length capacity
                if length_loads[k] + p.length > self.outfeeds[k].max_length:
                    continue
                # simulate move
                new_time = time_loads.copy()
                new_len  = length_loads.copy()
                if cur is not None:
                    new_time[cur]   -= st
                    new_len[cur]    -= p.length
                new_time[k]       += st
                new_len[k]        += p.length
                # accept if improves imbalance
                if imbalance_time(self, new_time) < imbalance_time(self, time_loads):
                    time_loads, length_loads = new_time, new_len
                    assign_w[p.id] = k
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # 4) Commit back into system state
    self.loads   = time_loads
    self.loads_l = length_loads
    for pid, k in assign_w.items():
        self.assignment[pid] = k


def handle_enter_scanner_time(self, evt, fes):
    """
    This function handles the ENTER_SCANNER event.
    It assigns the parcel to an outfeed based on a greedy strategy,
    after that it updates the sliding window of recent parcels and performs local search
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
    # 1) copy current length loads and prepare assignment map
    length_loads = self.loads_l.copy()
    assign_wl = {}

    # 2) initial greedy assignment based on minimal length-load
    for (_, p) in self.window:
        k = greedy_length(self, p)
        assign_wl[p.id] = k
        if k is not None:
            length_loads[k] += p.length

    # 3) hill-climb to reduce max-min length imbalance
    for _ in range(max_iters):
        improved = False
        for (_, p) in self.window:
            cur = assign_wl[p.id]
            for k in p.feasible_outfeeds:
                # skip same or infeasible by length
                if k == cur or length_loads[k] + p.length > self.outfeeds[k].max_length:
                    continue
                # simulate move
                new_loads = length_loads.copy()
                if cur is not None:
                    new_loads[cur] -= p.length
                new_loads[k] += p.length
                if imbalance_length(self, new_loads) < imbalance_length(self, length_loads):
                    length_loads = new_loads
                    assign_wl[p.id] = k
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # 4) commit updated loads and assignments
    self.loads_l = length_loads
    for pid, k in assign_wl.items():
        self.assignment_l[pid] = k


def handle_enter_scanner_length(self, evt, fes):
    """ This function handles the ENTER_SCANNER event for length-based load balancing with local search 
    and basically does the same as handle_enter_scanner_time, but uses length instead of time for load balancing."""   
    from DES_GoodCode_implemented import Event

    p = evt.parcel

    # pick the gate exactly as the handler would
    k0 = greedy_length(self, p)

    
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

    feat = self.ml_model.extract_features(p, self)
    self.training_data.append((feat, final_k))
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

def load_balance_time_simple(parcel):
    """time-based load balancing without local search."""
    return None

def load_balance_length_simple(parcel):
    """length-based load balancing without local search."""
    return None

def handle_enter_scanner_time_simple(self, evt, fes):
    """ This function handles the ENTER_SCANNER event for time-based load balancing without local seaerch."""
    from DES_GoodCode_implemented import Event

    p = evt.parcel
    # Greedy based on current time-loads
    k0 = greedy_time(self, p)
    if k0 is None:
        # No feasible outfeed can accept the parcel right now, so we recirculate it.
        self.recirculated_count += 1
        dt = timedelta(seconds=(self.dist_outfeeds_to_infeeds / self.belt_speed))
        fes.add(Event(Event.RECIRCULATE, evt.time + dt, p))
    else:
        # Schedule the ENTER_OUTFEED event for the chosen outfeed.
        dt = timedelta(seconds=(self.dist_scanner_to_outfeeds + k0 * self.dist_between_outfeeds) / self.belt_speed)
        fes.add(Event(Event.ENTER_OUTFEED, evt.time + dt, p, outfeed_id=k0))

def handle_enter_scanner_length_simple(self, evt, fes):
    """ This function handles the ENTER_SCANNER event for length-based load balancing without local seaerch."""
    from DES_GoodCode_implemented import Event

    p = evt.parcel
    # Greedy based on current length-loads
    k0 = greedy_length(self, p)
    if k0 is None:
        # No feasible outfeed can accept the parcel right now, so we recirculate it.
        self.recirculated_count += 1
        dt = timedelta(seconds=(self.dist_outfeeds_to_infeeds / self.belt_speed))
        fes.add(Event(Event.RECIRCULATE, evt.time + dt, p))
    else:
        # Schedule the ENTER_OUTFEED event for the chosen outfeed.
        dt = timedelta(seconds=(self.dist_scanner_to_outfeeds + k0 * self.dist_between_outfeeds) / self.belt_speed)
        fes.add(Event(Event.ENTER_OUTFEED, evt.time + dt, p, outfeed_id=k0))


def fcfs(parcel) -> deque[int]:
    """
    First-Come-First-Serve algorithm for assigning parcels to outfeeds.
    :param parcel: parcel object.
    :return: a double-ended queue containing the ordered outfeeds ID's.
    """
    return deque(parcel.feasible_outfeeds.copy())


def genetic(parcel, system, population_size=50, generations=100, mutation_rate=0.1) -> int:
    """
    Genetic algorithm to choose the optimal outfeed for a parcel.
    :param parcel: Parcel object.
    :param system: the PosiSorterSystem.
    :return: ID of optimal outfeed.
    """
    feasible_outfeeds = parcel.feasible_outfeeds

    def fitness(outfeed_id) -> float:
        """
        Assigns fitness based on current buffer content in the outfeed.
        :return: the fitness as a float.
        """
        current_load = system.loads_l[outfeed_id]
        return 1 / (1 + current_load)  # Avoid division by zero

    # Initial population
    population = [random.choice(feasible_outfeeds) for _ in range(population_size)]

    # Evaluate fitness (ranking)
    for _ in range(generations):
        fitness_scores = [(outfeed_id, fitness(outfeed_id)) for outfeed_id in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)  # Higher fitness first

        # Selection
        selected = [id for id, _ in fitness_scores[:population_size // 2]]

        # Crossover
        children = []
        while len(children) < population_size - len(selected):
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            # Crossover strategy: randomly pick one of the two parents
            child = random.choice([parent1, parent2])
            children.append(child)

        # Mutation
        for i in range(len(children)):
            if random.random() < mutation_rate:
                children[i] = random.choice(feasible_outfeeds)

        # New generation
        population = selected + children

    # Return best outfeed from final population
    best_outfeed = max(population, key=fitness)
    return best_outfeed


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




def mlfs(parcel, system) -> deque[int]:
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
    feats = ml_model.extract_features(parcel, system).reshape(1, -1)
    proba   = ml_model.clf.predict_proba(feats)[0]   # e.g. [0.1,0.7,0.2]
    classes = ml_model.clf.classes_                 # e.g. [0,1,2]

    label_proba = { gate: p for gate, p in zip(classes, proba) }
    # Identify top class and its confidence
    idx_max = int(np.argmax(proba))
    p_max   = proba[idx_max]
    feas    = list(parcel.feasible_outfeeds)
    feas_conf = [( label_proba.get(g, 0.0), g ) for g in feas]


    print(f"[DEBUG] parcel {parcel.id}: proba={proba}, top_idx={idx_max}, top_p={p_max:.3f}, feas={feas}")
    
    if feas_conf:
        # 3) pick best among feasible
        best_p, best_gate = max(feas_conf, key=lambda x: x[0])
        # 4) if it clears the threshold, bump it to front
        if best_p >= ml_model.threshold:
            rest = [g for g in feas if g != best_gate]
            return deque([best_gate] + rest)
        # 5) otherwise fall back to full ML-ranked feasible ordering
        #    or even FCFS order—your choice
        #    Here: full ML ranking among feas
        ranked = [g for _,g in sorted(feas_conf, key=lambda x: -x[0])]
        return deque(ranked)
        
    # 6) if no mechanical options, fall back to ranking all gates by ML
    ranked_all = [c for _, c in sorted(zip(proba, classes), key=lambda x: -x[0])]
    parcel.feasible_outfeeds = ranked_all
    return deque(ranked_all)