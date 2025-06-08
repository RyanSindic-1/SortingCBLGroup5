# algorithm.py

from collections import deque
import random
from datetime import timedelta

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

    from DES_GoodCode import Event

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
    if k0 is None:
        # Recirculate: schedule ENTER_SCANNER again after full loop back time
        self.recirculated_count += 1
        dt = timedelta(seconds=(self.dist_outfeeds_to_infeeds / self.belt_speed))
        fes.add(Event(Event.RECIRCULATE, t + dt, p))
    else:
        # Schedule ENTER_OUTFEED: travel time from scanner to gate final_k
        dt = (self.dist_scanner_to_outfeeds + k0 * self.dist_between_outfeeds) / self.belt_speed
        dt_gate = timedelta(seconds=dt)
        fes.add(Event(Event.ENTER_OUTFEED, t + dt_gate, p, outfeed_id=k0))

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


def handle_enter_scanner(self, evt, fes):
        p = evt.parcel
        
        k0 = greedy(self, p)
        self.assignment[p.id] = k0
        if k0 is None:
            # If it can fit but all channels are currently “full,” record first‐pass failure and schedule a recirculation below.
            self.first_pass_failures.add(p.id)

        # Update the sliding window with the current time and parcel
        self.window.append((evt.time, p))
        while self.window and (evt.time - self.window[0][0] > self.WINDOW_DURATION):
            self.window.popleft()

        self.rebal_ctr += 1
        if self.rebal_ctr >= self.REBALANCE_INTERVAL:
            self.run_local_search()
            self.rebal_ctr = 0

        # If we have a feasible outfeed, schedule the parcel to enter it
        t = evt.time
        if k0 is None:
            # It can physically fit on some outfeed eventually, but none are open right now.
            self.recirculated_count += 1
            dt = (self.d_sc_of + self.d_between * self.num_outfeeds) / self.belt_speed
            fes.add(Event(Event.RECIRCULATE, t + dt, p))
        else:
            # Schedule the time it takes to travel from scanner to the chosen outfeed gate
            dt = (self.d_sc_of + k0 * self.d_between) / self.belt_speed
            fes.add(Event(Event.ENTER_OUTFEED, t + dt, p, outfeed_id=k0))
        
def load_balance(parcel):
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


from collections import deque

def mlfs(parcel) -> deque[int]:
    """
    Machine-Learning First Serve:
      1. Predict one outfeed via ml_model.predict(parcel).
      2. If that ID is in parcel.feasible_outfeeds, try it first, then the rest.
      3. If not, fall back to the full feasible list.
      4. If feasible list is empty, rank all channels by ml_model.predict_proba(parcel),
         update parcel.feasible_outfeeds to that ranking, and return it.
    """
    if ml_model is None or not ml_model.is_trained:
        raise RuntimeError(
            "ml_model not initialized or not trained. "
            "Call initialize_ml_model(...) and then .fit(...) before using mlfs()."
        )

    chosen = ml_model.predict(parcel)
    feas = parcel.feasible_outfeeds

    # 1) If there *are* feasible channels, use them
    if feas:
        if chosen in feas:
            rest = [f for f in feas if f != chosen]
            return deque([chosen] + rest)
        return deque(feas)

    # 2) No feasible channels: rank *all* channels by predicted probability
    proba = ml_model.predict_proba(parcel)      # e.g. array([0.1, 0.7, 0.2])
    classes = ml_model.clf.classes_            # e.g. array([0,1,2])
    # Sort (probability, class) pairs descending, extract class IDs
    ranked = [c for _, c in sorted(zip(proba, classes), key=lambda x: -x[0])]

    # **Key fix:** update the parcel’s feasible_outfeeds so simulate() can
    # safely do `feasible_outfeeds[-1]` later.
    parcel.feasible_outfeeds = ranked

    return deque(ranked)
