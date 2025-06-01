from collections import deque


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
    feas = [k for k in p.feasible_outfeeds if loads[k] + p.length <= outfeeds[k].max_length]
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

