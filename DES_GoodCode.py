import heapq
import pandas as pd
import random
from datetime import timedelta
from collections import deque

import sorting_algorithms as sa 

# --------------------------------------------------------------------------
# DATA CLEANING FUNCTIONS
# --------------------------------------------------------------------------
def remove_outliers_iqr(df, cols):
    for c in cols:
        Q1, Q3 = df[c].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df = df[df[c].between(Q1 - 2 * IQR, Q3 + 2 * IQR)]
    return df

def drop_rows_without_true_outfeed(df, prefix="Outfeed"):
    cols = [c for c in df.columns if c.startswith(prefix)]
    return df[df[cols].any(axis=1)] if cols else df

def clean_parcel_data(df):
    df = df.dropna().reset_index(drop=True)
    df = remove_outliers_iqr(df, ["Length", "Width", "Height"])
    df = drop_rows_without_true_outfeed(df)
    return df

def load_parcels_from_clean_df(df):
    parcels = []
    for _, r in df.iterrows():
        parcels.append(Parcel(
            pid=int(r["Parcel Number"]),
            arrival_time=pd.to_datetime(r["Arrival Time"]),
            length=float(r["Length"]),
            width=float(r["Width"]),
            height=float(r["Height"]),
            weight=float(r["Weight"]),
            feasible=[i for i, f in enumerate(
                [r["Outfeed 1"], r["Outfeed 2"], r["Outfeed 3"]]) if f]
        ))
    return sorted(parcels, key=lambda p: p.arrival_time)

# --------------------------------------------------------------------------
# EVENT, FES, PARCEL
# --------------------------------------------------------------------------
class Event:
    ARRIVAL = 0
    ENTER_SCANNER = 1
    ENTER_OUTFEED = 2
    EXIT_OUTFEED = 3
    RECIRCULATE = 4

    def __init__(self, typ, time, parcel, outfeed_id=None):
        self.type = typ
        self.time = time
        self.parcel = parcel
        self.outfeed_id = outfeed_id

    def __lt__(self, other):
        return self.time < other.time

class FES:
    def __init__(self):
        self.events = []

    def add(self, event):
        heapq.heappush(self.events, event)

    def next(self):
        return heapq.heappop(self.events)

    def isEmpty(self):
        return len(self.events) == 0

class Parcel:
    def __init__(self, pid, arrival_time, length, width, height, weight, feasible):
        self.id = pid
        self.arrival_time = arrival_time
        self.length = length
        self.width = width
        self.height = height
        self.weight = weight
        self.feasible_outfeeds = feasible
        self.recirculation_count = 0

    def get_volume(self):
        return self.length * self.width * self.height

# --------------------------------------------------------------------------
# OUTFEED MODEL
# --------------------------------------------------------------------------
def compute_outfeed_time(parcel):
    base_time = 4.5
    volume = parcel.get_volume()
    if volume < 0.035:
        vol_delay = 0
    elif volume < 0.055:
        vol_delay = 1
    else:
        vol_delay = 2

    weight = parcel.weight
    if weight < 1700:
        wt_delay = 0
    elif weight < 2800:
        wt_delay = 1
    else:
        wt_delay = 2
    return base_time + vol_delay + wt_delay

class Outfeed:
    def __init__(self, max_length=3.0):
        self.max_length = max_length
        self.current_length = 0.0
        self.queue = []
        self.next_time = 0.0

    def add_parcel(self, parcel):
        t = compute_outfeed_time(parcel)
        self.queue.append((parcel, t))
        self.current_length += parcel.length
        if len(self.queue) == 1:
            self.next_time = t

    def update(self, dt):
        self.next_time -= dt
        if self.next_time <= 0 and self.queue:
            p, _ = self.queue.pop(0)
            self.current_length -= p.length
            if self.queue:
                self.next_time = self.queue[0][1]

# --------------------------------------------------------------------------
# POSISORTER with External Sorting + Recirc Cap
# --------------------------------------------------------------------------
class PosiSorterSystem:
    WINDOW_SIZE = 10
    REBALANCE_INTERVAL = 1
    MAX_RECIRCULATIONS = 3    # give up after 3 failed recircs

    def __init__(self, layout_df):
        L = layout_df.set_index("Layout property")["Value"]
        self.belt_speed = L["Belt Speed"]
        self.d_in_sc = L["Distance Infeeds to Scanner"]
        self.d_sc_of = L["Distance Scanner to Outfeeds"]
        self.d_between = L["Distance between Outfeeds"]
        self.d_of_in = L["Distance Outfeeds to Infeeds"]
        self.num_outfeeds = 3
        self.outfeeds = [Outfeed() for _ in range(self.num_outfeeds)]

        # stats & state
        self.recirculated_count = 0
        self.outfeed_counts = [0] * self.num_outfeeds
        self.first_pass_failures = set()
        self.loads = {k: 0.0 for k in range(self.num_outfeeds)}
        self.assignment = {}
        self.window = deque(maxlen=self.WINDOW_SIZE)
        self.rebal_ctr = 0

    def handle_enter_scanner(self, evt, fes):
        p = evt.parcel

        # 1) initial greedy assignment
        k0 = sa.greedy(p, self.loads, self.outfeeds)
        self.assignment[p.id] = k0
        if k0 is None:
            self.first_pass_failures.add(p.id)

        # 2) windowed local search
        self.window.append(p)
        self.rebal_ctr += 1
        if self.rebal_ctr >= self.REBALANCE_INTERVAL:
            sa.run_local_search(self.window, self.loads, self.outfeeds, self.assignment)
            self.rebal_ctr = 0

        # 3) schedule next
        t = evt.time
        if self.assignment[p.id] is None:
            p.recirculation_count += 1
            # cap recirculations
            if p.recirculation_count <= self.MAX_RECIRCULATIONS:
                self.recirculated_count += 1
                dt = (self.d_sc_of + self.d_between * self.num_outfeeds) / self.belt_speed
                fes.add(Event(Event.RECIRCULATE, t + dt, p))
            else:
                print(f"[{t:.1f}s] Dropping parcel {p.id} after {p.recirculation_count} recircs")
        else:
            k = self.assignment[p.id]
            dt = (self.d_sc_of + k * self.d_between) / self.belt_speed
            fes.add(Event(Event.ENTER_OUTFEED, t + dt, p, outfeed_id=k))

    def simulate(self, parcels):
        fes = FES()
        t0 = parcels[0].arrival_time
        for p in parcels:
            t = (p.arrival_time - t0).total_seconds()
            fes.add(Event(Event.ARRIVAL, t, p))

        while not fes.isEmpty():
            evt = fes.next()
            t = evt.time

            if evt.type == Event.ARRIVAL:
                dt = self.d_in_sc / self.belt_speed
                fes.add(Event(Event.ENTER_SCANNER, t + dt, evt.parcel))

            elif evt.type == Event.ENTER_SCANNER:
                self.handle_enter_scanner(evt, fes)

            elif evt.type == Event.ENTER_OUTFEED:
                k, p = evt.outfeed_id, evt.parcel
                f = self.outfeeds[k]
                f.add_parcel(p)
                self.outfeed_counts[k] += 1
                self.loads[k] += p.length
                if len(f.queue) == 1:
                    fes.add(Event(Event.EXIT_OUTFEED, t + f.queue[0][1], p, outfeed_id=k))

            elif evt.type == Event.EXIT_OUTFEED:
                k, p = evt.outfeed_id, evt.parcel
                f = self.outfeeds[k]
                f.update(f.next_time)
                self.loads[k] -= p.length
                actual_time = t0 + timedelta(seconds=t)
                print(f"[{actual_time.time()}] Parcel {p.id} removed from outfeed {k}")
                if f.queue:
                    fes.add(Event(Event.EXIT_OUTFEED, t + f.queue[0][1], f.queue[0][0], outfeed_id=k))

            elif evt.type == Event.RECIRCULATE:
                dt = (self.d_of_in + self.d_in_sc) / self.belt_speed
                fes.add(Event(Event.ENTER_SCANNER, t + dt, evt.parcel))

        # summary
        total = len(parcels)
        sorted_total = sum(self.outfeed_counts)
        success_rate = (total - len(self.first_pass_failures)) / total * 100
        print("\n--- Simulation Summary ---")
        print(f"Total parcels:           {total}")
        print(f"Recirculated (count):    {self.recirculated_count}")
        print(f"Success rate (1st pass): {success_rate:.2f}%")
        for i, cnt in enumerate(self.outfeed_counts):
            pct = cnt / sorted_total * 100 if sorted_total else 0
            print(f" Outfeed {i}: {cnt} parcels ({pct:.1f}%)")
        print(f"Throughput (all events): {sorted_total + self.recirculated_count}")

# ----------------------------------------------------------------------------
# MAIN ENTRY
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    xls = pd.ExcelFile("PosiSorterData1.xlsx")
    df_p = clean_parcel_data(xls.parse("Parcels"))
    df_l = xls.parse("Layout")
    parcels = load_parcels_from_clean_df(df_p)
    system = PosiSorterSystem(df_l)
    system.simulate(parcels)
