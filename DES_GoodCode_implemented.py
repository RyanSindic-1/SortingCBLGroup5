# main.py

import heapq
import time
import pandas as pd
import random
from datetime import timedelta 
from sorting_algorithms_implemented import (fcfs, genetic, initialize_ml_model, mlfs, 
handle_enter_scanner_time, load_balance_time, handle_enter_scanner_length, 
load_balance_length, load_balance_length_simple, load_balance_time_simple, 
handle_enter_scanner_length_simple, handle_enter_scanner_time_simple
)
import math
from collections import deque

from data_cleaning import (
    drop_non_chronological_arrivals,
    remove_outliers_iqr,
    drop_rows_without_true_outfeed,
    clean_parcel_data,
    load_parcels_from_clean_df
)


# -------------------------------------------------------------------------- #
# EVENT CLASS                                                                #  
# -------------------------------------------------------------------------- #

start = time.time()

class Event:
    """
    A class that represents events.
    """
    ARRIVAL = 0
    ENTER_SCANNER = 1
    ENTER_OUTFEED = 2
    EXIT_OUTFEED = 3 
    RECIRCULATE = 4

    def __init__(self, typ, time, parcel, outfeed_id = None) -> None:
        self.type = typ
        self.time = time
        self.parcel = parcel
        self.outfeed_id = outfeed_id

    def __lt__(self, other) -> bool:
        return self.time < other.time


class FES:
    """
    Future Event Set implemented as a min-heap of Event objects.
    """
    def __init__(self) -> None:
        self.events = []

    def add(self, event) -> None:
        heapq.heappush(self.events, event)

    def next(self) -> Event:
        return heapq.heappop(self.events)

    def isEmpty(self) -> bool:
        return len(self.events) == 0


class Parcel:
    """
    A class to represent a parcel. 
    """
    def __init__(self, parcel_id, arrival_time, length, width, height, weight, feasible_outfeeds) -> None:
        self.id = parcel_id
        self.arrival_time = arrival_time
        self.length = length
        self.width = width
        self.height = height
        self.weight = weight
        self.feasible_outfeeds = feasible_outfeeds
        self.sorted = False
        self.recirculated = False
        self.outfeed_attempts = []  
        self.recirculation_count = 0  
        self.sorted_first_try = False

    def get_volume(self) -> float: 
        return self.length * self.width * self.height


def compute_outfeed_time(parcel) -> float:
    """
    Determines the amount of time it costs to get unloaded from the outfeed.
    """
    base_time = 4.5

    # Volume classes
    volume = parcel.get_volume()
    if volume < 0.035:
        volume_class_delay = 0
    elif volume < 0.055:
        volume_class_delay = 1
    else:
        volume_class_delay = 2

    # Weight classes
    weight = parcel.weight
    if weight < 1700:
        weight_class_delay = 0
    elif weight < 2800:
        weight_class_delay = 1
    else:
        weight_class_delay = 2

    return base_time + volume_class_delay + weight_class_delay


class Outfeed:
    """
    A class to represent the outfeed.
    """
    def __init__(self, max_length: float) -> None:
        self.max_length = max_length
        self.current_parcels = []
        self.current_length = 0.0
        self.time_until_next_discharge = 0.0

    def can_accept(self, parcel) -> bool:
        return self.current_length + parcel.length <= self.max_length

    def add_parcel(self, parcel) -> None:
        self.current_parcels.append((parcel, compute_outfeed_time(parcel)))
        self.current_length += parcel.length
        if len(self.current_parcels) == 1:
            self.time_until_next_discharge = self.current_parcels[0][1]

    def update(self, time_step) -> None:
        if self.current_parcels:
            self.time_until_next_discharge -= time_step
            if self.time_until_next_discharge <= 0:
                parcel, _ = self.current_parcels.pop(0)
                self.current_length -= parcel.length
                if self.current_parcels:
                    self.time_until_next_discharge = self.current_parcels[0][1]


class PosiSorterSystem:
    """
    A class to represent the sorter system. 
    """
    def __init__(self, layout_df, num_outfeeds, sorting_algorithm) -> None:
        self.belt_speed                = layout_df.loc[
            layout_df['Layout property'] == 'Belt Speed', 'Value'
        ].values[0]
        self.dist_infeeds_to_scanner   = layout_df.loc[
            layout_df['Layout property'] == 'Distance Infeeds to Scanner', 'Value'
        ].values[0]
        self.dist_scanner_to_outfeeds  = layout_df.loc[
            layout_df['Layout property'] == 'Distance Scanner to Outfeeds', 'Value'
        ].values[0]
        self.dist_between_outfeeds     = layout_df.loc[
            layout_df['Layout property'] == 'Distance between Outfeeds', 'Value'
        ].values[0]
        outfeed_lengths = (
        layout_df[layout_df["Layout property"].str.startswith("Length outfeed")]
        .sort_values("Layout property")["Value"]
        .tolist()
)

        possible_keys = ["Distance Outfeeds to Infeeds", "Distance Infeeds to Arrival"]
        match = layout_df[layout_df['Layout property'].isin(possible_keys)]
        if not match.empty:
            self.dist_outfeeds_to_infeeds = match['Value'].values[0]
        else:
            self.dist_outfeeds_to_infeeds = None

        self.num_outfeeds = num_outfeeds
        # self.outfeeds = [Outfeed(max_length=4.5) for _ in range(self.num_outfeeds)]
        self.outfeeds = [Outfeed(max_length=outfeed_lengths[k]) for k in range(self.num_outfeeds)]
        self.sorting_algorithm = sorting_algorithm

        # For statistics
        self.recirculated_count = 0
        self.outfeed_counts = [0] * self.num_outfeeds
        self.non_sorted_parcels = 0
    
        # ─── LOAD‐BALANCING STATE ───────────────────────────────────────────────
        # track “sum of service times” on each channel
        self.loads = {k: 0.0 for k in range(self.num_outfeeds)}
        # track "load of the total lenght in each outfeed"
        self.loads_l = {k: 0.0 for k in range(self.num_outfeeds)}
        # track service times of parcels (parcel.id → service_time)
        self.service_times = {}
        # track which outfeed each parcel was assigned to (parcel.id → outfeed_id)
        self.assignment = {}
        self.assignment_l = {}
    
        self.WINDOW_DURATION = timedelta(seconds=(self.dist_scanner_to_outfeeds / self.belt_speed))
        self.window = deque()
        # how many arrivals until we run one local‐search
        self.REBALANCE_INTERVAL = 1
        self.rebal_ctr = 0
        # record parcels that had to recirculate because no outfeed was free “this round”
        self.first_pass_failures = set()
    def handle(self, evt):
        pass
    
    def simulate(self, parcels) -> None:
        fes = FES()
        t = timedelta(0)
        t0 = parcels[0].arrival_time

        last_arrival_time = timedelta(0)
        safety_spacing = 0.3    
        int_arrival_safety_time = timedelta(seconds=safety_spacing / self.belt_speed)

        # Schedule all ARRIVAL events up front
        for p in parcels:
            proposed_arrival_time = p.arrival_time - t0
            if proposed_arrival_time < last_arrival_time + int_arrival_safety_time:
                proposed_arrival_time = last_arrival_time + int_arrival_safety_time
            arrival_event = Event(Event.ARRIVAL, proposed_arrival_time, p)
            fes.add(arrival_event)
            last_arrival_time = proposed_arrival_time

        while not fes.isEmpty():
            tOld = t
            evt = fes.next()
            self.handle(evt)
            t = evt.time

            if evt.type == Event.ARRIVAL:
                parcel = evt.parcel  
                time_to_scanner = timedelta(seconds=self.dist_infeeds_to_scanner / self.belt_speed)
                scan = Event(Event.ENTER_SCANNER, t + time_to_scanner, parcel)
                fes.add(scan)

            elif evt.type == Event.ENTER_SCANNER:
                parcel = evt.parcel

                # If “load_balance” is chosen as the sorting_algorithm, jump to Load balancing code. You can choose between length based ot time based load balancing:
                if self.sorting_algorithm is load_balance_time:
                    handle_enter_scanner_time(self, evt, fes)
                elif self.sorting_algorithm is load_balance_length:
                    handle_enter_scanner_length(self, evt, fes)
                elif self.sorting_algorithm is load_balance_length_simple:
                    handle_enter_scanner_length_simple(self, evt, fes)
                elif self.sorting_algorithm is load_balance_time_simple:
                    handle_enter_scanner_time_simple(self, evt, fes)
                else:
                    # Otherwise, run the other chosen sorting algorithm:
                    choice = self.sorting_algorithm(parcel)
                    if isinstance(choice, int):
                        parcel.outfeed_attempts = deque([choice])
                    else:
                        parcel.outfeed_attempts = deque(choice)

                    first_choice = parcel.outfeed_attempts.popleft()
                    time_to_outfeed = timedelta(
                    seconds = (self.dist_scanner_to_outfeeds + first_choice * self.dist_between_outfeeds) / self.belt_speed
                    )
                    fes.add(Event(Event.ENTER_OUTFEED, t + time_to_outfeed, parcel, outfeed_id=first_choice))


            elif evt.type == Event.ENTER_OUTFEED:
                k      = evt.outfeed_id
                feed   = self.outfeeds[k]
                parcel = evt.parcel
                wall_clock = (t0 + evt.time).time()

                if not feed.can_accept(parcel):
                    # Try next available outfeed (if any)
                    if parcel.outfeed_attempts:
                        next_k = parcel.outfeed_attempts.popleft()
                        time_to_next = timedelta(seconds=self.dist_between_outfeeds / self.belt_speed)
                        fes.add(Event(Event.ENTER_OUTFEED, t + time_to_next, parcel, outfeed_id=next_k))
                    else:
                        last_outfeed = parcel.feasible_outfeeds[-1]
                        # No options left → recirculate
                        self.recirculated_count += 1
                        time_start_recirculation = timedelta(
                            seconds=(self.dist_between_outfeeds * (self.num_outfeeds - last_outfeed)) / self.belt_speed
                        )
                        fes.add(Event(Event.RECIRCULATE, t + time_start_recirculation, parcel))
                    continue

                # If feed can accept:
                feed.add_parcel(parcel)
                self.outfeed_counts[k] += 1

                # Record service time and upload tracker
                service_time = feed.current_parcels[-1][1]
                self.service_times[parcel.id] = service_time
                self.loads[k] += service_time
                self.loads_l[k] += parcel.length

                if len(feed.current_parcels) == 1:
                    theta = 25  # degrees
                    theta_rad = math.radians(theta)
                    g = 9.81
                    mu = 0.03
                    a = g * (math.sin(theta_rad) - mu * math.cos(theta_rad))
                    time_to_bottom = math.sqrt((2 * feed.max_length) / a)
                    discharge_time = timedelta(seconds=feed.current_parcels[0][1] + time_to_bottom)

                    fes.add(Event(Event.EXIT_OUTFEED, t + discharge_time, parcel, outfeed_id=k))


            elif evt.type == Event.EXIT_OUTFEED:
                k    = evt.outfeed_id
                parcel = evt.parcel
                self.assignment[parcel.id] = k + 1 # Store outfeed assignment (1-indexed)
                feed = self.outfeeds[k]
                wall_clock = (t0 + evt.time).time()
                if parcel.recirculation_count == 0:
                    parcel.sorted_first_try = True
                print(f"[{wall_clock}] Parcel {parcel.id} removed from outfeed {k}")

                feed.update(feed.time_until_next_discharge)
                self.loads[k] -= self.service_times.pop(parcel.id)
                self.loads_l[k] -= parcel.length
                if feed.current_parcels:
                    discharge_time = timedelta(seconds=feed.current_parcels[0][1])
                    next_parcel = feed.current_parcels[0][0]
                    fes.add(Event(Event.EXIT_OUTFEED, t + discharge_time, next_parcel, outfeed_id=k))


            elif evt.type == Event.RECIRCULATE:
                parcel = evt.parcel
                # ── NEW: log the recirculation event ───────────────────────────────────
                wall_clock = (t0 + evt.time).time()          # convert sim-time → clock-time
                attempt    = parcel.recirculation_count + 1  # about to increment below
                print(
                    f"[{wall_clock}] Parcel {parcel.id} with feasible outfeeds "
                    f"{parcel.feasible_outfeeds} has been recirculated (attempt {attempt})."
                )
                # ───────────────────────────────────────────────────────────────────────
                if parcel.recirculation_count < 3:
                    parcel.recirculation_count += 1
                    time_to_arrival = timedelta(seconds=self.dist_outfeeds_to_infeeds / self.belt_speed)
                    proposed_arrival_time = t + time_to_arrival

                    if proposed_arrival_time < last_arrival_time + int_arrival_safety_time:
                        proposed_arrival_time = last_arrival_time + int_arrival_safety_time

                    arrival = Event(Event.ARRIVAL, proposed_arrival_time, parcel)
                    fes.add(arrival)
                    last_arrival_time = proposed_arrival_time
                else:
                    print(f"[{wall_clock}] Parcel {parcel.id} discarded after 3 recirculations.")
                    self.non_sorted_parcels += 1

        end = time.time()

        # Print statistics
        print("\n--- Simulation Summary ---")
        print(f"Total parcels processed: {len(parcels)}")
        print(f"Parcels recirculated: {self.recirculated_count}")
        for i, count in enumerate(self.outfeed_counts):
            print(f"Parcels sent to Outfeed {i}: {count}")
        print(f"Parcels not sorted (recirculated 3 times): {self.non_sorted_parcels}")
        print(f"Throughput (sorted): {sum(self.outfeed_counts)}")
        print(f"Sorting success rate: (incl. recirculation) {((len(parcels) - self.non_sorted_parcels)  / len(parcels) * 100):.2f}% ")
        sorted_first_try_count = sum(1 for p in parcels if p.sorted_first_try)
        print(f"Sorting success rate (on first try): {( sorted_first_try_count / len(parcels) * 100):.2f}% ")
        print(f"Run time: {(end - start): .4f} seconds")


# -------------------------------------------------------------------------- #
# RUN SIMULATION                                                             #
# -------------------------------------------------------------------------- #

def main():
    # 1. LOAD & CLEAN DATA
    xlsx_path = pd.ExcelFile(r"C:\Users\20234607\OneDrive - TU Eindhoven\Y2\Q4\CBL\Code\Useful code files\PosiSorterData2(1).xlsx")
    xls = pd.ExcelFile(xlsx_path)
    parcels_df = xls.parse('Parcels')
    layout_df = xls.parse('Layout')

    parcels_df, drop_info = clean_parcel_data(parcels_df)
    parcels, num_outfeeds = load_parcels_from_clean_df(parcels_df)

    # 2. INITIALIZE & (OPTIONALLY) TRAIN / LOAD ML MODEL
    #    ───────────────────────────────────────────────────────────────────────
    #    If you have a pretrained model, specify its path. Otherwise, pass None:
    model_path = None  # e.g. "outfeed_model.pkl" if you saved earlier
    ml = initialize_ml_model(model_path)

    if model_path is None:
        # Quick example: train on some tiny synthetic dataset.
        # In practice, replace this with real historical labels:
        example_parcels = random.sample(parcels, min(len(parcels), 200))
        X_train = [ml.parcel_to_features(p) for p in example_parcels]
        # For labels, let’s pretend “true” outfeed = the first feasible_outfeed for each parcel:
        raw_sample = random.sample(parcels, min(len(parcels), 200))
        # Keep only those with nonempty feasible list
        example_parcels = [p for p in raw_sample if p.feasible_outfeeds]
        y_train          = [p.feasible_outfeeds[0] for p in example_parcels]
        if not example_parcels:
            raise RuntimeError("All sampled parcels lack feasible_outfeeds; aborting ML training.")
        ml.fit(example_parcels, y_train)


    # 3. CHOOSE WHICH ALGORITHM TO USE:
    #    ───────────────────────────────────────────────────────────────────────
    #    Comment/uncomment whichever one you want:
    
    #sorting_algo = fcfs
    #sorting_algo = genetic
    #sorting_algo = mlfs
    sorting_algo = load_balance_time
    #sorting_algo = load_balance_length
    #sorting_algo = load_balance_length_simple
    #sorting_algo = load_balance_time_simple


    system = PosiSorterSystem(layout_df, num_outfeeds, sorting_algorithm=sorting_algo)
    system.simulate(parcels)

        # ------------------------------------------------------------------
    # 4.  WRITE A NEW WORKBOOK WITH THE EXTRA “Simulated Outfeed” COLUMN
    # ------------------------------------------------------------------
    outfeed_col = parcels_df["Parcel Number"].map(system.assignment)     # aligns by ID
    parcels_df_with_outfeed = parcels_df.copy()
    parcels_df_with_outfeed["Simulated Outfeed"] = outfeed_col

    in_path  = r"C:\Users\20234607\OneDrive - TU Eindhoven\Y2\Q4\CBL\Code\Useful code files\PosiSorterData2(1).xlsx"        # <─ plain string
    xls      = pd.ExcelFile(in_path)         # use for reading
    out_path = in_path.replace(
    ".xlsx", "_with_sim_outfeeds.xlsx")  # works because in_path is str


    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
        # Overwrite Parcels sheet with the augmented dataframe
        parcels_df_with_outfeed.to_excel(writer, sheet_name="Parcels", index=False)
        # Copy the other sheets exactly as they were
        layout_df.to_excel(writer, sheet_name="Layout",  index=False)
        # (Add more sheets here if your original file contains them)

        
if __name__ == "__main__":
    main()