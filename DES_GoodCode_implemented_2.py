# main.py
import numpy as np
import heapq
import time
import pandas as pd
import random
from datetime import timedelta 
from sorting_algorithms_implemented import (fcfs, genetic, initialize_ml_model, mlfs, 
handle_enter_scanner_time, load_balance_time, handle_enter_scanner_length, 
load_balance_length, load_balance_length_simple, load_balance_time_simple, 
handle_enter_scanner_length_simple, handle_enter_scanner_time_simple, LBSL_with_logging
)
import math
from collections import deque

from data_cleaning_2 import (
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
        """
        Initializes the attributes for the parcel class. 
        :param typ: what type of event it is, e.g, Arrival, scanner...
        :param time: at what time the event occurs
        :param parcel: all the information of the parcel that is being processed
        :param outfeed_id: the outfeed to which the parcel goes. It is an optional parameter since it is only needed for events 2 and 3.
        """
        self.type = typ
        self.time = time
        self.parcel = parcel
        self.outfeed_id = outfeed_id  # Only used for ENTER_OUTFEED and EXIT_OUTFEED events

    def __lt__(self, other) -> bool:
        """
        Compares this object with another object based on the 'time' attribute.
        :param other: Another thing in the class to compare with.
        :return: True if this object's time is less than the other object's time, False otherwise.
        """
        return self.time < other.time


class FES:
    """
    A class that represents a Future Event Set (FES) for discrete event simulation.
    This class uses a priority queue to manage events in the simulation.
    Events are sorted by their time attribute, which is a float.

    ...

    Methods
    -------
    def add(self, event) -> None:
        Adds an event to the Future Event Set
    
    def next(self) -> Event:
        Retrieves and removes the next event from the Future Event Set.
    
    def isEmpty(self) -> bool:
        Checks if the the Future Event Set is empty.
    """
    def __init__(self) -> None:
        """
        Initializes the attribute for the FES class.
        """
        self.events = []

    def add(self, event) -> None:
        """
        Adds an event to the Future Event Set.
        """
        heapq.heappush(self.events, event)

    def next(self) -> Event:
        """
        Retrieves and removes the next event from the Future Event Set.
        :return: the next event from the Future Event Set.
        """
        return heapq.heappop(self.events)

    def isEmpty(self) -> bool:
        """
        Checks if the the Future Event Set is empty.
        :return: True if the set is empty, False otherwise.
        """
        return len(self.events) == 0


class Parcel:
    """
    A class to represent a parcel. 

    ...
    
    Methods 
    -------
    get_volume(self) -> float:
        Calculates the volume of the parcel

    compute_outfeed_time(parcel) -> float:
        Determines the amount of time it costs to get unloaded from the outfeed.
    """
    def __init__(self, parcel_id, arrival_time, length, width, height, weight, feasible_outfeeds) -> None:
        """
        Initializes the attributes for the parcel class. 
        :param parcel_id: the parcel id number 
        :param arrival_time: arrival time of the parcel
        :param length: length of the parcel
        :param width: width of the parcel
        :param height: height of the parcel
        :param weight: weight of the parcel
        :param feasible_outfeeds: feasible outfeeds for each parcel
        """
        self.id = parcel_id
        self.arrival_time = arrival_time
        self.length = length
        self.width = width
        self.height = height
        self.weight = weight
        self.feasible_outfeeds = feasible_outfeeds
        self.sorted = False
        self.recirculated = False
        self.outfeed_attempts = []  # Afterwards, makes a copy of the feasible outfeeds of the parcel. Used for the algorithm functioning
        self.recirculation_count = 0  # Used to keep track of the number of recirculations of each parcel so that we can cap it at 3
        self.sorted_first_try = False

    def get_volume(self) -> float:
        """
        Calculates the volume of the parcel.
        :return: The volume of the parcel.
        """
        return self.length * self.width * self.height


def compute_outfeed_time(parcel) -> float:
    """
    Determines the amount of time it costs to get unloaded from the outfeed
    based on the volume and weight of the parcel.
    :param parcel: a parcel
    :return: the time it takes for the parcel to get unloaded from the outfeed.
    """
    base_time = 4.5

    # Volume classes
    volume = parcel.get_volume()  # Get the volume of parcels and catergorize it
    if volume < 0.035:
        volume_class_delay = 0
    elif volume < 0.055:
        volume_class_delay = 1
    else:
        volume_class_delay = 2

    # Weight classes
    weight = parcel.weight  # Get the weight of parcels and catergorize it
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

    ...

    Methods
    -------
    can_accept(self, parcel) -> bool:
        Determines if parcel can be accepted in an outfeed

    add_parcel(self, parcel) -> None:
        Adds parcel to certain outfeed

    update(self, time_step, system_time) -> None:
        Keeps track of all timings in the system.
    """
    def __init__(self, max_length: float) -> None:
        """
        Initializes the attributes for the parcel class. 
        :param max_lentgh: the maximal length of an outfeed.
        """
        self.max_length = max_length            # Maximum length of the outfeed
        self.current_parcels = []               # Stores the occupied length of the outfeed 
        self.current_length = 0.0               # Current length set to 0 at start
        self.time_until_next_discharge = 0.0    # Current waiting time, before parcel gets unloaded, set to 0 at start

    def can_accept(self, parcel) -> bool:
        """
        Determines if parcel can be accepted in a outfeed.
        :param parcel: a parcel
        :return: True if the parcel can be accepted, False otherwise.
        """
        return self.current_length + parcel.length <= self.max_length

    def add_parcel(self, parcel) -> None:
        """
        Adds parcel to certain outfeed.
        :param parcel: a parcel.
        """
        self.current_parcels.append((parcel, compute_outfeed_time(parcel)))
        self.current_length += parcel.length
        if len(self.current_parcels) == 1:
            # Timer for the current parcel
            self.time_until_next_discharge = self.current_parcels[0][1]

    def update(self, time_step) -> None:
        """
        Keeps track of all timings in the system. 
        :param time_step: time step
        """
        if self.current_parcels:
            self.time_until_next_discharge -= time_step
            if self.time_until_next_discharge <= 0:
                parcel, _ = self.current_parcels.pop(0)
                self.current_length -= parcel.length
                if self.current_parcels:
                    # Timer for the next parcel in line
                    self.time_until_next_discharge = self.current_parcels[0][1]


class PosiSorterSystem:
    """
    A class to represent the sorter system.

    ...

    Methods
    -------
    def simulate(self, parcels) -> None:
        Simulates the system.
    """
    def __init__(self, layout_df, num_outfeeds, sorting_algorithm, ml_model=None):
        self.belt_speed = layout_df.loc[
            layout_df['Layout property'] == 'Belt Speed', 'Value'
        ].values[0]
        self.dist_infeeds_to_scanner = layout_df.loc[
            layout_df['Layout property'] == 'Distance Arrival to Scanner', 'Value'
        ].values[0]
        self.dist_scanner_to_outfeeds = layout_df.loc[
            layout_df['Layout property'] == 'Distance Scanner to Outfeeds', 'Value'
        ].values[0]
        self.dist_between_outfeeds = layout_df.loc[
            layout_df['Layout property'] == 'Distance between Outfeeds', 'Value'
        ].values[0]

        # handle wrap-around distance
        possible_keys = [
            'Distance Outfeeds to Infeeds',
            'Distance Infeeds to Arrival',
            'Distance Outfeeds to Arrival'
        ]
        match = layout_df.loc[
            layout_df['Layout property'].isin(possible_keys), 'Value'
        ]
        self.dist_outfeeds_to_infeeds = match.values[0] if not match.empty else None

        self.num_outfeeds = num_outfeeds
        #self.outfeeds = [Outfeed() for _ in range(num_outfeeds)]
        # pull the per-outfeed lengths from your layout sheet
        outfeed_lengths = (
            layout_df[layout_df["Layout property"].str.startswith("Length outfeed")]
            .sort_values("Layout property")["Value"]
            .tolist()
        )
        # now build each Outfeed with its own max_length
        self.outfeeds = [
            Outfeed(max_length=outfeed_lengths[k])
            for k in range(self.num_outfeeds)
        ]
        self.sorting_algorithm = sorting_algorithm

        # statistics and state
        self.recirculated_count = 0
        self.outfeed_counts = [0] * num_outfeeds
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
        # how many arrivals until we run one local-search
        self.REBALANCE_INTERVAL = 1
        self.rebal_ctr = 0
        

        if ml_model is None:
            raise ValueError("You must pass ml_model to PosiSorterSystem")
        self.ml_model = ml_model
        self.first_pass_failures = set()
        self.training_data = []
        self._baseline_algo = None

    def _logging_wrapper(self, parcel):
        """
        Internal: run baseline_algo, log (features,label), then return label.
        """
        # 1) extract features against the CURRENT system state
        feat  = self.ml_model.extract_features(parcel, self)
        # 2) get the ground-truth label
        label = self._baseline_algo(parcel)
        # 3) store it
        self.training_data.append((feat, label))
        return label
    
    def collect_training_data(self, parcels: list, baseline_algo):
        """
        Run one simulation under `baseline_algo`, logging (feat,label) pairs in
        self.training_data.
        """
        # clear any old data
        del self.training_data[:]  
        # remember which algo to call inside the wrapper
        self._baseline_algo    = baseline_algo
        # swap in our logging wrapper
        self.sorting_algorithm = self._logging_wrapper
        # run the sim (fills training_data)
        self.simulate(parcels)
        # restore the original algorithm
        self.sorting_algorithm = baseline_algo

    def train_ml_model(self):
        """
        Build X,y from self.training_data and fit ml_model to imitate the baseline.
        """
        if not self.training_data:
            raise RuntimeError("No training data: call collect_training_data first")
        feats, labels = zip(*self.training_data)
        X = np.vstack(feats)
        y = np.array(labels, dtype=int)
        self.ml_model.clf.fit(X, y)
        self.ml_model.is_trained = True
        print(f"Trained ML model on {len(y)} examples")

    def handle(self, evt):
        pass

    def simulate(self, parcels) -> None:
        for p in parcels:
            p.recirculation_count = 0
            p.sorted_first_try   = False
        self.recirculated_count = 0
        self.outfeed_counts    = [0] * self.num_outfeeds
        self.non_sorted_parcels = 0
        self.loads             = {k: 0.0 for k in range(self.num_outfeeds)}
        self.loads_l           = {k: 0.0 for k in range(self.num_outfeeds)}
        self.service_times     = {}
        self.assignment        = {}
        self.assignment_l      = {}
        self.window.clear()
        self.rebal_ctr         = 0
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
                #print(f"[{wall_clock}] Parcel {parcel.id} removed from outfeed {k}")

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
        print(f"Sorting Alrgorithm: {self.sorting_algorithm.__name__ if callable(self.sorting_algorithm) else self.sorting_algorithm}")


def main():
    # 1. LOAD & CLEAN DATA
    xlsx_string = r"C:\Users\20234607\OneDrive - TU Eindhoven\Y2\Q4\CBL\Code\Useful code files\SampleDataChallenge.xlsx"
    xls = pd.ExcelFile(xlsx_string, engine="openpyxl")
    parcels_df = xls.parse('Parcels')
    layout_df  = xls.parse('Layout')

    parcels_df, drop_info        = clean_parcel_data(parcels_df)
    parcels, num_outfeeds        = load_parcels_from_clean_df(parcels_df)

    # 2. INITIALIZE & (OPTIONALLY) TRAIN / LOAD ML MODEL
    model_path = None
    ml = initialize_ml_model(model_path)

    # Build the system once — you can re-use it for each simulation if you clear state between runs
    system = PosiSorterSystem(
        layout_df,
        num_outfeeds,
        sorting_algorithm=None,  # we’ll set this per-run below
        ml_model=ml
    )

    # ────────────────────────────────────────────────────────────────────────────────
    # 3. CHOOSE WHICH ALGORITHM TO USE:
    #
    #    Simply comment or uncomment one of these lines, then call system.simulate()
    #    to run with that algorithm.
    # ────────────────────────────────────────────────────────────────────────────────

    # system.sorting_algorithm = lambda p: mlfs(p, system)
    #system.sorting_algorithm = fcfs
    # system.sorting_algorithm = genetic
    #system.sorting_algorithm = load_balance_time
    # system.sorting_algorithm = load_balance_length
    system.sorting_algorithm = load_balance_length_simple
    # system.sorting_algorithm = load_balance_time_simple

    # run the simulation
    system.simulate(parcels)

    # ────────────────────────────────────────────────────────────────────────────────
    # 4. WRITE OUT “Simulated Outfeed” COLUMN
    # ────────────────────────────────────────────────────────────────────────────────
    parcels_df['Simulated Outfeed'] = parcels_df['Parcel Number'].map(system.assignment)
    out_path = xlsx_string.replace('.xlsx', '_with_sim_outfeeds.xlsx')
    with pd.ExcelWriter(out_path, engine='openpyxl', mode='w') as writer:
        parcels_df.to_excel(writer, sheet_name='Parcels', index=False)
        layout_df.to_excel(writer, sheet_name='Layout',  index=False)

if __name__ == "__main__":
    main()
