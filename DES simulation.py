import heapq
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
import pandas as pd


class Event:
    ARRIVAL    = 0
    EXIT_OUTFEED  = 1    
    ENTER_SCANNER   = 2
    ENTER_OUTFEED   = 3

    def __init__(self, typ, time, cust=None):
        self.type = typ
        self.time = time
        self.customer = cust      # we pass Parcel objects here

    def __lt__(self, other):
        return self.time < other.time

    def __str__(self):
        # Kept simple: unknown types fall back to their numeric value
        names = {
            0: "Arrival", 1: "ExitOutfeed",
            2: "EnterScanner", 3: "EnterOutfeed"
        }
        label = names.get(self.type, f"Event{self.type}")
        return f"{label} of parcel {self.customer.parcel_id} at t={self.time:.2f}"


class FES:
    def __init__(self):
        self.events = []

    def add(self, event):
        heapq.heappush(self.events, event)

    def next(self):
        return heapq.heappop(self.events)

    def isEmpty(self):
        return len(self.events) == 0

    # unchanged helper (not used by the sorter, but left intact)
    def updateEventTimes(self, t, oldQL, newQL):
        events2 = []
        for e in self.events:
            if e.type == Event.EXIT_OUTFEED:
                e.time = t + (e.time - t) * newQL / oldQL
            heapq.heappush(events2, e)
        self.events = events2

    def __str__(self):
        s = ''
        for e in sorted(self.events):
            s += str(e) + '\n'
        return s


# --------------------------------------------------------------------------- #
#  DOMAIN OBJECTS                                                             #
# --------------------------------------------------------------------------- #
class Parcel:
    def __init__(self, parcel_id, arr_time, length, width, height, weight, feasible):
        self.parcel_id = parcel_id
        self.arrival  = arr_time
        self.length = length
        self.width = width
        self.height = height
        self.weight = weight
        self.feasible     = feasible[:]   # list of 0‑based outfeed indices
        self.remaining    = []            # will be (re)filled at scanner
        self.sorted       = False
        self.recirculated = False


class Outfeed:
    def __init__(self, capacity_m, discharge_time):
        self.capacity_m   = capacity_m     # available length on spur
        self.discharge_time  = discharge_time
        self.occupied_m   = 0.0
        self.queue: list[Parcel] = []

    def can_accept(self, parcel: Parcel) -> bool:
        return self.occupied_m + parcel.length <= self.capacity_m #still to be randomized between length and width

    def push(self, p: Parcel): #add parcel to outfeed
        self.queue.append(p)
        self.occupied_m += p.length

    def pop(self) -> Parcel: #remove parcel from outfeed
        p = self.queue.pop(0)
        self.occupied_m -= p.length
        return p


#Simulation class
class PosiSorterSim:
    def __init__(self, parcels: list[Parcel], layout: pd.DataFrame):
        ###
        self.belt_speed = layout.loc[layout['Layout property'] == 'Belt Speed', 'Value'].values[0]
        self.outfeeds = [Outfeed(max_length=3.0) for _ in range(self.num_outfeeds)]
        self.recirculation_belt = []
        self.processed_parcels = []
        ###
        self.parcels = parcels
        self.fes = FES()
        self.clock = parcels[0].arrival

        # layout data --------------------------------------------------------
        self.v = float(layout.loc[layout["Layout property"] == "Belt Speed",
                                  "Value"].iat[0])          # m/s
        self.dE_S = float(layout.loc[layout["Layout property"] == "E→S",
                                     "Value"].iat[0])        # m
        self.dS_O = [float(layout.loc[
                      layout["Layout property"] == f"S→O{i}", "Value"].iat[0])
                     for i in (1, 2, 3)]                     # m
        self.loop = float(layout.loc[layout["Layout property"] ==
                                     "Loop length", "Value"].iat[0])

        self.outfeeds = [Outfeed(capacity_m=3.0, discharge_time=2.0)
                         for _ in range(3)]

        self.stats = {"sorted": 0, "recirc": 0}

    # -------------- helpers -------------------------------------------------
    def schedule(self, delay_s, typ, parcel):
        self.fes.add(Event(typ, self.clock + delay_s, parcel))

    def dist_time(self, dist_m) -> float:
        return dist_m / self.v

    def handle(self, ev: Event):
        self.clock = ev.time
        p = ev.customer
        et = ev.type

        if et == Event.ARRIVAL:
            # Induction → travel to scanner
            self.schedule(self.dist_time(self.dE_S),
                          Event.ENTER_SCANNER, p)

        elif et == Event.ENTER_SCANNER:
            # reset the list of to‑visit outfeeds
            p.remaining = [idx for idx in (0, 1, 2) if idx in p.feasible]
            self.try_next_outfeed(p, from_scanner=True)

        elif et == Event.ENTER_OUTFEED:
            idx = p.remaining[0]
            of  = self.outfeeds[idx]
            if of.can_accept(p):
                # accept
                of.push(p)
                p.sorted = True
                self.stats["sorted"] += 1
                self.schedule(of.discharge_time, Event.EXIT_OUTFEED, p)
            else:
                # full → try next feasible outfeed this lap
                p.remaining.pop(0)
                self.try_next_outfeed(p)

        elif et == Event.EXIT_OUTFEED:           # parcel physically leaves outfeed
            idx = p.feasible[0] if p.feasible else -1
            self.outfeeds[idx].pop()          # update length

    # choose next outfeed or recirculate -------------------------------------
    def try_next_outfeed(self, p: Parcel, *, from_scanner=False):
        if p.remaining:
            idx = p.remaining[0]
            # distance to that outfeed nose:
            base = self.dS_O[idx]
            if not from_scanner:                      # already on loop
                prev_idx = (idx+3-1) % 3              # previous in order
                base = (self.loop + self.dS_O[idx] -
                        self.dS_O[prev_idx]) % self.loop
            self.schedule(self.dist_time(base),
                          Event.ENTER_OUTFEED, p)
        else:
            # none left → recirculate
            p.recirculated = True
            self.stats["recirc"] += 1
            self.schedule(self.dist_time(self.loop),
                          Event.ENTER_SCANNER, p)

    def run(self):
        for p in self.parcels:
            self.fes.add(Event(Event.ARRIVAL, p.arrival, p))

        while not self.fes.isEmpty():
            self.handle(self.fes.next())

        # report
        n = len(self.parcels)
        print("\n--- RESULTS --------------------------------------")
        print(f"Parcels processed   : {n}")
        print(f"Sorted first attempt: {self.stats['sorted']}")
        print(f"Recirculated        : {self.stats['recirc']}")
        print("---------------------------------------------------")



def load_from_excel(xlsx: str):
    xls  = pd.ExcelFile(xlsx)
    pdf  = xls.parse("Parcels")
    ldf  = xls.parse("Layout")

    parcels = []
    for _, r in pdf.iterrows():
        feas = [i for i, ok in enumerate(
                    [r["Outfeed 1"], r["Outfeed 2"], r["Outfeed 3"]]) if ok]
        parcels.append(
            Parcel(int(r["Parcel Number"]),
                   pd.to_datetime(r["Arrival Time"]),
                   float(r["Length"]),
                   feas)
        )
    parcels.sort(key=lambda p: p.arrival)
    return parcels, ldf


if __name__ == "__main__":
    plist, layout = load_from_excel("PosiSorterData1.xlsx")
    PosiSorterSim(plist, layout).run()
