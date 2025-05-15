import heapq
import pandas as pd
import random
# -------------------------------------------------------------------------- #
# IMPORT CLEANED DATA FROM EXCEL FILE                                        #
# -------------------------------------------------------------------------- #
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + 2 * IQR
        df = df[df[col].between(lower_bound, upper_bound)]
    return df

def drop_rows_without_true_outfeed(df, prefix="Outfeed"):
    outfeed_cols = [col for col in df.columns if col.startswith(prefix)]
    if not outfeed_cols:
        return df
    mask = df[outfeed_cols].any(axis=1)
    return df[mask]

def clean_parcel_data(parcels_df):
    drop_info = {}
    drop_info['initial'] = initial_count = len(parcels_df)

    parcels_df = parcels_df.dropna().reset_index(drop=True)
    after_na = len(parcels_df)
    drop_info['na_dropped'] = initial_count - after_na

    before_outliers = len(parcels_df)
    parcels_df = remove_outliers_iqr(parcels_df, ["Length", "Width", "Height"])
    after_outliers = len(parcels_df)
    drop_info['outliers_dropped'] = before_outliers - after_outliers

    before_outfeeds = len(parcels_df)
    parcels_df = drop_rows_without_true_outfeed(parcels_df)
    after_outfeeds = len(parcels_df)
    drop_info['no_outfeeds_dropped'] = before_outfeeds - after_outfeeds

    drop_info['total_dropped'] = drop_info['na_dropped'] + drop_info['outliers_dropped'] + drop_info['no_outfeeds_dropped']

    return parcels_df, drop_info

def load_parcels_from_clean_df(df):
    parcels = []
    for _, row in df.iterrows():
        parcel_id = int(row['Parcel Number'])
        arrival_time = pd.to_datetime(row['Arrival Time'])
        length = float(row['Length'])
        width = float(row['Width'])
        height = float(row['Height'])
        weight = float(row['Weight'])
        feasible_outfeeds = [i for i, flag in enumerate([row['Outfeed 1'], row['Outfeed 2'], row['Outfeed 3']]) if flag]
        parcels.append(Parcel(parcel_id, arrival_time, length, width, height, weight, feasible_outfeeds))
    return sorted(parcels, key=lambda p: p.arrival_time)

# -------------------------------------------------------------------------- #
# EVENT CLASS                                                                #  
# -------------------------------------------------------------------------- #
class Event:

    ARRIVAL = 0
    ENTER_SCANNER = 1
    ENTER_OUTFEED = 2
    EXIT_OUTFEED = 3 
    # We should maybe add later on:  RECIRCULATE = 4 
    def __init__(self, typ, time, payload = None):  # type is a reserved word
        self.type = typ
        self.time = time
        self.payload = payload
        
    def __lt__(self, other):
        return self.time < other.time

class FES :
    """
    Future Event Set (FES) for discrete event simulation.
    This class uses a priority queue to manage events in the simulation.
    Events are sorted by their time attribute, which is a float.
    """
    def __init__(self):
        self.events = []
        
    def add(self, event):
        heapq.heappush(self.events, event)
        
    def next(self):
        return heapq.heappop(self.events)
    
    def isEmpty(self):
        return len(self.events) == 0

class Parcel: #This replicates the Customer class, in which info about the customers (parcels) is stored.

    def __init__(self, parcel_id, arrival_time, length, width, height, weight, feasible_outfeeds):
        self.id = parcel_id
        self.arrival_time = arrival_time
        self.length = length
        self.width = width
        self.height = height
        self.weight = weight
        self.feasible_outfeeds = feasible_outfeeds
        self.sorted = False
        self.recirculated = False

    def get_volume(self):
        return self.length * self.width * self.height

def compute_outfeed_time(parcel):
    base_time = 4.5

    # Volume classes
    volume = parcel.get_volume()
    if volume < 0.035:
        volume_class_delay = random.uniform(0.0, 0.5)
    elif volume < 0.055:
        volume_class_delay = random.uniform(0.5, 1.5)
    else:
        volume_class_delay = random.uniform(1.5, 2.5)

    # Weight classes
    weight = parcel.weight
    if weight < 1700:
        weight_class_delay = random.uniform(0.0, 0.5)
    elif weight < 2800:
        weight_class_delay = random.uniform(0.5, 1.5)
    else:
        weight_class_delay = random.uniform(1.5, 2.5)

    return base_time + volume_class_delay + weight_class_delay

class Outfeed:
    def __init__(self, max_length=3.0):
        self.max_length = max_length
        self.current_parcels = []
        self.current_length = 0.0
        self.time_until_next_discharge = 0.0 

    def can_accept(self, parcel):
        return self.current_length + parcel.length <= self.max_length

    def add_parcel(self, parcel):
        self.current_parcels.append((parcel, compute_outfeed_time(parcel)))
        self.current_length += parcel.length
        if len(self.current_parcels) == 1:
            #Timer for the current parcel 
            self.time_until_next_discharge = self.current_parcels[0][1] 


    def update(self, time_step):
        if self.current_parcels:
            self.time_until_next_discharge -= time_step
            if self.time_until_next_discharge <= 0:
                parcel, _ = self.current_parcels.pop(0)
                self.current_length -= parcel.length
                print(f"Parcel {parcel.id} removed from outfeed")
                if self.current_parcels:
                    #Timer for the next parcel in line
                    self.time_until_next_discharge = self.current_parcels[0][1]


class PosiSorterSystem:

    def __init__(self, layout_df): #Not sure if i should add arrdist or something similar here.
        self.belt_speed                = layout_df.loc[layout_df['Layout property'] == 'Belt Speed',                 'Value'].values[0]
        self.dist_infeeds_to_scanner   = layout_df.loc[layout_df['Layout property'] == 'Distance Infeeds to Scanner','Value'].values[0]
        self.dist_scanner_to_outfeeds  = layout_df.loc[layout_df['Layout property'] == 'Distance Scanner to Outfeeds','Value'].values[0]
        self.dist_between_outfeeds     = layout_df.loc[layout_df['Layout property'] == 'Distance between Outfeeds',  'Value'].values[0]
        self.dist_outfeeds_to_infeeds  = layout_df.loc[layout_df['Layout property'] == 'Distance Outfeeds to Infeeds','Value'].values[0]
        self.num_outfeeds = 3  # Given in Excel sheet
        self.outfeeds = [Outfeed(max_length=3.0) for _ in range(self.num_outfeeds)]
    
    def simulate(self, T):
        fes =  FES()

        a = parcels[0].arrival_time #first arrival time
        firstEvent = Event(Event.ARRIVAL, a)
        firstParcel = Parcel(parcel_id = int(row['Parcel Number']) , arrival_time=a, length=0, width=0, height=0, weight=0, feasible_outfeeds=[])
        fes.add(firstEvent)

        while t < T:
            tOld = t
            evt = fes.next() #retrieve the next event in the FES and removes it
            t = evt.time
            #Handle ARRIVAL events
            if evt.type == Event.ARRIVAL:
                b =  self.dist_infeeds_to_scanner / self.belt_speed   #time from infeed to scanner calculated with conveyor speed
                scan = Event(Event.ENTER_SCANNER, t + b)
                fes.add(scan) #scheduele the event
                arr = Event(Event.ARRIVAL, parcels[i].arrival_time) #scheduele next arrival
            
            #handle ENTER_SCANNER events
            elif evt.type == Event.ENTER_SCANNER:
                #information about the parcel is know from the sorting system
                parcel = evt.payload
                chosen_idx = None
                for idx in parcel.feasible_outfeeds:
                    if self.outfeeds[idx].can_accept(parcel):
                        chosen_idx = idx
                        break
                if chosen_idx is not None:
                    # travel time from scanner to that outfeed’s diverter
                    dist = self.dist_scanner_to_outfeeds + chosen_idx * self.dist_between_outfeeds
                    travel_time = dist / self.belt_speed
                    at_outfeed  = self.clock + pd.Timedelta(seconds=travel_time)
                    self.fes.add(Event(Event.ENTER_OUTFEED, at_outfeed, (parcel, chosen_idx)))
            #handle ENTER_OUTFEED events
            elif evt.type == Event.ENTER_OUTFEED:
            #handle EXIT_OUTFEED events
            elif evt.type == Event.EXIT_OUTFEED:





# -------------------------------------------------------------------------- #
# SIMULATION CLASS                                                           #
# -------------------------------------------------------------------------- #









# -------------------------------------------------------------------------- #
# RUN SIMUALTION                                                             #
# -------------------------------------------------------------------------- #
def main():
    xls = pd.ExcelFile("PosiSorterData1.xlsx")
    parcels_df = xls.parse('Parcels')
    layout_df = xls.parse('Layout')

    parcels_df, drop_info = clean_parcel_data(parcels_df)

    parcels = load_parcels_from_clean_df(parcels_df)
    system = PosiSorterSystem(layout_df)
    system.drop_info = drop_info
    # system.run_simulation(parcels)

if __name__ == "__main__":
    main()
