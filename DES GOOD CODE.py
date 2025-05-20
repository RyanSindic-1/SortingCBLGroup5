import heapq
import pandas as pd
import random
from datetime import timedelta
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
    RECIRCULATE = 4
    def __init__(self, typ, time, parcel, outfeed_id = None):  # type is a reserved word
        """
        Parameters:
        param1 (typ): What type of event it is, e.g, Arrival, scanner...
        param2 (time): at what time the event occurs.
        param3 (parcel): all the information of the parcel that is being processed.
        param4 (outfeed_id): the outfeed to which the parcel goes. It is an optional parameter since it is only needed for events 2 and 3.
        """
        self.type = typ
        self.time = time
        self.parcel = parcel
        self.outfeed_id = outfeed_id  # Only used for ENTER_OUTFEED and EXIT_OUTFEED events
        
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
        self.outfeed_attempts = [] #Afterwards, makes a copy of the feasible outfeeds of the parcel. Used for the algorithm functioning.

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

#Probably not to be used in this code, I leave it here now just in case, but the outfeed functioning goes in the simualation class.
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
        self.num_outfeeds = 3  # Given in Excel sheet. Can be automatically detected from the layout_df if needed.
        self.outfeeds = [Outfeed(max_length=3.0) for _ in range(self.num_outfeeds)]
        #These are used to print statistics about the system:
        self.recirculated_count = 0
        self.outfeed_counts = [0] * self.num_outfeeds
        self.total_processed = 0
    
    def simulate(self, parcels):
        fes =  FES()
        t = 0
        t0 = parcels[0].arrival_time #We need to convert the arrival time to seconds, since the rest of the times are in seconds.
        #With this for loop, we initiate the simulation by adding all the arrival events of the excel file to the FES. It might not be the most efficient way, 
        # but I think it works since anyways, the events are sorted by time in the FES afterwards.
        for p in parcels:
            arrival_event =  Event(Event.ARRIVAL, p.arrival_time - t0, p) 
            fes.add(arrival_event) #scheduele the event 
        while not fes.isEmpty(): #T is still to be determined
            tOld = t #not being used right now, ususally used to store waiting times.
            evt = fes.next() #retrieve the next event in the FES and removes it
            t = evt.time
            #Handle ARRIVAL events
            if evt.type == Event.ARRIVAL:
                parcel = evt.parcel  
                time_to_scanner =  timedelta(seconds = self.dist_infeeds_to_scanner / self.belt_speed)   #time from infeed to scanner calculated with conveyor speed
                scan = Event(Event.ENTER_SCANNER, t + time_to_scanner, parcel)
                fes.add(scan) #scheduele the event

            elif evt.type == Event.ENTER_SCANNER:
                #In order to implement the algorithm, we should probably make a new list that contains the parcels that 
                #have been scanned, so that it only consider those parcels instead of all of them.
                #something like scanned_parcels = []
                #I do not know how to implement the distance to different outfeeds yet. I put it with a k, the number of the outfeed.
                parcel = evt.parcel
                parcel.outfeed_attempts = list(parcel.feasible_outfeeds)  # Copy of the list
                
                parcel = evt.parcel
                parcel.outfeed_attempts = list(parcel.feasible_outfeeds)
                first_choice = parcel.outfeed_attempts.pop(0)

                time_to_outfeed = timedelta(seconds=(self.dist_scanner_to_outfeeds + first_choice * self.dist_between_outfeeds) / self.belt_speed)
                fes.add(Event(Event.ENTER_OUTFEED, t + time_to_outfeed, parcel, outfeed_id=first_choice))


            elif evt.type == Event.ENTER_OUTFEED:
                    #number of the outfeed should be included in the parcels data 
                    #here we replicate a G/G/1 queue in each outfeed
                    
                        k      = evt.outfeed_id
                        feed   = self.outfeeds[k]
                        parcel = evt.parcel
                        wall_clock = (t0 + evt.time).time()
                        
                        # if this parcel is starting a (possibly empty) queue,
                        # schedule its exit when its discharge time elapses
                        if not feed.can_accept(parcel):
                            # Try next available outfeed, if any
                            if parcel.outfeed_attempts:
                                next_k = parcel.outfeed_attempts.pop(0)
                                time_to_next = timedelta(seconds=self.dist_between_outfeeds / self.belt_speed)
                                fes.add(Event(Event.ENTER_OUTFEED, t + time_to_next, parcel, outfeed_id=next_k))
                            else:
                                # No options left — recirculate
                                self.recirculated_count += 1
                                time_start_recirculation = timedelta(seconds=(self.dist_between_outfeeds * self.num_outfeeds + self.dist_outfeeds_to_infeeds + self.dist_infeeds_to_scanner) / self.belt_speed)
                                fes.add(Event(Event.RECIRCULATE, t + time_start_recirculation, parcel))
                            continue  # Skip rest of ENTER_OUTFEED
                        
                        #So if the outfeed can accept the parcel, we add it to the outfeed and update the current length of the outfeed.
                        feed.add_parcel(parcel)
                        self.outfeed_counts[k] += 1

                        if len(feed.current_parcels) == 1:
                            discharge_time = timedelta(seconds=feed.current_parcels[0][1])
                            fes.add(Event(Event.EXIT_OUTFEED, t + discharge_time, parcel, outfeed_id=k))

            elif evt.type == Event.EXIT_OUTFEED:
                k    = evt.outfeed_id
                feed = self.outfeeds[k]
                parcel = evt.parcel

                wall_clock = (t0 + evt.time).time()
                print(f"[{wall_clock}] Parcel {parcel.id} removed from outfeed {k}")
                # remove the parcel that just left; add the next discharge event if any
                feed.update(feed.time_until_next_discharge)         # knocks the head parcel out
                if feed.current_parcels:                            # there is another one waiting
                    discharge_time = timedelta(seconds = feed.current_parcels[0][1])
                    next_parcel = feed.current_parcels[0][0]
                    fes.add(Event(Event.EXIT_OUTFEED,
                                t + discharge_time,
                                next_parcel,
                                outfeed_id=k))
            elif evt.type == Event.RECIRCULATE:
                parcel = evt.parcel
                time_to_scanner = timedelta(seconds = (self.dist_outfeeds_to_infeeds + self.dist_infeeds_to_scanner) / self.belt_speed)
                rescan = Event(Event.ENTER_SCANNER, t + time_to_scanner, parcel)
                fes.add(rescan)
        
        # Print statistics
        print("\n--- Simulation Summary ---")
        print(f"Total parcels processed: {len(parcels)}")
        print(f"Parcels recirculated: {self.recirculated_count}")
        for i, count in enumerate(self.outfeed_counts):
            print(f"Parcels sent to Outfeed {i}: {count}")
        print(f"Throughput (sorted + recirculated): {sum(self.outfeed_counts) + self.recirculated_count}")




# -------------------------------------------------------------------------- #
# RUN SIMUALTION                                                             #
# -------------------------------------------------------------------------- #
def main():
    xls = pd.ExcelFile("C:/Users/20231616/Documents/CBL Q4/PosiSorterData1.xlsx")
    parcels_df = xls.parse('Parcels')
    layout_df = xls.parse('Layout')

    parcels_df, drop_info = clean_parcel_data(parcels_df)

    parcels = load_parcels_from_clean_df(parcels_df)
    system = PosiSorterSystem(layout_df)
    system.simulate(parcels)  # Simulate for 1 hour/s (in seconds)
    # system.run_simulation(parcels)

if __name__ == "__main__":
    main()
