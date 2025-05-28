import heapq
import time
import pandas as pd
import random
from datetime import timedelta
import math

# -------------------------------------------------------------------------- #
# IMPORT CLEANED DATA FROM EXCEL FILE                                        #
# -------------------------------------------------------------------------- #

def drop_non_chronological_arrivals(df):                  # Remove rows with non-chronological arrival times
    valid_times = []
    last_time = pd.Timestamp.min                          # Initialize with the earliest possible timestamp

    for i, time in enumerate(df["Arrival Time"]):         # Loop through each arrival time in the DataFrame
        if time >= last_time:
            valid_times.append(True)                      # If not chronogica, append
            last_time = time
        else:
            valid_times.append(False)                     # If not chronogical, don't append

    return df[valid_times].reset_index(drop=True)

def remove_outliers_iqr(df, columns) -> pd.DataFrame:
    """
    Removes outliers from colums using IQR.
    :param columns: columns in data frame
    :return: data frame of excel file with outliers removed.
    """

    for col in columns:                                     
        Q1 = df[col].quantile(0.25)                         # First quartile
        Q3 = df[col].quantile(0.75)                         # Last quartile
        IQR = Q3 - Q1                                       # Interquertile range
        lower_bound = Q1 - 2 * IQR                          # Lower bound for outlier detection 
        upper_bound = Q3 + 2 * IQR                          # Upper bound for outlier detection 
        df = df[df[col].between(lower_bound, upper_bound)]  # Keep only rows that are in bound
    return df

def drop_rows_without_true_outfeed(df, prefix="Outfeed") -> pd.DataFrame:
    """
    Removes columns from data frame that have no possible outfeed.
    :param df: data frame from excel file
    :param prefix: prefix set to "Outfeed"
    :return: data frame of excel file with parcels without feasible outfeeds removed.
    """

    outfeed_cols = [col for col in df.columns if col.startswith(prefix)]  # Get all columns
    if not outfeed_cols:                                                  # Remove columns that have no outfeed
        return df        
    mask = df[outfeed_cols].any(axis=1)                                   # Store columns with outfeed                    
    return df[mask]

def clean_parcel_data(parcels_df) -> tuple[pd.DataFrame, dict]:
    """
    Cleans the data.
    :param parcels_df: data frame of excel file with parcels
    :return: clean data frame and information on how many rows were removed.
    """

    drop_info = {}                                           # Store removed rows
    drop_info['initial'] = initial_count = len(parcels_df)   # Initialize count
 
    parcels_df = parcels_df.dropna().reset_index(drop=True)  # Remove rows with NaNs
    after_na = len(parcels_df) 
    drop_info['na_dropped'] = initial_count - after_na       # Track amount of rows dropped due to NaNs

    before_outliers = len(parcels_df)                        # Remove rows with outliers
    parcels_df = remove_outliers_iqr(parcels_df, ["Length", "Width", "Height"])
    after_outliers = len(parcels_df)
    drop_info['outliers_dropped'] = before_outliers - after_outliers  # Track amount of rows dropped due to outlying values

    before_outfeeds = len(parcels_df)  # Remove rows with no feasible outfeeds
    parcels_df = drop_rows_without_true_outfeed(parcels_df)                      
    after_outfeeds = len(parcels_df) 
    drop_info['no_outfeeds_dropped'] = before_outfeeds - after_outfeeds     # Track amount of rows dropped due to no feasible outfeeds

    before_chrono = len(parcels_df)                                             # Remove rows with non-chronological arrival times
    parcels_df = drop_non_chronological_arrivals(parcels_df)
    after_chrono = len(parcels_df)
    drop_info['non_chrono_dropped'] = before_chrono - after_chrono              # Track amount of rows dropped due to no non-chronological arrival time
    
    drop_info['total_dropped'] = drop_info['na_dropped'] + drop_info['outliers_dropped'] + drop_info['no_outfeeds_dropped'] + drop_info['non_chrono_dropped']  # Total amount of rows dropped

    return parcels_df, drop_info

def load_parcels_from_clean_df(df) -> list:
    """
    Loads the data from the excel file
    :param df: data frame from excel file
    :return: list of parcel objects.
    """
    outfeed_columns = [col for col in df.columns if col.startswith("Outfeed ")]
    num_outfeeds = sum(col.split("Outfeed ")[1].isdigit() for col in outfeed_columns) # Count number of outfeeds
    
    parcels = []                                                # Store parcel information
    for _, row in df.iterrows():                                 
        parcel_id = int(row['Parcel Number'])                   # Parcel number
        arrival_time = pd.to_datetime(row['Arrival Time'])      # Arrival time package
        length = float(row['Length'])                           # Lenght of package
        width = float(row['Width'])                             # Width of package
        height = float(row['Height'])                           # Height of package
        weight = float(row['Weight'])                           # Weight of package
        feasible_outfeeds = [i for i, col in enumerate(outfeed_columns) if row[col]] # Feasible outfeeds
        parcels.append(Parcel(parcel_id, arrival_time, length, width, height, weight, feasible_outfeeds))
    return sorted(parcels, key=lambda p: p.arrival_time), num_outfeeds

# -------------------------------------------------------------------------- #
# EVENT CLASS                                                                #  
# -------------------------------------------------------------------------- #

class Event:
    """
    A class that represents events.
    """

    ARRIVAL = 0
    ENTER_SCANNER = 1
    ENTER_OUTFEED = 2
    EXIT_OUTFEED = 3 
    RECIRCULATE = 4

    def __init__(self, typ, time, parcel, outfeed_id = None) -> None:  # type is a reserved word
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

class FES :
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
        self.outfeed_attempts = []  # Afterwards, makes a copy of the feasible outfeeds of the parcel. Used for the algorithm functioning.
        self.recirculation_count = 0  # Used to keep track of the number of recirculations of each parcel so that we can cap it at 3

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
        volume_class_delay = random.uniform(0.0, 0.5)
    elif volume < 0.055:
        volume_class_delay = random.uniform(0.5, 1.5)
    else:
        volume_class_delay = random.uniform(1.5, 2.5)

    # Weight classes
    weight = parcel.weight  # Get the weight of parcels and catergorize it
    if weight < 1700:
        weight_class_delay = random.uniform(0.0, 0.5)
    elif weight < 2800:
        weight_class_delay = random.uniform(0.5, 1.5)
    else:
        weight_class_delay = random.uniform(1.5, 2.5)

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

    def __init__(self, max_length=3.0) -> None:
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
     A class to represent the outfeed.

    ...

    Methods
    -------
    def simulate(self, parcels) -> None:
        Simulates the system.
    """

    def __init__(self, layout_df, num_outfeeds) -> None:  # Not sure if i should add arrdist or something similar here.
        """
        Initializes the attributes for the PosiSorterSystem class.
        :param layout_df: layout sheet of excel file.
        """

        self.belt_speed                = layout_df.loc[layout_df['Layout property'] == 'Belt Speed',                 'Value'].values[0]
        self.dist_infeeds_to_scanner   = layout_df.loc[layout_df['Layout property'] == 'Distance Infeeds to Scanner','Value'].values[0]
        self.dist_scanner_to_outfeeds  = layout_df.loc[layout_df['Layout property'] == 'Distance Scanner to Outfeeds','Value'].values[0]
        self.dist_between_outfeeds     = layout_df.loc[layout_df['Layout property'] == 'Distance between Outfeeds',  'Value'].values[0]
        possible_keys = ["Distance Outfeeds to Infeeds", "Distance Infeeds to Arrival"]
        match = layout_df[layout_df['Layout property'].isin(possible_keys)]
        if not match.empty:
          self.dist_outfeeds_to_infeeds = match['Value'].values[0]
        else:
          self.dist_outfeeds_to_infeeds = None  
        self.num_outfeeds = num_outfeeds  # Given in Excel sheet. Can be automatically detected from the layout_df if needed.
        self.outfeeds = [Outfeed(max_length=3.0) for _ in range(self.num_outfeeds)]
        #  These are used to print statistics about the system:
        self.recirculated_count = 0
        self.outfeed_counts = [0] * self.num_outfeeds
        self.non_sorted_parcels = 0
    
    def simulate(self, parcels) -> None:
        """
        Simulates the system.
        :param parcels: parcels. 
        """

        fes =  FES()
        t = timedelta(0)
        t0 = parcels[0].arrival_time 
        #With this for loop, we initiate the simulation by adding all the arrival events of the excel file to the FES. It might not be the most efficient way, 
        # but I think it works since anyways, the events are sorted by time in the FES afterwards.
        #I NOW ASSUME THAT THE MINIMUM DISTANCE IN BETWEEN PARCELS HAS TO BE 0.3 M SO THAT THEN SCANNER WORKS CORRECTLY
        last_arrival_time = timedelta(0)
        #Basic calculation v*t = 0.3m
        safety_spacing = 0.3    
        int_arrival_safety_time = timedelta(seconds= safety_spacing / self.belt_speed)
        for p in parcels:
            proposed_arrival_time = p.arrival_time - t0

            # Ensure there's enough spacing
            if proposed_arrival_time < last_arrival_time + int_arrival_safety_time:
                proposed_arrival_time = last_arrival_time + int_arrival_safety_time

            arrival_event = Event(Event.ARRIVAL, proposed_arrival_time, p)
            fes.add(arrival_event)
            last_arrival_time = proposed_arrival_time
        while not fes.isEmpty(): 
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
                                last_outfeed = parcel.feasible_outfeeds[-1]  # Last outfeed in the original list
                                # No options left — recirculate
                                self.recirculated_count += 1
                                time_start_recirculation = timedelta(seconds=(self.dist_between_outfeeds * (self.num_outfeeds - last_outfeed)) / self.belt_speed)
                                fes.add(Event(Event.RECIRCULATE, t + time_start_recirculation, parcel))
                            continue  # Skip rest of ENTER_OUTFEED
                        
                        #So if the outfeed can accept the parcel, we add it to the outfeed and update the current length of the outfeed.
                        feed.add_parcel(parcel)
                        self.outfeed_counts[k] += 1

                        if len(feed.current_parcels) == 1:
                            # To add the additional time that the parcel takes since it enters the outfeed until it is at the bottom of it, I assume
                            # we are treating with GRAVITY ROLLER CONVEYORS, since if we had a belt conveyor, the outfeeds could not stop there and form a queue.
                            # This time is low, so it is only important when there are no more parcels in the outfeed.
                            # I simulate a simple inclined plane physics problem, where the outfeed is tilted 25 degrees, initial velocity is 0, and there is a bit of friction.
                            theta = 25  # degrees
                            theta_rad = math.radians(theta)  # convert to radians
                            g = 9.81 #m/s^2
                            mu = 0.03 # coefficient of friction
                            a = g * (math.sin(theta_rad) - mu * math.cos(theta_rad))  # acceleration down the slope
                            time_to_bottom = math.sqrt((2 * feed.max_length) / a)  # time to reach the bottom of the outfeed
                            discharge_time = timedelta(seconds=feed.current_parcels[0][1] + time_to_bottom)
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

                if parcel.recirculation_count < 3:  # If the parcel has been recirculated 3 times, it is discarded.
                    parcel.recirculation_count += 1 #Add a recirculation count to the parcel.
                    time_to_arrival = timedelta(seconds=(self.dist_outfeeds_to_infeeds) / self.belt_speed)
                    # candidate arrival moment on the belt
                    proposed_arrival_time = t + time_to_arrival

                    # enforce the same safety spacing used for ordinary arrivals
                    if proposed_arrival_time < last_arrival_time + int_arrival_safety_time:
                        proposed_arrival_time = last_arrival_time + int_arrival_safety_time

                    # schedule and remember
                    arrival = Event(Event.ARRIVAL, proposed_arrival_time, parcel)
                    fes.add(arrival)
                    last_arrival_time = proposed_arrival_time
                else:
                    print(f"[{wall_clock}] Parcel {parcel.id} discarded after 3 recirculations.")
                    self.non_sorted_parcels += 1
                #PREVIOUS LOGIC IN WHICH MERGE INTO ARRIVBAL POINT IS NOT TAKEN INTO ACCOUNT
                #time_to_scanner = timedelta(seconds = (self.dist_outfeeds_to_infeeds + self.dist_infeeds_to_scanner) / self.belt_speed)
                #rescan = Event(Event.ENTER_SCANNER, t + time_to_scanner, parcel)
                #fes.add(rescan)
        
        # Print statistics
        print("\n--- Simulation Summary ---")
        print(f"Total parcels processed: {len(parcels)}")
        print(f"Parcels recirculated: {self.recirculated_count}")
        for i, count in enumerate(self.outfeed_counts):
            print(f"Parcels sent to Outfeed {i}: {count}")
        print(f"Parcels not sorted (recirculated 3 times): {self.non_sorted_parcels}")
        print(f"Throughput (sorted): {sum(self.outfeed_counts)}")

# -------------------------------------------------------------------------- #
# RUN SIMUALTION                                                             #
# -------------------------------------------------------------------------- #

def main():
    xls = pd.ExcelFile("PosiSorterData1.xlsx")              # Load excel sheet with parcel data
    parcels_df = xls.parse('Parcels')                       # Gets Parcels sheet of Excel file
    layout_df = xls.parse('Layout')                         # Gets Layout sheet of Excel file

    parcels_df, drop_info = clean_parcel_data(parcels_df)   # Clear the data

    parcels, num_outfeeds = load_parcels_from_clean_df(parcels_df)  # Configurate the clean data into a list with parcel objects
    system = PosiSorterSystem(layout_df, num_outfeeds)
    system.simulate(parcels)  # Runs the simulation

if __name__ == "__main__":
    main()