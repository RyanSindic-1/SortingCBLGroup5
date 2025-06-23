import pandas as pd

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
    from Final_DES_Code import Parcel
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
        feasible_outfeeds = [i for i, col in enumerate(outfeed_columns) if row[col]] # Feasible outfeeds # Feasible outfeeds
        parcels.append(Parcel(parcel_id, arrival_time, length, width, height, weight, feasible_outfeeds))
    return sorted(parcels, key=lambda p: p.arrival_time), num_outfeeds