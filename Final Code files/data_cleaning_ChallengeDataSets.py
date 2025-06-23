import pandas as pd
import re

def drop_non_chronological_arrivals(df):
    """
    Remove rows with non-chronological arrival times.
    """
    valid_times = []
    last_time = pd.Timestamp.min

    for time in df["Arrival Time"]:
        if time >= last_time:
            valid_times.append(True)
            last_time = time
        else:
            valid_times.append(False)

    return df[valid_times].reset_index(drop=True)


def remove_outliers_iqr(df, columns) -> pd.DataFrame:
    """
    Removes outliers from specified columns using the IQR method.
    """
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + 2 * IQR
        df = df[df[col].between(lower_bound, upper_bound)]
    return df.reset_index(drop=True)


def drop_rows_without_true_outfeed(df, prefix="Outfeed") -> pd.DataFrame:
    """
    Removes rows that have no feasible outfeed flags (e.g., Outfeed 1, Outfeed 2, ...).
    """
    outfeed_cols = [col for col in df.columns if re.match(rf"{re.escape(prefix)} \d+$", col)]
    if not outfeed_cols:
        return df.reset_index(drop=True)
    mask = df[outfeed_cols].any(axis=1)
    return df[mask].reset_index(drop=True)


def clean_parcel_data(parcels_df) -> tuple[pd.DataFrame, dict]:
    """
    Applies a series of cleaning steps to the parcels DataFrame.
    Returns the cleaned DataFrame and a dict of drop statistics.
    """
    drop_info = {}
    drop_info['initial'] = initial_count = len(parcels_df)

    # Identify which columns are truly required to be non-missing:
    outfeed_flag_cols = [col for col in parcels_df.columns if re.match(r"Outfeed \d+$", col)]
    required = [
        'Parcel Number', 'Arrival Time',
        'Length', 'Width', 'Height', 'Weight',
        'Service Time Outfeed'
    ] + outfeed_flag_cols

    # 1) drop rows missing any of the required fields
    parcels_df = parcels_df.dropna(subset=required).reset_index(drop=True)
    drop_info['na_dropped'] = initial_count - len(parcels_df)

    # 2) remove size outliers
    before_outliers = len(parcels_df)
    parcels_df = remove_outliers_iqr(parcels_df, ["Length", "Width", "Height"])
    drop_info['outliers_dropped'] = before_outliers - len(parcels_df)

    # 3) remove rows without any true outfeed flags
    before_outfeeds = len(parcels_df)
    parcels_df = drop_rows_without_true_outfeed(parcels_df)
    drop_info['no_outfeeds_dropped'] = before_outfeeds - len(parcels_df)

    # 4) remove non-chronological arrivals
    before_chrono = len(parcels_df)
    parcels_df = drop_non_chronological_arrivals(parcels_df)
    drop_info['non_chrono_dropped'] = before_chrono - len(parcels_df)

    drop_info['total_dropped'] = (
        drop_info['na_dropped'] +
        drop_info['outliers_dropped'] +
        drop_info['no_outfeeds_dropped'] +
        drop_info['non_chrono_dropped']
    )
    return parcels_df, drop_info


def load_parcels_from_clean_df(df) -> tuple[list, int]:
    """
    Converts the cleaned DataFrame into a list of Parcel objects.
    Infers feasible outfeeds from Outfeed <number> boolean columns.
    Returns a sorted list of parcels and the number of outfeeds.
    """
    from Final_DES_Code_ChallengeDataSets import Parcel

    # detect and sort the Outfeed flag columns
    outfeed_cols = [col for col in df.columns if re.match(r"Outfeed \d+$", col)]
    outfeed_cols_sorted = sorted(outfeed_cols, key=lambda x: int(x.split()[1]))
    num_outfeeds = len(outfeed_cols_sorted)

    parcels = []
    for _, row in df.iterrows():
        feasible = [i for i, col in enumerate(outfeed_cols_sorted) if bool(row[col])]
        p = Parcel(
            parcel_id        = int(row['Parcel Number']),
            arrival_time     = pd.to_datetime(row['Arrival Time']),
            length           = float(row['Length']),
            width            = float(row['Width']),
            height           = float(row['Height']),
            weight           = float(row['Weight']),
            feasible_outfeeds= feasible
        )
        parcels.append(p)

    # sort by arrival time
    parcels_sorted = sorted(parcels, key=lambda p: p.arrival_time)
    return parcels_sorted, num_outfeeds