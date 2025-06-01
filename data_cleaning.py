import pandas as pd

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
    from DES_GoodCode import Parcel 
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


