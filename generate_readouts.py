# generate_readouts.py

import pandas as pd
from DES_simulation import PosiSorterSim, Event, Parcel  # Import your classes
from pathlib import Path

def load_parcels_and_layout(xlsx_path):
    """
    Reads Parcels and Layout sheets from the given Excel file,
    constructs Parcel objects with all required fields, and
    returns (parcels_list, layout_df).
    """
    xls = pd.ExcelFile(xlsx_path)
    parcels_df = xls.parse("Parcels")
    layout_df  = xls.parse("Layout")

    parcels = []
    # Normalize arrival times to real datetime (pd.Timestamp)
    for _, row in parcels_df.iterrows():
        parcel_id = int(row["Parcel Number"])
        arrival   = pd.to_datetime(row["Arrival Time"])
        length    = float(row["Length"])
        width     = float(row["Width"])
        height    = float(row["Height"])
        weight    = float(row["Weight"])
        # build feasible outfeed index list
        feasible  = [i for i, flag in enumerate(
                        (row["Outfeed 1"], row["Outfeed 2"], row["Outfeed 3"])
                     ) if flag]
        parcels.append(Parcel(parcel_id, arrival, length, width, height, weight, feasible))

    # sort by arrival time
    parcels.sort(key=lambda p: p.arrival)
    return parcels, layout_df

def main():
    EXCEL = "PosiSorterData1.xlsx"

    # 1) Load parcels & layout with the correct signature
    parcels, layout = load_parcels_and_layout(EXCEL)

    # 2) Instantiate the simulator and prepare event logging
    sim = PosiSorterSim(parcels, layout)
    event_log = []

    # keep reference to original handle
    orig_handle = sim.handle

    def handle_and_log(ev):
        # log event
        event_log.append({
            "parcel_id": ev.customer.parcel_id,
            "event":     ev.type,
            "time":      ev.time
        })
        # delegate to original logic
        return orig_handle(ev)

    # monkey-patch
    sim.handle = handle_and_log

    # 3) Run the simulation
    sim.run()

    # 4) Build and save events.csv
    ev_df = pd.DataFrame(event_log)
    ev_df["event_name"] = ev_df["event"].map({
        Event.ARRIVAL:      "Arrival",
        Event.ENTER_SCANNER:"EnterScanner",
        Event.ENTER_OUTFEED:"EnterOutfeed",
        Event.EXIT_OUTFEED: "ExitOutfeed"
    })
    ev_df.to_csv("events.csv", index=False)
    print(f"Wrote events.csv ({len(ev_df)} rows)")

    # 5) Build and save parcel_summary.csv
    summary = []
    for pid, grp in ev_df.groupby("parcel_id"):
        times = grp.set_index("event_name")["time"]
        summary.append({
            "parcel_id":     pid,
            "arrival_time":  times["Arrival"],
            "scanner_time":  times["EnterScanner"],
            "outfeed_enter": times.get("EnterOutfeed", pd.NaT),
            "outfeed_exit":  times.get("ExitOutfeed", pd.NaT),
            "recirculations": max(0, 
                grp[grp["event"] == Event.ENTER_SCANNER].shape[0] - 1
            )
        })
    sum_df = pd.DataFrame(summary)
    sum_df.to_csv("parcel_summary.csv", index=False)
    print(f"Wrote parcel_summary.csv ({len(sum_df)} rows)")

if __name__ == "__main__":
    main()
