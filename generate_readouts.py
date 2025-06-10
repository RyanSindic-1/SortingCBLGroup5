# generate_readouts.py

import pandas as pd
from DES_GoodCode_implemented import (            # adjust to your module name
    clean_parcel_data,
    load_parcels_from_clean_df,
    PosiSorterSystem,
    Event,
)
from sorting_algorithms_implemented import fcfs # pick your algorithm

def main():
    EXCEL = "PosiSorterData2.xlsx"

    # 1) Read in raw sheets
    xls        = pd.ExcelFile(EXCEL)
    parcels_df = xls.parse("Parcels")
    layout_df  = xls.parse("Layout")

    # 2) Clean & build Parcel objects
    parcels_df, drop_info       = clean_parcel_data(parcels_df)
    parcels, num_outfeeds       = load_parcels_from_clean_df(parcels_df)

    # 3) Instantiate system
    system = PosiSorterSystem(
        layout_df,
        num_outfeeds,
        sorting_algorithm=fcfs,
    )

    # 4) Monkey‐patch a handler to log every event
    event_log = []
    # Capture the original bound method once:
    orig_handle = system.handle

    def handle_and_log(evt):
        # 4a) Log it
        event_log.append({
            "parcel_id":  evt.parcel.id,
            "event":      evt.type,
            "time":       evt.time,
            "outfeed_id": getattr(evt, "outfeed_id", None),
        })
        # 4b) Call the original (no-op) handler so we don't lose any logic
        orig_handle(evt)

    # Override the instance method
    system.handle = handle_and_log

    # 5) Run the simulation
    system.simulate(parcels)

    # 6) Dump CSVs as before...
    if event_log:
        ev_df = pd.DataFrame(event_log)
        ev_df["num_outfeeds"] = num_outfeeds
        ev_df["event_name"]   = ev_df["event"].map({
            Event.ARRIVAL:       "Arrival",
            Event.ENTER_SCANNER: "EnterScanner",
            Event.ENTER_OUTFEED: "EnterOutfeed",
            Event.EXIT_OUTFEED:  "ExitOutfeed",
            Event.RECIRCULATE:   "Recirculate",
        })
        ev_df.to_csv("events.csv", index=False)
        print(f"Wrote events.csv ({len(ev_df)} rows)")

        # Parcel‐level summary...
        summary = []
        for pid, grp in ev_df.groupby("parcel_id"):
            times = grp.set_index("event_name")["time"]
            summary.append({
                "parcel_id":     pid,
                "arrival_time":  times.get("Arrival",      pd.NaT),
                "scanner_time":  times.get("EnterScanner", pd.NaT),
                "outfeed_enter": times.get("EnterOutfeed", pd.NaT),
                "outfeed_exit":  times.get("ExitOutfeed",  pd.NaT),
                "recirculations": max(0,
                    grp[grp["event"] == Event.ENTER_SCANNER].shape[0] - 1
                ),
                "num_outfeeds":  num_outfeeds,
            })
        sum_df = pd.DataFrame(summary)
        sum_df.to_csv("parcel_summary.csv", index=False)
        print(f"Wrote parcel_summary.csv ({len(sum_df)} rows)")
    else:
        print("No events were logged. Ensure your simulate() calls `self.handle(evt)`.")

if __name__ == "__main__":
    main()
