# generate_readouts.py

import pandas as pd
from DES_GoodCode import (            # adjust this to your module name
    clean_parcel_data,
    load_parcels_from_clean_df,
    PosiSorterSystem,
    Event,
)

def main():
    EXCEL = "PosiSorterData1.xlsx"

    # 1) Read in raw sheets
    xls        = pd.ExcelFile(EXCEL)
    parcels_df = xls.parse("Parcels")
    layout_df  = xls.parse("Layout")

    # 2) Clean & build Parcel objects
    parcels_df, drop_info = clean_parcel_data(parcels_df)
    parcels               = load_parcels_from_clean_df(parcels_df)

    # 3) Instantiate & grab number of outfeeds
    system    = PosiSorterSystem(layout_df)
    n_out     = system.num_outfeeds

    # 4) Monkey-patch a handler to log every event
    event_log = []
    orig_handle = getattr(system, "handle", None)
    def handle_and_log(evt):
        event_log.append({
            "parcel_id":  evt.parcel.id,
            "event":      evt.type,
            "time":       evt.time,
            "outfeed_id": getattr(evt, "outfeed_id", None),
        })
        if orig_handle:
            orig_handle(evt)

    system.handle = handle_and_log

    # 5) Run the simulation (make sure simulate() calls self.handle(evt))
    system.simulate(parcels)

    # 6) Dump events.csv (with num_outfeeds column)
    if event_log:
        ev_df = pd.DataFrame(event_log)
        ev_df["num_outfeeds"] = n_out
        ev_df["event_name"]   = ev_df["event"].map({
            Event.ARRIVAL:       "Arrival",
            Event.ENTER_SCANNER: "EnterScanner",
            Event.ENTER_OUTFEED: "EnterOutfeed",
            Event.EXIT_OUTFEED:  "ExitOutfeed",
            Event.RECIRCULATE:   "Recirculate",
        })
        ev_df.to_csv("events.csv", index=False)
        print(f"Wrote events.csv ({len(ev_df)} rows)")

        # 7) Build parcel_summary.csv (also with num_outfeeds)
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
                "num_outfeeds":  n_out,
            })
        sum_df = pd.DataFrame(summary)
        sum_df.to_csv("parcel_summary.csv", index=False)
        print(f"Wrote parcel_summary.csv ({len(sum_df)} rows)")
    else:
        print("No events were logged. Ensure your simulate() calls `self.handle(evt)`.")

if __name__ == "__main__":
    main()
