# simpy_simulation.py

import simpy
import pandas as pd
import random

# Parameters
MAX_OUTFEED_LENGTH = 3.0    # meters
BASE_SERVICE_TIME  = 4.5    # seconds
DT_RECORD          = 0.1    # recording interval in seconds

def compute_service_time(parcel):
    """Compute processing time at an outfeed for a parcel."""
    volume = parcel['length'] * parcel['width'] * parcel['height']
    # Volume‐class delay
    if volume < 0.035:
        vol_delay = random.uniform(0.0, 0.5)
    elif volume < 0.055:
        vol_delay = random.uniform(0.5, 1.5)
    else:
        vol_delay = random.uniform(1.5, 2.5)
    # Weight‐class delay
    w = parcel['weight']
    if w < 1700:
        wt_delay = random.uniform(0.0, 0.5)
    elif w < 2800:
        wt_delay = random.uniform(0.5, 1.5)
    else:
        wt_delay = random.uniform(1.5, 2.5)
    return BASE_SERVICE_TIME + vol_delay + wt_delay

def load_layout(layout_df):
    """
    Pull belt speed, decision‐point distances & total loop length
    from the Layout sheet. Matches your actual property names.
    """
    # Belt speed
    belt_speed = float(
        layout_df.loc[
            layout_df['Layout property'] == 'Belt Speed',
            'Value'
        ].iloc[0]
    )

    # Distance Infeeds → Scanner
    d_scan = float(
        layout_df.loc[
            layout_df['Layout property'] == 'Distance Infeeds to Scanner',
            'Value'
        ].iloc[0]
    )

    # Distance Scanner → first outfeed
    d_scanner_to_out = float(
        layout_df.loc[
            layout_df['Layout property'] == 'Distance Scanner to Outfeeds',
            'Value'
        ].iloc[0]
    )

    # Distance between consecutive outfeeds
    d_between = float(
        layout_df.loc[
            layout_df['Layout property'] == 'Distance between Outfeeds',
            'Value'
        ].iloc[0]
    )

    # Distance from last outfeed back to infeed
    d_back = float(
        layout_df.loc[
            layout_df['Layout property'] == 'Distance Outfeeds to Infeeds',
            'Value'
        ].iloc[0]
    )

    # Build cumulative decision‐point distances from the start of the loop:
    #   point[0] = at scanner
    #   point[1] = at outfeed1
    #   point[2] = at outfeed2
    #   point[3] = at outfeed3
    points = []
    run = d_scan
    points.append(run)                # scanner
    run += d_scanner_to_out
    points.append(run)                # outfeed1
    run += d_between
    points.append(run)                # outfeed2
    run += d_between
    points.append(run)                # outfeed3

    loop_length = run + d_back

    return {
        'belt_speed':  belt_speed,
        'points':      points,
        'loop_length': loop_length
    }

def parcel_process(env, parcel, layout, outfeed_states, results):
    """A single parcel travelling until it finds a free outfeed."""
    belt_speed = layout['belt_speed']
    points     = layout['points']
    loop_len   = layout['loop_length']

    while True:
        served = False
        for idx in parcel['feasible_outfeeds']:
            # decision_dist is the distance at that outfeed decision point
            decision_dist = points[idx + 1]
            travel_t = decision_dist / belt_speed
            yield env.timeout(travel_t)

            state = outfeed_states[idx]
            if state['length'] + parcel['length'] <= MAX_OUTFEED_LENGTH:
                # occupy outfeed, service, then leave
                state['length'] += parcel['length']
                svc_t = compute_service_time(parcel)
                yield env.timeout(svc_t)
                state['length'] -= parcel['length']
                results['sorted'].append(parcel['id'])
                served = True
                break

        if served:
            return

        # no outfeed fit → recirculate full loop
        yield env.timeout(loop_len / belt_speed)
        results['recirculated'].append(parcel['id'])

def recorder_process(env, parcels, layout, results, recorder):
    """
    Every DT_RECORD seconds, record each unsorted parcel's
    position (distance along loop).
    """
    loop_len   = layout['loop_length']
    belt_speed = layout['belt_speed']

    while True:
        t = env.now
        # stop recording after all parcels have had time plus 30s buffer
        if t > max(p['arrival'] for p in parcels) + 30:
            return

        for p in parcels:
            if p['arrival'] <= t and p['id'] not in results['sorted']:
                elapsed = t - p['arrival']
                dist    = (elapsed * belt_speed) % loop_len
                recorder.append({
                    'time_s':    elapsed,
                    'parcel_id': p['id'],
                    'dist_m':    dist
                })
        yield env.timeout(DT_RECORD)

def main():
    # --- Load Excel data ---
    xls       = pd.ExcelFile('PosiSorterData1.xlsx')
    parcelsDf = xls.parse('Parcels')
    layoutDf  = xls.parse('Layout')

    # Build parcels list, normalizing arrival to seconds from zero
    first_arr = pd.to_datetime(parcelsDf['Arrival Time']).min()
    parcels = []
    for _, row in parcelsDf.iterrows():
        arr_s     = (pd.to_datetime(row['Arrival Time']) - first_arr).total_seconds()
        feasible  = [i for i,f in enumerate((row['Outfeed 1'], row['Outfeed 2'], row['Outfeed 3'])) if f]
        parcels.append({
            'id':                int(row['Parcel Number']),
            'arrival':           arr_s,
            'length':            float(row['Length']),
            'width':             float(row['Width']),
            'height':            float(row['Height']),
            'weight':            float(row['Weight']),
            'feasible_outfeeds': feasible
        })

    # --- Set up SimPy environment ---
    layout         = load_layout(layoutDf)
    outfeed_states = [{'length': 0.0} for _ in range(3)]
    results        = {'sorted': [], 'recirculated': []}
    recorder       = []

    env   = simpy.Environment()
    end_t = max(p['arrival'] for p in parcels) + 30

    # Start the recorder
    env.process(recorder_process(env, parcels, layout, results, recorder))

    # Schedule each parcel's process at its arrival time
    for p in parcels:
        def start(env, p=p):
            yield env.timeout(p['arrival'])
            yield env.process(parcel_process(env, p, layout, outfeed_states, results))
        env.process(start(env))

    # Run simulation
    env.run(until=end_t)

    # --- Summary & CSV dump ---
    total   = len(parcels)
    sorted_ = len(results['sorted'])
    recirc_ = len(results['recirculated'])
    print(f"Total parcels:    {total}")
    print(f"Sorted:           {sorted_}")
    print(f"Recirculated:     {recirc_}")

    df = pd.DataFrame(recorder)
    df.to_csv('positions.csv', index=False)
    print("Wrote positions.csv")

if __name__ == '__main__':
    main()
