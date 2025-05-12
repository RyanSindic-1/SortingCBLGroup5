# viz_plotly.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def load_layout_geometry(xlsx_path):
    """Read Layout sheet and return belt geometry."""
    df = pd.ExcelFile(xlsx_path).parse('Layout')
    # distances (meters)
    d_scan = float(df.loc[df['Layout property']=='Distance Infeeds to Scanner','Value'])
    d_s2o  = float(df.loc[df['Layout property']=='Distance Scanner to Outfeeds','Value'])
    d_betw = float(df.loc[df['Layout property']=='Distance between Outfeeds','Value'])
    d_back = float(df.loc[df['Layout property']=='Distance Outfeeds to Infeeds','Value'])
    # bottom straight length and return straight (symmetrical)
    Lb = d_scan + d_s2o + 2*d_betw
    Lr = Lb
    # semicircle radius so that 2πr + Lr = d_back
    r = max((d_back - Lr)/(2*np.pi), 0.1*Lb)
    L_loop = Lb + 2*np.pi*r + Lr
    return {'Lb':Lb, 'Lr':Lr, 'r':r, 'L':L_loop}

def dist_to_xy(s, geom):
    """Map scalar distance s along the loop to x,y coordinates."""
    Lb, Lr, r, L = geom['Lb'], geom['Lr'], geom['r'], geom['L']
    m = s % L
    # bottom straight
    if m < Lb:
        return (m, 0.0)
    m -= Lb
    # right semicircle
    if m < np.pi*r:
        θ = -np.pi/2 + m/r
        return (Lb + r*np.cos(θ), r + r*np.sin(θ))
    m -= np.pi*r
    # top straight (leftwards)
    if m < Lr:
        return (Lb - m, 2*r)
    m -= Lr
    # left semicircle
    θ = np.pi/2 + m/r
    return (r*np.cos(θ), r + r*np.sin(θ))

def build_belt_trace(geom, samples=300):
    """Create a Plotly trace for the belt outline."""
    s = np.linspace(0, geom['L'], samples)
    xy = np.array([dist_to_xy(si, geom) for si in s])
    return go.Scatter(
        x=xy[:,0], y=xy[:,1],
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    )

def main():
    # --- Load data ---
    CSV    = 'positions.csv'
    XLSX   = 'PosiSorterData1.xlsx'
    pos    = pd.read_csv(CSV)
    geom   = load_layout_geometry(XLSX)

    # compute x,y for each record
    pos[['x','y']] = pd.DataFrame(
        pos['dist_m'].apply(lambda d: dist_to_xy(d, geom)).tolist(),
        index=pos.index
    )

    # downsample time steps (e.g. keep every 3rd)
    times_all = sorted(pos['time_s'].unique())
    times_ds  = times_all[::3]
    pos        = pos[pos['time_s'].isin(times_ds)]

    # convert time to string for Plotly slider
    pos['time_str'] = pos['time_s'].round(2).astype(str)

    # --- Build the animated scatter ---
    fig = px.scatter(
        pos,
        x='x', y='y',
        animation_frame='time_str',
        animation_group='parcel_id',
        hover_name='parcel_id',
        range_x=[-1, geom['Lb']+1],
        range_y=[-1, 2*geom['r']+1],
        title='2D Conveyor Simulation (Plotly)',
        labels={'time_str':'Time (s)'}
    )

    # overlay belt outline
    belt = build_belt_trace(geom, samples=500)
    fig.add_trace(belt)

    # annotate decision points
    ddf = pd.ExcelFile(XLSX).parse('Layout')
    d_scan = float(ddf.loc[ddf['Layout property']=='Distance Infeeds to Scanner','Value'])
    d_s2o  = float(ddf.loc[ddf['Layout property']=='Distance Scanner to Outfeeds','Value'])
    d_betw = float(ddf.loc[ddf['Layout property']=='Distance between Outfeeds','Value'])
    ddists = [
        0.0,
        d_scan,
        d_scan + d_s2o,
        d_scan + d_s2o + d_betw,
        d_scan + d_s2o + 2*d_betw
    ]
    dp_labels = ['Infeed','Scanner','Outfeed 1','Outfeed 2','Outfeed 3']
    dp_xy = np.array([dist_to_xy(d, geom) for d in ddists])
    for (x,y), lbl in zip(dp_xy, dp_labels):
        fig.add_annotation(
            x=x, y=y,
            text=lbl,
            showarrow=False,
            font=dict(color='red', size=10),
            yshift=10
        )

    # add Play/Pause buttons
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {
                  'label': 'Play',
                  'method': 'animate',
                  'args': [None, {'frame':{'duration':100}, 'fromcurrent':True}]
                },
                {
                  'label': 'Pause',
                  'method': 'animate',
                  'args': [[None], {'mode':'immediate'}]
                }
            ],
            'pad': {'r':10, 't':10}
        }]
    )

    fig.show()

if __name__ == '__main__':
    main()
