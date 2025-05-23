import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# -------------------------------------------------------------------------- #
# LOAD AND PREPROCESS DATA                                                   #
# -------------------------------------------------------------------------- #
def load_data(events_path='events.csv', summary_path='parcel_summary.csv'):
    ev = pd.read_csv(events_path)
    ps = pd.read_csv(summary_path)

    # Convert event times
    ev['time_td'] = pd.to_timedelta(ev['time'], errors='coerce')
    ev['time_s']  = ev['time_td'].dt.total_seconds()

    # Convert summary times
    for col in ['arrival_time', 'scanner_time', 'outfeed_enter', 'outfeed_exit']:
        ps[f'{col}_td'] = pd.to_timedelta(ps[col], errors='coerce')
    ps['arrival_s']       = ps['arrival_time_td'].dt.total_seconds()
    ps['scanner_s']       = ps['scanner_time_td'].dt.total_seconds()
    ps['outfeed_enter_s'] = ps['outfeed_enter_td'].dt.total_seconds()
    ps['outfeed_exit_s']  = ps['outfeed_exit_td'].dt.total_seconds()

    # Compute time in system
    ps['time_in_system']  = ps['outfeed_exit_s'] - ps['arrival_s']

    # Numeric IDs
    ev['parcel_id']   = pd.to_numeric(ev['parcel_id'], errors='coerce').fillna(-1).astype(int)
    ev['outfeed_id']  = pd.to_numeric(ev.get('outfeed_id', -1), errors='coerce').fillna(-1).astype(int)
    ps['parcel_id']   = pd.to_numeric(ps.get('parcel_id', -1), errors='coerce').fillna(-1).astype(int)
    ps['num_outfeeds']= pd.to_numeric(ps.get('num_outfeeds', 0), errors='coerce').fillna(0).astype(int)

    return ev, ps

# -------------------------------------------------------------------------- #
# DASHBOARD CREATION                                                         #
# -------------------------------------------------------------------------- #
def create_dashboard(ev, ps):
    app = Dash(__name__)

    ev_exit = ev[ev['event_name'] == 'ExitOutfeed'].sort_values('time_s')
    max_time = ev['time_s'].max()
    n_out = ps['num_outfeeds'].iat[0]

    # Conveyor layout positions
    if n_out > 1:
        y_positions = {i: 1 - 2 * i/(n_out-1) for i in range(n_out)}
    else:
        y_positions = {0: 0}

    # Layout: hide slider, use only Interval for real-time updates
    app.layout = html.Div([
        html.H1('PosiSorter Interactive Dashboard'),
        dcc.Interval(
            id='interval-component',
            interval=500,    # 500ms per update
            n_intervals=0
        ),
        dcc.Graph(id='anim-graph'),
        html.Div([
            dcc.Graph(id='throughput-cumulative'),
            dcc.Graph(id='bar-outfeed-counts')
        ], style={'display': 'flex'}),
        html.Div([
            dcc.Graph(id='hist-recirculations'),
            dcc.Graph(id='scatter-time-in-system')
        ], style={'display': 'flex'})
    ], style={'padding': '20px'})

    # ---------------------------------------------------------------------- #
    # ANIMATION CALLBACK                                                   #
    # ---------------------------------------------------------------------- #
    @app.callback(
        Output('anim-graph', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_animation(n):
        # calculate simulation time
        step = max_time / 200
        t = (n * step) % (max_time + step)
        scanner_frac = 0.3

        # map parcels to positions
        df = ps.copy()
        mapping = ev_exit.groupby('parcel_id')['outfeed_id'].first()
        df['outfeed_id'] = df['parcel_id'].map(mapping).fillna(-1).astype(int)
        df['recirculations'] = df.get('recirculations', pd.Series(0))

        positions = []
        for _, row in df.iterrows():
            arr, scn, ent, ext, oid = (
                row['arrival_s'], row['scanner_s'],
                row['outfeed_enter_s'], row['outfeed_exit_s'], row['outfeed_id']
            )
            if pd.isna(arr) or t < arr:
                continue
            if t < scn:
                frac = (t - arr) / (scn - arr)
                x, y = frac * scanner_frac, 0
            elif t < ent:
                x, y = scanner_frac, 0
            elif t < ext:
                frac = (t - ent) / (ext - ent)
                x = scanner_frac + frac * (1 - scanner_frac)
                y = frac * y_positions.get(oid, 0)
            else:
                if row['recirculations'] > 0:
                    cycle = 50.0
                    progress = ((t - ext) % cycle) / cycle
                    x, y = progress * scanner_frac, 0
                else:
                    x, y = scanner_frac + (1 - scanner_frac), y_positions.get(oid, 0)
            positions.append({'parcel_id': row['parcel_id'], 'x': x, 'y': y, 'recirculations': row['recirculations']})
        pos_df = pd.DataFrame(positions)
        pos_df['is_recirc'] = pos_df['recirculations'] > 0

        fig = px.scatter(
            pos_df,
            x='x', y='y',
            color='recirculations',
            symbol='is_recirc', symbol_map={False: 'circle', True: 'x'},
            color_continuous_scale='Turbo',
            hover_data=['parcel_id'],
            title=f'Parcel Positions at t={t:.1f}s',
            size_max=12  # this can stay or be removed
        )

        # 1. Make the points larger
        fig.update_traces(marker=dict(size=20))

        # 2. Draw conveyor lines behind the scatter points
        shapes = [
            dict(
                type='line',
                x0=0, y0=0, x1=scanner_frac, y1=0,
                line=dict(width=2, color='black'),
                layer='below'               # ← send behind the points
            )
        ]
        for y in y_positions.values():
            shapes.append(
                dict(
                    type='line',
                    x0=scanner_frac, y0=0, x1=1, y1=y,
                    line=dict(width=2, color='black'),
                    layer='below'           # ← same here
                )
            )

        fig.update_layout(
            shapes=shapes,
            plot_bgcolor='white',
            xaxis=dict(range=[-0.1, 1.1], showgrid=False, zeroline=False),
            yaxis=dict(range=[-1.1, 1.1], showgrid=False, zeroline=False)
        )
        return fig

    # ---------------------------------------------------------------------- #
    # STATIC CHART CALLBACKS                                                #
    # ---------------------------------------------------------------------- #
    @app.callback(
        Output('throughput-cumulative', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_throughput(_):
        df = ev_exit.copy()
        df['cumulative'] = df.groupby('outfeed_id').cumcount() + 1
        fig = px.line(df, x='time_s', y='cumulative', color='outfeed_id', markers=True, title='Cumulative Throughput')
        fig.update_layout(xaxis_title='Time (s)', yaxis_title='Parcels Processed')
        return fig

    @app.callback(
        Output('bar-outfeed-counts', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_bar(_):
        counts = ev_exit['outfeed_id'].value_counts().sort_index().reset_index()
        counts.columns = ['outfeed_id', 'count']
        fig = px.bar(counts, x='outfeed_id', y='count', title='Total Parcels per Outfeed')
        fig.update_layout(xaxis_title='Outfeed ID', yaxis_title='Parcel Count')
        return fig

    @app.callback(
        Output('hist-recirculations', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_hist(_):
        fig = px.histogram(ps, x='recirculations', nbins=int(ps['recirculations'].max()+1), title='Recirculation Distribution')
        fig.update_layout(xaxis_title='Recirculations', yaxis_title='Count')
        return fig

    @app.callback(
        Output('scatter-time-in-system', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_scatter(_):
        fig = px.scatter(ps, x='arrival_s', y='time_in_system', title='Time in System vs Arrival Time')
        fig.update_layout(xaxis_title='Arrival Time (s)', yaxis_title='Time in System (s)')
        return fig

    return app

# -------------------------------------------------------------------------- #
# RUN SERVER                                                                 #
# -------------------------------------------------------------------------- #
if __name__ == '__main__':
    ev_df, ps_df = load_data()
    app = create_dashboard(ev_df, ps_df)
    app.run(debug=True)
