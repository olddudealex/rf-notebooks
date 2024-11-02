import copy

import plot_data as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import numpy as np
import scipy.signal as sp

df1 = 5 * (10**6)
df2 = 10 * (10**6)
T1 = 5 * (10**(-6))
T2 = 25 * (10**(-6))

s1 = pd.LfmSignal(f0=3*(10**9), delta_f=df1, tu=10**(-11),
                  l_sec=T1, s_sec=0, sim_dur=T1)
s2 = pd.LfmSignal(f0=3*(10**9), delta_f=df2, tu=10**(-11),
                  l_sec=T2, s_sec=0, sim_dur=T2)

cor1 = sp.correlate(s1.sig.data, s1.sig.data, "full")
cor2 = sp.correlate(s2.sig.data, s2.sig.data, "full")

fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(y=np.power(cor1, 2),
                         name=f"Autocorrelation for T*dF={round(T1*df1)}"), row=1, col=1)
fig.add_trace(go.Scatter(y=np.power(cor2, 2),
                         name=f"Autocorrelation for T*dF={round(T2*df2)}"), row=2, col=1)

fig.update_layout(hovermode='x unified', height=500)
#fig.update_xaxes(exponentformat="SI")
#fig.update_traces(mode="lines")
fig.show()
