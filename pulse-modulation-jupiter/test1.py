import plot_data as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Initial parameters
f0 = 3 * (10**9)                # central freq of CW  3GHz
delta_f = 0                     # max - min freq diff 0MHz
tu = 1*10**(-12)                # time units (time resolution) 1ps
fs = 1/tu                       # sampling freq of simulation

L_sec = 1000 * (10**(-9))       # length of impulse   100ns
s_sec = 0 * (10**(-9))          # start of impulse    0ns
sim_dur = 1000 * (10**(-9))     # simulation duration 100ns
N = sim_dur / tu                # amount of samples
f_res = fs / N                  # FFT bin size

L = int(L_sec/tu)               # length of impulse in time units
s = int(s_sec/tu)               # start of impulse in time units
e = s + L                       # end of simulation

t = np.arange(0, sim_dur, tu)

print(f"FFT Bin Size={f_res/1000000:.2f}Mhz")

# create the CW impulse
rf = np.zeros(t.shape)
rf[s:e] = np.sin(2*np.pi*f0*t[s:e])

data = pd.PlotData(rf, fs)
data.trim_freq(-5 * 10**9, 5 * 10**9)
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=data.freq_domain_x, y=data.freq_domain, name='RF CW spectrum'),
              row=1, col=1)
pd.mark_peaks(fig, data, row=1, col=1)
pd.mark_bands(fig, data, row=1, col=1)

fig.update_layout(hovermode='x unified', height=500)
fig.show()
