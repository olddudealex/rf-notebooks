import plot_data as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

rf = pd.LfmSignal(f0=3*(10**9), delta_f=100*(10**6), tu=10**(-12),
                  l_sec=1000*(10**(-9)), s_sec=0, sim_dur=1000*(10**(-9)))

print(f"FFT Bin Size={rf.f_res/1000000:.2f}Mhz")

data = pd.PlotData(rf.data, rf.fs)
data.trim_freq(-5 * 10 ** 9, 5 * 10 ** 9)
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=data.freq_domain_x,
                         y=data.freq_domain,
                         name='RF modulated impulse spectrum'),
              row=1, col=1)
peaks = data.freq_peaks
pd.mark_peaks(fig, data, row=1, col=1)
pd.mark_bands(fig, data, row=1, col=1)

fig.update_layout(hovermode='x unified', height=800)
fig.show()
