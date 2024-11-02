import plot_data as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

s1 = pd.LfmSignal(f0=3*(10**9), delta_f=10*(10**6), tu=10**(-12),
                  l_sec=1000*(10**(-9)), s_sec=0, sim_dur=1000*(10**(-9)))
s2 = pd.LfmSignal(f0=3*(10**9), delta_f=100*(10**6), tu=10**(-12),
                  l_sec=1000*(10**(-9)), s_sec=0, sim_dur=1000*(10**(-9)))
s3 = pd.LfmSignal(f0=3*(10**9), delta_f=250*(10**6), tu=10**(-12),
                  l_sec=1000*(10**(-9)), s_sec=0, sim_dur=1000*(10**(-9)))
s4 = pd.LfmSignal(f0=3*(10**9), delta_f=1000*(10**6), tu=10**(-12),
                  l_sec=1000*(10**(-9)), s_sec=0, sim_dur=1000*(10**(-9)))

data1 = pd.PlotData(s1.data, s1.fs)
data1.trim_freq(s1.f0 - s1.delta_f, s1.f0 + s1.delta_f)
data2 = pd.PlotData(s2.data, s2.fs)
data2.trim_freq(s2.f0 - s2.delta_f, s2.f0 + s2.delta_f)
data3 = pd.PlotData(s3.data, s3.fs)
data3.trim_freq(s3.f0 - s3.delta_f, s3.f0 + s3.delta_f)
data4 = pd.PlotData(s4.data, s4.fs)
data4.trim_freq(s4.f0 - s4.delta_f, s4.f0 + s4.delta_f)

fig = make_subplots(rows=2, cols=2)
fig.add_trace(go.Scatter(x=data1.freq_domain_x,
                         y=data1.freq_domain,
                         name='T*dF=10'), row=1, col=1)

fig.add_trace(go.Scatter(x=data2.freq_domain_x,
                         y=data2.freq_domain,
                         name='T*dF=100'), row=2, col=1)
fig.add_trace(go.Scatter(x=data3.freq_domain_x,
                         y=data3.freq_domain,
                         name='T*dF=250'), row=1, col=2)
fig.add_trace(go.Scatter(x=data4.freq_domain_x,
                         y=data4.freq_domain,
                         name='T*dF=1000'), row=2, col=2)

fig.update_layout(hovermode='x unified', height=500)
fig.show()
