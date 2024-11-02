import copy

import plot_data as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import numpy as np

df1 = 4 * (10**6)      # 4Mhz
df2 = 10 * (10**6)     # 10Mhz
T1 = 5 * (10**(-6))    # 5uS
T2 = 20 * (10**(-6))   # 10uS

s1 = pd.LfmSignal(f0=3*(10**9), delta_f=df1, tu=10**(-12),
                  l_sec=T1, s_sec=0, sim_dur=T1)
s2 = pd.LfmSignal(f0=3*(10**9), delta_f=df2, tu=10**(-12),
                  l_sec=T2, s_sec=0, sim_dur=T2)

data1 = pd.PlotData(s1.data, s1.fs)
data1.trim_freq(s1.f0 - s1.delta_f, s1.f0 + s1.delta_f)
data1_1 = copy.deepcopy(data1)
data1_2 = copy.deepcopy(data1)
data1_3 = copy.deepcopy(data1)
data1_1.trim_freq(s1.f0 - s1.delta_f, s1.f1)
data1_2.trim_freq(s1.f1, s1.f2)
data1_3.trim_freq(s1.f2, s1.f0 + s1.delta_f)

data2 = pd.PlotData(s2.data, s2.fs)
data2.trim_freq(s2.f0 - s2.delta_f, s2.f0 + s2.delta_f)
data2_1 = copy.deepcopy(data2)
data2_2 = copy.deepcopy(data2)
data2_3 = copy.deepcopy(data2)
data2_1.trim_freq(s2.f0 - s2.delta_f, s2.f1)
data2_2.trim_freq(s2.f1, s2.f2)
data2_3.trim_freq(s2.f2, s2.f0 + s2.delta_f)

fig = make_subplots(rows=2, cols=2)
fig.add_trace(go.Scatter(x=data1_1.freq_domain_x,
                         y=data1_1.freq_domain,
                         line_color="#6A6BE1",
                         fill="tozeroy",
                         fillcolor="#4CDBB3",
                         name=f"T*dF={round(T1*df1)}, out of LFM band"), row=1, col=1)
fig.add_trace(go.Scatter(x=data1_2.freq_domain_x,
                         y=data1_2.freq_domain,
                         line_color="#6A6BE1",
                         fill="tozeroy",
                         fillcolor="#D9F7EF",
                         name=f"T*dF={round(T1*df1)}, in the LFM band"), row=1, col=1)
fig.add_trace(go.Scatter(x=data1_3.freq_domain_x,
                         y=data1_3.freq_domain,
                         line_color="#6A6BE1",
                         fill="tozeroy",
                         fillcolor="#4CDBB3",
                         showlegend=False), row=1, col=1)

fig.add_trace(go.Scatter(x=data2_1.freq_domain_x,
                         y=data2_1.freq_domain,
                         line_color="#00CC92",
                         fill="tozeroy",
                         fillcolor="#4CDBB3",
                         name=f"T*dF={round(T2*df2)}, out of LFM band"), row=1, col=2)
fig.add_trace(go.Scatter(x=data2_2.freq_domain_x,
                         y=data2_2.freq_domain,
                         line_color="#00CC92",
                         fill="tozeroy",
                         fillcolor="#D9F7EF",
                         name=f"T*dF={round(T2*df2)}, in the LFM band"), row=1, col=2)
fig.add_trace(go.Scatter(x=data2_3.freq_domain_x,
                         y=data2_3.freq_domain,
                         line_color="#00CC92",
                         fill="tozeroy",
                         fillcolor="#4CDBB3",
                         showlegend=False), row=1, col=2)

fig.add_trace(go.Scatter(x=data1_1.freq_domain_x,
                         y=np.power(data1_1.freq_domain, 2),
                         line_color="#F7573F",
                         fill="tozeroy",
                         fillcolor="#4CDBB3",
                         name=f"T*dF={round(T1*df1)}, out of LFM band (power)"), row=2, col=1)
fig.add_trace(go.Scatter(x=data1_2.freq_domain_x,
                         y=np.power(data1_2.freq_domain, 2),
                         line_color="#F7573F",
                         fill="tozeroy",
                         fillcolor="#D9F7EF",
                         name=f"T*dF={round(T1*df1)}, in the LFM band (power)"), row=2, col=1)
fig.add_trace(go.Scatter(x=data1_3.freq_domain_x,
                         y=np.power(data1_3.freq_domain, 2),
                         line_color="#F7573F",
                         fill="tozeroy",
                         fillcolor="#4CDBB3",
                         showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=data2_1.freq_domain_x,
                         y=np.power(data2_1.freq_domain, 2),
                         line_color="#AC63FB",
                         fill="tozeroy",
                         fillcolor="#4CDBB3",
                         name=f"T*dF={round(T2*df2)}, out of LFM band (power)"), row=2, col=2)
fig.add_trace(go.Scatter(x=data2_2.freq_domain_x,
                         y=np.power(data2_2.freq_domain, 2),
                         line_color="#AC63FB",
                         fill="tozeroy",
                         fillcolor="#D9F7EF",
                         name=f"T*dF={round(T2*df2)}, in the LFM band (power)"), row=2, col=2)
fig.add_trace(go.Scatter(x=data2_3.freq_domain_x,
                         y=np.power(data2_3.freq_domain, 2),
                         line_color="#AC63FB",
                         fill="tozeroy",
                         fillcolor="#4CDBB3",
                         showlegend=False), row=2, col=2)

fig.update_layout(hovermode='x unified', height=500)
fig.update_xaxes(exponentformat="SI")
fig.update_traces(mode="lines")
fig.show()

print(f"For T*dF={int(T1*df1)} Pout/Pin="
      f"{100 * 2 * np.sum(np.power(data1_1.freq_domain, 2)) / np.sum(np.power(data1_2.freq_domain, 2)):.2f}%")

print(f"For T*dF={int(T2*df2)} Pout/Pin="
      f"{100 * 2 * np.sum(np.power(data2_1.freq_domain, 2)) / np.sum(np.power(data2_2.freq_domain, 2)):.2f}%")
