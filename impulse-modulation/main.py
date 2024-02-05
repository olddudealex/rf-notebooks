import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter


SPAN = 10


def not_close_to(value_to_check, values, span):
    for val in values:
        if abs(value_to_check - val) < span:
            return False
    return True


def find_peaks(data, number, span):
    peaks_indexes_unfiltered = np.argsort(data)[-number*span:]  # highest peaks

    peaks_indexes = []
    i = number*span-1
    while True:
        peaks_indexes.append(peaks_indexes_unfiltered[i])
        i = i - 1
        while True:
            if not_close_to(peaks_indexes_unfiltered[i], peaks_indexes, span):
                break
            elif i >= 0:
                i = i - 1
            else:
                break
        if i == 0 or len(peaks_indexes) == number:
            break
    return peaks_indexes


def convert_fft_to_zero_symmetrical(fft_raw):
    m = fft_raw.size // 2
    data = np.concatenate([fft_raw[m+1:], fft_raw[0:m]])
    return data


def find_band(data, peak_index, level=0.7):
    max_val = data[peak_index]
    thrsd = level * max_val

    l = peak_index - 1
    while data[l] > thrsd:
        l = l - 1

    r = peak_index + 1
    while data[r] > thrsd:
        r = r + 1

    return [l, r]


def mark_points(x, y, points_indexes, ax):
    for i in range(len(points_indexes)):
        x_point = x[points_indexes[i]]
        y_point = y[points_indexes[i]]
        ax.annotate(f"{x_point / 1_000_000:.2f}MHz",
                    xy=(x_point, y_point))


def mark_band(x, y, edges, ax):
    x_l = x[edges[0]]
    y_l = y[edges[0]]
    ax.annotate(f"{x_l / 1_000_000:.2f}MHz",
                xy=(x_l, y_l),
                xytext=(-70, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle="->"))
    x_r = x[edges[1]]
    y_r = y[edges[1]]
    ax.annotate(f"{x_r / 1_000_000:.2f}MHz",
                xy=(x_r, y_r),
                xytext=(20, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle="->"))
    ax.annotate(f"dF={(x_r-x_l) / 1_000_000:.2f}MHz",
                xy=(x_r, y_r),
                xytext=(20, -10), textcoords='offset points')


def plot_with_spectre(data, fs, plot_name):
    fig, axs = plt.subplots(2, figsize=(10, 6), num=plot_name)
    fig.suptitle(plot_name)
    spectrum = convert_fft_to_zero_symmetrical(np.fft.fft(data))
    t = np.linspace(0, data.size/fs, data.size)
    f = convert_fft_to_zero_symmetrical(np.fft.fftfreq(data.size, 1/fs))
    sp_abs = np.abs(spectrum)
    axs[0].plot(t, data)
    axs[1].plot(f, sp_abs)
    axs[1].set_xlim(-10 * 10 ** 9, 10 * 10 ** 9)
    freq_peaks = find_peaks(sp_abs, 5, SPAN)

    for peak in freq_peaks:
        # mark all bands that have max amplitude at least 0.5 of main peak
        if sp_abs[peak] / sp_abs[freq_peaks[0]] > 0.5:
            mark_points(f, sp_abs, [peak], axs[1])
            band_edges = find_band(sp_abs, peak, 0.7)
            mark_band(f, sp_abs, band_edges, axs[1])

            # band_edges_2 = find_band(sp_abs[sp_abs.size // 2 + 1:], 0.7)
            # for i in range(len(band_edges_2)):
            #     band_edges_2[i] += sp_abs.size // 2 + 1
            # mark_band(f, sp_abs, band_edges_2, axs[1])


def butter_lowpass_filter(data, fs, cutoff, order):
    nyq = fs/2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


# Initial parameters
f0 = 3 * (10**9)                # central freq of CW  3GHz
delta_f = 100 * (10**6)         # max - min freq diff 100MHz
tu = 1*10**(-12)                # time units (time resolution) 1ps
fs = 1/tu                       # sampling freq of simulation

L_sec = 10 * (10**(-9))         # length of impulse   10ns
s_sec = 10 * (10**(-9))         # start of impulse    10ns
sim_dur = 100 * (10**(-9))      # simulation duration 100ns
N = sim_dur / tu                # amount of samples
f_res = fs / N                  # FFT bin size

L = int(L_sec/tu)               # length of impulse in time units
s = int(s_sec/tu)               # start of impulse in time units
e = s + L                       # end of simulation

t = np.arange(0, sim_dur, tu)

# make simple window function

wind = np.zeros(t.shape)
wind[s:e] = 1
plot_with_spectre(wind, fs, "Rectangular impulse")

# create the CW impulse
rf = np.zeros(t.shape)
rf[s:e] = np.sin(2*np.pi*f0*t[s:e])
plot_with_spectre(rf, fs, "RF CW impulse")

# create the freq modulated impulse
f = np.linspace(f0-delta_f/2, f0+delta_f/2, L)
rf = np.zeros(t.shape)
rf[s:e] = np.sin(2*np.pi*f*t[s:e])
plot_with_spectre(rf, fs, "RF modulated impulse")

# now let's try to make regular down conversion
bs1 = rf * np.cos(2*np.pi*f0*t)
plot_with_spectre(bs1, fs, "Regular BB signal")

# filter the highs
bs2 = butter_lowpass_filter(bs1, fs, 500*10**6, 2)
plot_with_spectre(bs2, fs, "Regular BB signal Filtered")

# apply IQ demodulation
I_bs = rf * np.cos(2*np.pi*f0*t)
Q_bs = rf * np.sin(2*np.pi*f0*t)

plot_with_spectre(I_bs, fs, "I BB signal")
plot_with_spectre(Q_bs, fs, "Q BB signal")

IQ_bs = I_bs + 1j*Q_bs
plot_with_spectre(IQ_bs, fs, "I + j*Q")

IQ_bs2 = I_bs - 1j*Q_bs
plot_with_spectre(IQ_bs2, fs, "I - j*Q")

plt.show()
