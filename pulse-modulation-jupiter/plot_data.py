import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter


SPAN = 20


class Signal:
    tu = None
    fs = None

    sim_dur = None  # simulation duration in seconds
    N = None  # amount of samples
    f_res = None  # FFT bin size in Hz

    t = None  # np array of time values
    data = None  # np array of signal values

    def __init__(self, tu, sim_dur):
        self.tu = tu
        self.fs = 1 / self.tu
        self.sim_dur = sim_dur  # simulation duration
        self.N = self.sim_dur / self.tu  # amount of samples
        self.f_res = self.fs / self.N  # FFT bin size
        self.t = np.arange(0, self.sim_dur, self.tu)


class LfmSignal:
    def __init__(self,
                 f0=3 * 10 ** 9, delta_f=0, tu=1 * 10 ** (-12),
                 l_sec=1000 * (10 ** (-9)), s_sec=0, sim_dur=1000 * (10 ** (-9))):
        self.sig = Signal(tu, sim_dur)

        # Initial parameters
        self.f0 = f0                # center freq
        self.delta_f = delta_f      # freq span
        self.f1 = f0 - delta_f / 2  # start freq
        self.f2 = f0 + delta_f / 2  # end freq

        self.L_sec = l_sec  # length of impulse
        self.s_sec = s_sec  # start of impulse

        self.L = int(self.L_sec / self.sig.tu)  # length of impulse in time units
        self.s = int(self.s_sec / self.sig.tu)  # start of impulse in time units
        self.e = self.s + self.L - 1        # end of simulation

        # create the freq modulated impulse
        self.sig.data = np.zeros(self.sig.t.shape)
        k = self.delta_f / (self.sig.t[self.e] - self.sig.t[self.s])
        self.sig.data[self.s:self.e] = (
            np.cos(2 * np.pi
                   * (self.f0 - self.delta_f / 2 + k * self.sig.t[self.s:self.e] / 2)
                   * self.sig.t[self.s:self.e]))


def not_close_to(value_to_check, values, span):
    for val in values:
        if abs(value_to_check - val) < span:
            return False
    return True


def find_peaks(data, number, span):
    if number*span > data.size:
        span = data.size // number

    peaks_indexes_unfiltered = np.argsort(data)[-number*span:]  # highest peaks

    peaks_indexes = []
    i = number*span-1
    while i >= 0:
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
    while l >= 0 and data[l] > thrsd:
        l = l - 1

    r = peak_index + 1
    while r < len(data) and data[r] > thrsd:
        r = r + 1

    return [l, r]


class FreqBand:
    def __init__(self, band_edges):
        self.freq_start = band_edges[0]
        self.freq_stop = band_edges[1]


class PlotData:
    def __init__(self, data, fs):
        self.is_complex = np.iscomplexobj(data)

        self.time_domain = data
        self.time_domain_x = np.linspace(0, data.size / fs, data.size)

        spectrum = convert_fft_to_zero_symmetrical(np.fft.fft(data))
        self.freq_domain = np.abs(spectrum)
        self.freq_domain_x = convert_fft_to_zero_symmetrical(np.fft.fftfreq(data.size, 1 / fs))

        self.freq_peaks = []
        self.freq_bands = []
        self.update_freq_peaks()
        self.update_freq_bands()

    def update_freq_peaks(self):
        self.freq_peaks = []
        freq_peaks = find_peaks(self.freq_domain, 4, SPAN)

        # filter out peaks less than 0.5 of the main peak
        for peak in freq_peaks:
            if self.freq_domain[peak] / self.freq_domain[freq_peaks[0]] > 0.5:
                self.freq_peaks.append(peak)

    def update_freq_bands(self):
        self.freq_bands = []
        for peak in self.freq_peaks:
            band_edges = find_band(self.freq_domain, peak, 0.7)
            self.freq_bands.append(FreqBand(band_edges))

    @staticmethod
    def get_freq_range_indexes(freqs, freq_start, freq_end):
        s = np.searchsorted(freqs, freq_start)
        e = np.searchsorted(freqs, freq_end)
        return [s, e] if (e  == (len(freqs) - 1)) else [s, e + 1]
    
    def trim_freq(self, freq_start, freq_end):
        s, e = self.get_freq_range_indexes(self.freq_domain_x, freq_start, freq_end)
        self.freq_domain_x = self.freq_domain_x[s:e]
        self.freq_domain = self.freq_domain[s:e]
        self.update_freq_peaks()
        self.update_freq_bands()
        return [s, e]


def mark_point_plotly(figure, x, y, text, **kwargs):
    figure.add_annotation(x=x, y=y,
                          text=text,
                          arrowhead=1,
                          **kwargs)


def mark_peaks(figure, data, **kwargs):
    even = True
    for peak in data.freq_peaks:
        even = not even
        mark_point_plotly(figure,
                          data.freq_domain_x[peak],
                          data.freq_domain[peak],
                          f"{data.freq_domain_x[peak] / 1_000_000:.2f}MHz",
                          ax=(50 if even else -50))


def mark_bands(figure, data, **kwargs):
    for band in data.freq_bands:
        x_l = data.freq_domain_x[band.freq_start]
        y_l = data.freq_domain[band.freq_start]
        x_r = data.freq_domain_x[band.freq_stop]
        y_r = data.freq_domain[band.freq_stop]
        mark_point_plotly(figure, x_l, y_l,
                          f"{x_l / 1_000_000:.2f}MHz",
                          ax=-50)
        mark_point_plotly(figure, x_r, y_r,
                          f"{x_r / 1_000_000:.2f}MHz",
                          ax=50)
        mark_point_plotly(figure, x_r, y_r,
                          f"dF={(x_r - x_l) / 1_000_000:.2f}MHz",
                          xshift=50, yshift=-30,
                          showarrow=False,
                          **kwargs)


def butter_lowpass_filter(data, fs, cutoff, order):
    nyq = fs/2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


################################################################################
# matplolib specific functions
# keep it for future usage
def mark_points_matplotlib(x, y, points_indexes, ax):
    for i in range(len(points_indexes)):
        x_point = x[points_indexes[i]]
        y_point = y[points_indexes[i]]
        ax.annotate(f"{x_point / 1_000_000:.2f}MHz",
                    xy=(x_point, y_point))


def mark_band_matplotlib(x, y, edges, ax):
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
    data_is_complex = np.iscomplexobj(data)
    fig, axs = plt.subplots(3 if data_is_complex else 2, figsize=(10, 9 if data_is_complex else 6), num=plot_name)
    fig.suptitle(plot_name)
    spectrum = convert_fft_to_zero_symmetrical(np.fft.fft(data))
    t = np.linspace(0, data.size/fs, data.size)
    f = convert_fft_to_zero_symmetrical(np.fft.fftfreq(data.size, 1/fs))
    sp_abs = np.abs(spectrum)

    if data_is_complex:
        axs[0].plot(t, data.real)
        axs[1].plot(t, data.imag)
        axs[2].plot(f, sp_abs)
        freq_axs = axs[2]
    else:
        axs[0].plot(t, data)
        axs[1].plot(f, sp_abs)
        freq_axs = axs[1]

    freq_axs.set_xlim(-10 * 10 ** 9, 10 * 10 ** 9)
    freq_peaks = find_peaks(sp_abs, 4, SPAN)

    for peak in freq_peaks:
        # mark all bands that have max amplitude at least 0.5 of main peak
        if sp_abs[peak] / sp_abs[freq_peaks[0]] > 0.5:
            mark_points_matplotlib(f, sp_abs, [peak], freq_axs)
            band_edges = find_band(sp_abs, peak, 0.7)
            mark_band_matplotlib(f, sp_abs, band_edges, freq_axs)
