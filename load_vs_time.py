import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

# --- Data loading functions ---
def load_raw_data(filepath):
    """Loads raw time and load data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        if 'Load [N]' not in df.columns or 'Time [s]' not in df.columns:
            print(f"Required columns missing in {filepath}")
            return None, None
        df = df[['Time [s]', 'Load [N]']].rename(columns={'Time [s]': 'Time', 'Load [N]': 'Load'})
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        df = df.sort_values('Time').reset_index(drop=True)
        # Limit to 190 seconds
        df = df[df['Time'] <= 190]
        return df['Time'].values, df['Load'].values
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

def get_all_curves(folder):
    """Loads all curves from a folder without resampling or smoothing."""
    curves = []
    filepaths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')])
    for path in filepaths:
        time, load = load_raw_data(path)
        if time is not None and load is not None and len(time) > 0:
            curves.append((time, load))
    return curves

# --- Plot raw curves ---
def plot_all_individual_curves(folder_40, folder_80):
    """Plots all individual raw curves for 40- and 80-wire samples (x limited to 190)."""
    curves_40 = get_all_curves(folder_40)
    curves_80 = get_all_curves(folder_80)

    plt.figure(figsize=(10, 6))
    for t, l in curves_40:
        plt.plot(t, l, color='red', alpha=0.6)
    for t, l in curves_80:
        plt.plot(t, l, color='blue', alpha=0.6)

    plt.title("All Raw Compression Curves (No Smoothing)")
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Load (N)")
    plt.xlim(0, 190)
    plt.grid(True)
    plt.legend(["40 wires", "80 wires"])
    plt.tight_layout()
    plt.savefig("raw_compression_individual_curves.png")
    plt.show()
    plt.close()

# --- Plot average and std ---
def plot_average_std_no_interp(folder_40, folder_80):
    """Plots average and std envelopes using raw unaligned data up to 190 seconds."""
    def compute_avg_std(curves):
        min_len = min(len(t) for t, _ in curves)
        trimmed_loads = [load[:min_len] for _, load in curves]
        trimmed_times = [time[:min_len] for time, _ in curves]
        avg_time = np.mean(trimmed_times, axis=0)
        load_matrix = np.array(trimmed_loads)
        avg_load = np.mean(load_matrix, axis=0)
        std_load = np.std(load_matrix, axis=0)
        return avg_time, avg_load, std_load

    curves_40 = get_all_curves(folder_40)
    curves_80 = get_all_curves(folder_80)

    if not curves_40 or not curves_80:
        print("No valid curves found.")
        return

    t_40, mean_40, std_40 = compute_avg_std(curves_40)
    t_80, mean_80, std_80 = compute_avg_std(curves_80)

    plt.figure(figsize=(10, 6))
    plt.plot(t_40, mean_40, color='red', label='40 wires - mean')
    plt.fill_between(t_40, mean_40 - std_40, mean_40 + std_40, color='red', alpha=0.3)

    plt.plot(t_80, mean_80, color='blue', label='80 wires - mean')
    plt.fill_between(t_80, mean_80 - std_80, mean_80 + std_80, color='blue', alpha=0.3)

    plt.title("Average Compression Curves")
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Load (N)")
    plt.xlim(0, 190)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("raw_compression_avg_std.png")
    plt.show()
    plt.close()

# --- FFT and frequency analysis ---
def plot_fft(time, load, title="FFT of Load Data"):
    dt = np.mean(np.diff(time))  # sampling interval
    n = len(load)

    # FFT with DC offset removed
    load_fft = fft(load - np.mean(load))
    freq = fftfreq(n, dt)

    mask = freq > 0
    freq = freq[mask]
    power = np.abs(load_fft[mask])

    plt.figure(figsize=(10,6))
    plt.plot(freq, power)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    dominant_freq = freq[np.argmax(power)]
    print(f"Dominant frequency: {dominant_freq:.2f} Hz")
    return dominant_freq

# --- Bandpass filter ---
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# --- Main execution ---
if __name__ == "__main__":
    folder_40 = '40_wires_compression_data'
    folder_80 = '80_wires_compression_data'

    # Plot raw data
    plot_all_individual_curves(folder_40, folder_80)
    plot_average_std_no_interp(folder_40, folder_80)

    # Load first sample from 40 wires folder for FFT and filtering
    curves_40 = get_all_curves(folder_40)
    if curves_40:
        t, l = curves_40[0]

        print("Performing FFT on first 40-wire sample...")
        dominant_freq_40 = plot_fft(t, l, title="Fourier Transform of 40-wire Sample 1")

        # Sampling frequency
        dt = np.mean(np.diff(t))
        fs = 1 / dt

        # Set bandpass filter range - example removing noise near dominant_freq ±5Hz
        lowcut = max(0.1, dominant_freq_40 - 5)  # don't go below 0.1 Hz
        highcut = dominant_freq_40 + 5

        print(f"Applying bandpass filter to keep frequencies between {lowcut:.2f} and {highcut:.2f} Hz...")
        filtered_load = bandpass_filter(l, lowcut, highcut, fs)

        # Plot raw vs filtered
        plt.figure(figsize=(10,6))
        plt.plot(t, l, label='Before')
        plt.plot(t, filtered_load, label='After', linewidth=2)
        plt.xlabel('Displacement (mm)')
        plt.ylabel('Load (N)')
        plt.legend()
        plt.title('Load Data Before and After Bandpass Filtering')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No 40-wire samples found for FFT and filtering.")
