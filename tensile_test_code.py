import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# -------------------- SETTINGS --------------------
folder_40 = "/Users/ranjinighosh/Desktop/Bhamla Lab (older data)/Summer_poster/40 wires"
folder_80 = "/Users/ranjinighosh/Desktop/Bhamla Lab (older data)/Summer_poster/80 wires"

load_threshold = 2.5
initial_height_mm = 20
max_strain = 110 / initial_height_mm
n_points = 500
# ---------------------------------------------------

def read_tensile_data(filepath):
    """Reads tensile-testing CSV, trims before threshold, converts to strain."""
    load = []
    dist = []

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()[16:]    # Skip header lines
            if len(lines) <= 2:
                print(f"⚠ Skipping (too few lines): {filepath}")
                return [], []

            reader = csv.DictReader(lines, delimiter='\t')
            for row in reader:
                try:
                    force = abs(float(row['Load [N]']))
                    d = float(row['Distance [mm]'])
                    load.append(force)
                    dist.append(d)
                except:
                    continue
    except:
        print(f"⚠ Could not read file: {filepath}")
        return [], []

    # Find point where load crosses threshold
    for i, L in enumerate(load):
        if L >= load_threshold:
            load_norm = [x - L for x in load[i:]]
            dist_norm = [d - dist[i] for d in dist[i:]]
            strain = np.array(dist_norm) / initial_height_mm
            return strain, np.array(load_norm)

    print(f"⚠ Threshold never reached: {filepath}")
    return [], []


def load_folder(folder):
    """Loads all CSVs inside a folder and interpolates to a uniform strain axis."""
    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        return np.linspace(0, max_strain, n_points), np.array([])

    files = sorted([x for x in os.listdir(folder) if x.endswith(".csv")])
    print(f"📂 {folder}: Found {len(files)} files")

    x_uniform = np.linspace(0, max_strain, n_points)
    curves = []

    for f in files:
        strain, load = read_tensile_data(os.path.join(folder, f))
        if len(strain) == 0:
            continue
        interp = np.interp(x_uniform, strain, load, left=np.nan, right=np.nan)
        curves.append(interp)

    curves = np.array(curves)
    print(f"✔ Loaded curves: {curves.shape[0]}")
    return x_uniform, curves


# ---------------- PLOTTING FUNCTIONS ----------------

def plot_combined(x, c40, c80):
    plt.figure(figsize=(10,6))

    for c in c40:
        plt.plot(x, c, color="red", alpha=0.5)
    for c in c80:
        plt.plot(x, c, color="blue", alpha=0.5)

    plt.xlabel("Strain")
    plt.ylabel("Load (N)")
    plt.title("All Tensile Curves")
    plt.grid(True)
    plt.savefig("combined_curves.png")
    plt.show()


def plot_average(x, c40, c80):
    if len(c40) > 0:
        avg40 = np.nanmean(c40, axis=0)
        plt.plot(x, avg40, color="red", lw=2, label="40-wire avg")

    if len(c80) > 0:
        avg80 = np.nanmean(c80, axis=0)
        plt.plot(x, avg80, color="blue", lw=2, label="80-wire avg")

    plt.xlabel("Strain")
    plt.ylabel("Load (N)")
    plt.title("Average Curves")
    plt.grid(True)
    plt.legend()
    plt.savefig("average_curves.png")
    plt.show()


def plot_std_envelope(x, curves, color, label):
    if len(curves) == 0:
        return
    avg = np.nanmean(curves, axis=0)
    std = np.nanstd(curves, axis=0)

    plt.fill_between(x, avg-std, avg+std, alpha=0.3, color=color, label=f"{label} ± std")
    plt.plot(x, avg, color=color, lw=2)


def plot_std_all(x, c40, c80):
    plt.figure(figsize=(10,6))

    plot_std_envelope(x, c40, "red", "40 wires")
    plot_std_envelope(x, c80, "blue", "80 wires")

    plt.xlabel("Strain")
    plt.ylabel("Load (N)")
    plt.title("Std. Deviation Envelopes")
    plt.grid(True)
    plt.legend()
    plt.savefig("std_envelopes.png")
    plt.show()


# -------------- MECHANICAL QUANTITIES -----------------

def compute_young_modulus(x, curves, percent=0.1):
    """Linear fit on first 10% of strain."""
    E_list = []
    cutoff = int(percent * len(x))

    for c in curves:
        y = c[:cutoff]
        x_seg = x[:cutoff]
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            continue
        coeff = np.polyfit(x_seg[mask], y[mask], 1)
        E_list.append(coeff[0])

    return np.array(E_list)


def compute_toughness(x, curves):
    """Area under load-strain curve."""
    A = []
    from numpy import trapezoid
    for c in curves:
        mask = ~np.isnan(c)
        if mask.sum() < 3:
            continue
        A.append(trapezoid(c[mask], x[mask]))
    return np.array(A)


def compute_peak_slopes(x, curves):
    """Computes left and right slopes near peak using log-log and semi-log fits."""
    slopes_loglog = []
    slopes_semilog = []

    for c in curves:
        if np.nanmax(c) <= 0:
            continue

        peak_idx = np.nanargmax(c)

        # Left side (1/2 to peak)
        left = c[peak_idx//2 : peak_idx]
        left_x = x[peak_idx//2 : peak_idx]

        mask = (left>0) & (left_x>0)
        if mask.sum() < 3:
            continue

        # log-log slope
        s1 = np.polyfit(np.log(left_x[mask]), np.log(left[mask]), 1)[0]
        slopes_loglog.append(s1)

        # semi-log slope
        s2 = np.polyfit(left_x[mask], np.log(left[mask]), 1)[0]
        slopes_semilog.append(s2)

    return np.array(slopes_loglog), np.array(slopes_semilog)


# ------------------------- MAIN -------------------------

print("\n=== LOADING DATA ===")
x40, c40 = load_folder(folder_40)
x80, c80 = load_folder(folder_80)

print("\n=== PLOTTING ===")
plot_combined(x40, c40, c80)
plot_average(x40, c40, c80)
plot_std_all(x40, c40, c80)

print("\n=== MECHANICAL QUANTITIES ===")

E40 = compute_young_modulus(x40, c40)
E80 = compute_young_modulus(x40, c80)
print("Young's modulus 40-wire:", E40)
print("Young's modulus 80-wire:", E80)

A40 = compute_toughness(x40, c40)
A80 = compute_toughness(x40, c80)
print("Toughness 40-wire:", A40)
print("Toughness 80-wire:", A80)

LL40, SL40 = compute_peak_slopes(x40, c40)
LL80, SL80 = compute_peak_slopes(x80, c80)
print("Peak slopes 40-wire (log-log):", LL40)
print("Peak slopes 80-wire (log-log):", LL80)

print("\nDONE.")
