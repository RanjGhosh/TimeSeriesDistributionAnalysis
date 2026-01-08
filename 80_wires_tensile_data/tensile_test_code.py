import os
import csv
import matplotlib.pyplot as plt

# ----------- SETTINGS ------------
folder_40 = "40 wires"
folder_80 = "80 wires"
save_individual = True
# ----------------------------------

def read_tensile_data(filepath):
    load = []
    distance = []

    print(f"\n📄 Reading file: {filepath}")
    with open(filepath, 'r') as file:
        lines = file.readlines()[16:]  # Skip metadata/header
        reader = csv.DictReader(lines, delimiter='\t')

        for i, row in enumerate(reader, start=17):
            try:
                force = abs(float(row['Load [N]']))
                dist = float(row['Distance [mm]'])
                load.append(force)
                distance.append(dist)
            except (ValueError, KeyError):
                continue

    return distance, load

def plot_individual_sample(distance, load, title, color, save_path=None):
    if not distance or not load:
        print(f"⚠️ Skipped empty data in {title}")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(distance, load, color=color, label=title)
    plt.title(title)
    plt.xlabel("Distance (mm)")
    plt.ylabel("Load (N)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_all_samples(folder_path, color, label_tag):
    all_curves = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder_path, filename)
            distance, load = read_tensile_data(filepath)

            if distance and load:
                title = f"{label_tag} - {filename}"
                save_path = f"{label_tag}_{filename.replace('.csv', '')}.png" if save_individual else None
                plot_individual_sample(distance, load, title, color, save_path)
                all_curves.append((distance, load, filename))
            else:
                print(f"⚠️ No valid data in file: {filename}")
    return all_curves

def plot_combined(all_40, all_80):
    plt.figure(figsize=(10, 6))
    any_data = False

    for distance, load, filename in all_40:
        if distance and load:
            plt.plot(distance, load, color='red', alpha=0.7, label=f"40 wires - {filename}")
            any_data = True

    for distance, load, filename in all_80:
        if distance and load:
            plt.plot(distance, load, color='blue', alpha=0.7, label=f"80 wires - {filename}")
            any_data = True

    if any_data:
        plt.title("Combined Load vs Distance: 40 and 80 Wire Samples")
        plt.xlabel("Distance (mm)")
        plt.ylabel("Load (N)")
        plt.grid(True)
        plt.legend(fontsize='small', loc='upper right')
        plt.tight_layout()
        plt.savefig("tensile_combined_plot.png")
        plt.show()
    else:
        print("⚠️ No data found for combined plot.")

# ------- MAIN EXECUTION -------
all_40_curves = plot_all_samples(folder_40, 'red', '40 wires')
all_80_curves = plot_all_samples(folder_80, 'blue', '80 wires')
plot_combined(all_40_curves, all_80_curves)






'''import os
import csv
import matplotlib.pyplot as plt

# ----------- SETTINGS ------------
folder_40 = "40 wires"
folder_80 = "80 wires"
save_individual = True
# ----------------------------------

def read_tensile_data(filepath):
    load = []
    distance = []

    print(f"\n📄 Reading file: {filepath}")
    with open(filepath, 'r') as file:
        lines = file.readlines()[16:]  # Skip metadata/header
        reader = csv.DictReader(lines, delimiter='\t')

        for i, row in enumerate(reader, start=17):
            try:
                force = abs(float(row['Load [N]']))
                dist = float(row['Distance [mm]'])
                load.append(force)
                distance.append(dist)
            except (ValueError, KeyError):
                continue

    return distance, load

def plot_individual_sample(distance, load, title, color, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(distance, load, label=title, color=color)
    plt.title(title)
    plt.xlabel("Distance (mm)")
    plt.ylabel("Load (N)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_all_samples(folder_path, color, label_tag):
    all_curves = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder_path, filename)
            distance, load = read_tensile_data(filepath)

            if distance:
                title = f"{label_tag} - {filename}"
                save_path = f"{label_tag}_{filename.replace('.csv', '')}.png" if save_individual else None
                plot_individual_sample(distance, load, title, color, save_path)
                all_curves.append((distance, load, filename))
    return all_curves

def plot_combined(all_40, all_80):
    plt.figure(figsize=(10, 6))

    for distance, load, filename in all_40:
        plt.plot(distance, load, color='red', alpha=0.7, label=f"40 wires - {filename}")

    for distance, load, filename in all_80:
        plt.plot(distance, load, color='blue', alpha=0.7, label=f"80 wires - {filename}")

    plt.title("Combined Load vs Distance: 40 and 80 Wire Samples")
    plt.xlabel("Distance (mm)")
    plt.ylabel("Load (N)")
    plt.xlim(0, 100)
    plt.ylim(0, 80)
    plt.grid(True)
    plt.legend(fontsize='small', loc='upper right')
    plt.tight_layout()
    plt.savefig("tensile_combined_plot.png")
    plt.show()

# ------- MAIN EXECUTION -------
all_40_curves = plot_all_samples(folder_40, 'red', '40 wires')
all_80_curves = plot_all_samples(folder_80, 'blue', '80 wires')
plot_combined(all_40_curves, all_80_curves)

'''

'''
import os
import csv
import matplotlib.pyplot as plt

# Read CSV using DictReader and skip first 16 lines
def read_tensile_data(filepath):
    load = []
    distance = []

    print(f"\n📄 Reading file: {filepath}")
    with open(filepath, 'r') as file:
        lines = file.readlines()[16:]  # Skip top 16 lines (header + metadata)
        reader = csv.DictReader(lines, delimiter='\t')  # Assuming tab-delimited

        for i, row in enumerate(reader, start=17):
            try:
                force = abs(float(row['Load [N]']))
                dist = float(row['Distance [mm]'])
                load.append(force)
                distance.append(dist)

                if len(load) <= 3:
                    print(f"Row {i}: Load = {force}, Distance = {dist}")
            except (ValueError, KeyError):
                continue

    print(f"→ Total points read: {len(load)}")
    return distance, load

def plot_all_samples(folder_40, folder_80):
    plt.figure(figsize=(10, 6))

    for filename in sorted(os.listdir(folder_40)):
        if filename.endswith('.csv'):
            path = os.path.join(folder_40, filename)
            distance, load = read_tensile_data(path)
            if distance:
                plt.plot(distance, load, color='blue', alpha=0.7, label='40 wires' if '40 wires' not in plt.gca().get_legend_handles_labels()[1] else "")

    for filename in sorted(os.listdir(folder_80)):
        if filename.endswith('.csv'):
            path = os.path.join(folder_80, filename)
            distance, load = read_tensile_data(path)
            if distance:
                plt.plot(distance, load, color='orange', alpha=0.7, label='80 wires' if '80 wires' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title("Tensile Test: Load vs Distance")
    plt.xlabel("Distance (mm)")
    plt.ylabel("Load Magnitude (N)")
    plt.xlim(0, 100)
    plt.ylim(0, 80)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tensile_comparison_plot.png")
    plt.show()

# Folder paths
folder_40 = "40 wires"
folder_80 = "80 wires"

plot_all_samples(folder_40, folder_80)
'''