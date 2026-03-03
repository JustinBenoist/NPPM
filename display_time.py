import os
import pickle
import argparse
import matplotlib.pyplot as plt
import itertools
import re
import numpy as np

def load_errors(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, (list, tuple)):
            return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return None

def normalize_name(name):
    return re.sub(r'_(small2|small)$', '', name.lower())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='output', help='Output directory containing test results')
    parser.add_argument('--metric', type=str, default='L1', choices=['L1', 'L2', 'FLIP', 'SMAPE', 'RELMSE'], help='Error metric to display')
    parser.add_argument('--no_legend', action="store_true", default=False, help="indicates if we use display the legend")
    return parser.parse_args()

def exponential_moving_average(data, alpha=0.3):
    ema = []
    for i, v in enumerate(data):
        if i == 0:
            ema.append(v)
        else:
            ema.append(alpha * v + (1 - alpha) * ema[-1])
    return np.array(ema)

def main():
    args = parse_args()
    base_dir = args.output
    metric_names = {'L1': 0, 'L2': 1, 'FLIP': 2, 'SMAPE': 3, 'RELMSE': 4}
    if args.metric not in metric_names:
        raise ValueError(f"Unknown metric: {args.metric}")

    error_idx = metric_names[args.metric]
    sizes = ["normal"]

    # Collect test folders and normalize technique names
    all_tests = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith("tmp")]
    print(all_tests)
    all_tests = ["pretrain", "pretrain_beta", "CPPM", "SPPM"]
    technique_names = sorted({normalize_name(d) for d in all_tests})
    technique_names = ["pretrain", "pretrain_beta", "cppm", "sppm"]
    color_map = {name: color for name, color in zip(technique_names, itertools.cycle(plt.cm.tab10.colors))}

    # Get scenes from the first test
    first_test_path = os.path.join(base_dir, all_tests[0])
    scenes = [d for d in os.listdir(first_test_path) if os.path.isdir(os.path.join(first_test_path, d))]

    for scene in scenes:
        fig, axs = plt.subplots(1, len(sizes), figsize=(7, 5), sharey=True)
        if not args.no_legend:
            fig.suptitle(f"Iteration time (seconds) - {scene}", fontsize=16)

        for i, size in enumerate(sizes):
            ax = axs[i] if len(sizes) > 1 else axs
            # ax.set_xlabel("Iterations")
            # if i == 0:
                # ax.set_ylabel(f"Error {args.metric}")

            legend = {}
            max_length = 0

            for test in all_tests:
                test_lower = test.lower()
                if size == "normal" and ("_small" in test_lower or "_small2" in test_lower):
                    continue
                if size != "normal" and not test_lower.endswith(f"_{size}"):
                    continue

                base_name = normalize_name(test)
                color = color_map[base_name]
                test_path = os.path.join(base_dir, test, scene)
                if not os.path.exists(test_path):
                    continue

                for file in os.listdir(test_path):
                    if not file.endswith(".pkl") or file.startswith(("DCV", "stats")):
                        continue

                    file_path = os.path.join(test_path, file)
                    errors = load_errors(file_path)
                    if not errors:
                        continue

                    try:
                        y_vals = np.diff([x[0] for x in errors], prepend=0)[5:]
                        y_vals = exponential_moving_average(y_vals, alpha=0.1)
                        label = f"{test}"
                        line, = ax.plot(range(5, len(y_vals)+5), y_vals, label=label, color=color)
                        max_length = max(max_length, len(y_vals))
                        # ax.set_xlim(left=5)
                        legend[label] = line
                    except Exception as e:
                        print(f"Failed to parse values in {file_path}: {e}")

            # Sort legend by base technique name
            sorted_labels = sorted(legend, key=lambda l: technique_names.index(normalize_name(l)))
            if not args.no_legend:
                ax.legend([legend[l] for l in sorted_labels], sorted_labels, fontsize=8)
            from matplotlib import ticker
            tick_spacing = 500
            ticks = list(range(0, max_length + 0, tick_spacing))

            # Make sure 5 is in the list
            if 5 not in ticks:
                ticks = [5] + ticks[1:]

            # Set them
            ax.set_xticks(ticks)
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
            # ax.yaxis.set_minor_locator(plt.NullLocator())
            # ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
            # ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
            ax.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = os.path.join(base_dir, f"iteration_time_{scene}.svg")
        plt.savefig(output_path, dpi=720)
        plt.show()
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()