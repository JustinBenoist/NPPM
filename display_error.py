import os
import pickle
import argparse
import matplotlib.pyplot as plt
import itertools
import re

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
    technique_names = sorted({normalize_name(d) for d in all_tests})
    color_map = {name: color for name, color in zip(technique_names, itertools.cycle(plt.cm.tab10.colors))}

    # Get scenes from the first test
    first_test_path = os.path.join(base_dir, all_tests[0])
    scenes = [d for d in os.listdir(first_test_path) if os.path.isdir(os.path.join(first_test_path, d))]

    for scene in scenes:
        fig, axs = plt.subplots(1, len(sizes), figsize=(7, 5), sharey=True)
        if not args.no_legend:
            fig.suptitle(f"Convergence - {scene} - Error {args.metric}", fontsize=16)

        for i, size in enumerate(sizes):
            ax = axs[i] if len(sizes) > 1 else axs
            # ax.set_xlabel("Iterations")
            # if i == 0:
                # ax.set_ylabel(f"Error {args.metric}")

            legend = {}

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
                        y_vals = [x[error_idx].item() for x in errors]
                        label = f"{test}"
                        line, = ax.loglog(range(1, len(y_vals)+1), y_vals, label=label, color=color)
                        legend[label] = line
                    except Exception as e:
                        print(f"Failed to parse values in {file_path}: {e}")

            # Sort legend by base technique name
            sorted_labels = sorted(legend, key=lambda l: technique_names.index(normalize_name(l)))
            if not args.no_legend:
                ax.legend([legend[l] for l in sorted_labels], sorted_labels, fontsize=8)
            from matplotlib.ticker import LogLocator
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.yaxis.set_minor_locator(plt.NullLocator())
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
            ax.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = os.path.join(base_dir, f"convergence_{scene}_{args.metric}.svg")
        plt.savefig(output_path, dpi=720)
        plt.show()
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()