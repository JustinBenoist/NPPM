#!/usr/bin/env python3
"""Profile sub-steps of Histogram2D.sample().

Creates micro-bench timings for the main operations inside `sample()` so
you can identify hotspots (flatten/sum, multinomial, indexing, random
generation, coordinate computation, pdf computation).

Usage:
  python3 tools/profile_histogram_sample.py --bins 512 --N 4096 --repeats 50 --measure_probs

If running on GPU, timings use `torch.cuda.synchronize()` for accuracy.
"""
import time
import argparse
import torch

from integrators.Guiding import Histogram2D


def benchmark(hist: Histogram2D, N: int = 4096, repeats: int = 50, measure_probs: bool = True):
    device = hist.device
    bins = hist.bins
    results = {}

    # ensure histogram is non-empty
    if hist.hist_sum.sum() == 0:
        hist.hist_sum += torch.rand((bins, bins), device=device)
        hist._probs_flat = None

    # Measure flatten()/sum()/normalize (uncached path)
    if measure_probs:
        hist._probs_flat = None
        times = []
        probs_norm = None
        for _ in range(repeats):
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            probs = hist.hist_sum.flatten()
            psum = probs.sum()
            probs_norm = probs / psum
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        results['probs_compute_ms'] = (sum(times) / len(times)) * 1000.0
        # cache for subsequent ops (mimics sample() behavior)
        hist._probs_flat = probs_norm

    # multinomial sampling from flattened probabilities
    probs = hist._probs_flat
    times = []
    for _ in range(repeats):
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        flat_idx = torch.multinomial(probs, N, replacement=True)
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['multinomial_ms'] = (sum(times) / len(times)) * 1000.0

    # index conversion (flat -> x_idx, y_idx)
    times = []
    for _ in range(repeats):
        flat_idx = torch.randint(0, bins * bins, (N,), device=device)
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        x_idx = flat_idx // bins
        y_idx = flat_idx % bins
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['index_conv_ms'] = (sum(times) / len(times)) * 1000.0

    # random coordinates inside bins (u,v)
    times = []
    for _ in range(repeats):
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        u = torch.rand(N, device=device)
        v = torch.rand(N, device=device)
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['rand_ms'] = (sum(times) / len(times)) * 1000.0

    # compute x/y coordinates inside chosen bins
    flat_idx = torch.randint(0, bins * bins, (N,), device=device)
    x_idx = flat_idx // bins
    y_idx = flat_idx % bins
    edges_x = hist.edges_x
    edges_y = hist.edges_y
    times = []
    for _ in range(repeats):
        u = torch.rand(N, device=device)
        v = torch.rand(N, device=device)
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ex = edges_x[x_idx]
        ex1 = edges_x[x_idx + 1]
        x = ex + u * (ex1 - ex)
        ey = edges_y[y_idx]
        ey1 = edges_y[y_idx + 1]
        y = ey + v * (ey1 - ey)
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['xy_ms'] = (sum(times) / len(times)) * 1000.0

    # compute pdf from cached flattened probs
    probs = hist._probs_flat
    flat_idx = torch.randint(0, bins * bins, (N,), device=device)
    times = []
    for _ in range(repeats):
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pdf = probs[flat_idx] / hist._bin_area
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['pdf_ms'] = (sum(times) / len(times)) * 1000.0

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', type=int, default=512)
    parser.add_argument('--N', type=int, default=4096)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--measure_probs', action='store_true', help='time flatten/sum/normalize (uncached)')
    args = parser.parse_args()

    hist = Histogram2D(bins=args.bins, device=args.device)
    # make sure histogram contains data
    hist.hist_sum += torch.rand(args.bins, args.bins, device=args.device)
    hist._probs_flat = None

    results = benchmark(hist, N=args.N, repeats=args.repeats, measure_probs=args.measure_probs)

    print('\nHistogram2D.sample() micro-benchmark (mean per-call) [ms]')
    for k, v in results.items():
        print(f'{k}: {v:.3f} ms')


if __name__ == '__main__':
    main()
