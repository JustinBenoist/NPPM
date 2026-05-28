import torch
import math
import time
import numpy as np

class Histogram2D:
    def __init__(self, bins=512, device="cuda"):
        """
        x: (N, 2) tensor
        bins: number of bins per dimension (n x n)

        returns:
            hist: (bins, bins) tensor
        """
        self.bins = bins
        self.device = device
        
        self.x_min = 0
        self.x_max = 1.0
        self.y_min = 0
        self.y_max = 1.0
        
        # Create bin edges
        self.edges_x = torch.linspace(self.x_min, self.x_max, bins + 1, device=self.device)
        self.edges_y = torch.linspace(self.y_min, self.y_max, bins + 1, device=self.device)
        
        self.hist_sum = torch.zeros((bins, bins), device=self.device)
        self.total_samples = 0
        # cached flattened normalized probabilities for fast sampling
        self._probs_flat = None
        # precompute bin area for pdf calculation
        self._bin_area = (self.edges_x[1] - self.edges_x[0]) * (self.edges_y[1] - self.edges_y[0])
        
    def add(self, x: torch.Tensor):
        """
        x: (N, 2)
        """
        bins = self.bins
        
        x0 = x[0, :].contiguous()
        x1 = x[1, :].contiguous()
        x_idx = torch.bucketize(x0, self.edges_x) - 1
        y_idx = torch.bucketize(x1, self.edges_y) - 1

        valid = (
            (x_idx >= 0) & (x_idx < bins) &
            (y_idx >= 0) & (y_idx < bins)
        )

        x_idx = x_idx[valid]
        y_idx = y_idx[valid]

        flat_idx = x_idx * bins + y_idx

        hist = torch.zeros(bins * bins, device=x.device, dtype=torch.float32)
        # hist = torch.ones(bins * bins, device=x.device, dtype=torch.float32)
        hist.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        hist = hist.view(bins, bins)
        
        self.hist_sum += hist
        self.total_samples += valid.sum().item()
        # self.total_samples += x.sum().item()
        # Invalidate cached sampling structures
        self._probs_flat = None
        self._alias_idx = None
        self._alias_prob = None
        
        # Precompute flattened-index -> (x_idx, y_idx) lookups and per-bin left edges/widths
        M = bins * bins
        idxs = torch.arange(M, device=self.device, dtype=torch.long)
        self._x_lookup = (idxs // bins).to(torch.long)
        self._y_lookup = (idxs % bins).to(torch.long)
        # left edge and width for each bin (length `bins`)
        self._edges_x_left = self.edges_x[:-1].contiguous()
        self._edges_x_width = (self.edges_x[1:] - self.edges_x[:-1]).contiguous()
        self._edges_y_left = self.edges_y[:-1].contiguous()
        self._edges_y_width = (self.edges_y[1:] - self.edges_y[:-1]).contiguous()

    def get_hist(self, normalize=False):
        if normalize:
            return self.hist_sum / max(self.total_samples, 1)
        return self.hist_sum
    
    def sample(self, N):
        """
        Returns:
            samples: (N, 2)
            pdf: (N,)
        """
        # If an alias table is available, use alias sampling (O(1) per sample)
        if getattr(self, '_alias_prob', None) is not None and getattr(self, '_alias_idx', None) is not None:
            return self.sample_alias(N)

        # Use cached flattened normalized probabilities when available to avoid
        # repeated flatten()/sum()/division work on the GPU.
        if self._probs_flat is None:
            probs = self.hist_sum.flatten()
            probs_sum = probs.sum()
            if probs_sum <= 0:
                raise RuntimeError("Histogram is empty")
            self._probs_flat = probs / probs_sum

        probs = self._probs_flat

        # Sample bins (fallback multinomial)
        flat_idx = torch.multinomial(probs, N, replacement=True)

        # Convert to 2D indices
        bins = self.bins
        x_idx = flat_idx // bins
        y_idx = flat_idx % bins

        # Sample uniformly inside each bin
        u = torch.rand(N, device=self.device)
        v = torch.rand(N, device=self.device)

        x = self.edges_x[x_idx] + u * (self.edges_x[x_idx + 1] - self.edges_x[x_idx])
        y = self.edges_y[y_idx] + v * (self.edges_y[y_idx + 1] - self.edges_y[y_idx])

        samples = torch.stack([x, y], dim=-1)

        # Compute PDF = p(bin) / area(bin)
        pdf = (probs[flat_idx] / self._bin_area)

        return samples, pdf

    def build_alias(self):
        """Build alias table (Vose's algorithm) and store on `self.device`.

        This is a one-time (per histogram update) O(M) operation where
        M = bins*bins. Sampling with the alias table is O(1) per sample.
        """
        probs = self.hist_sum.flatten()
        probs_sum = probs.sum()
        if probs_sum <= 0:
            raise RuntimeError("Histogram is empty")

        # Normalize and move to CPU numpy for the alias build
        probs_cpu = (probs / probs_sum).cpu().numpy().astype(np.float64)
        M = probs_cpu.size

        scaled = probs_cpu * float(M)
        small = []
        large = []
        for i, val in enumerate(scaled):
            if val < 1.0:
                small.append(i)
            else:
                large.append(i)

        alias = np.empty(M, dtype=np.int64)
        prob = np.empty(M, dtype=np.float32)

        while small and large:
            l = small.pop()
            g = large.pop()
            prob[l] = float(scaled[l])
            alias[l] = int(g)
            scaled[g] = scaled[g] - (1.0 - scaled[l])
            if scaled[g] < 1.0:
                small.append(g)
            else:
                large.append(g)

        for g in large + small:
            prob[g] = 1.0
            alias[g] = int(g)

        # Store alias arrays on the correct device
        self._alias_prob = torch.from_numpy(prob).to(self.device)
        self._alias_idx = torch.from_numpy(alias).to(self.device)
        # Also store normalized probabilities for pdf computation
        self._probs_flat = torch.from_numpy(probs_cpu.astype(np.float32)).to(self.device)

    def sample_alias(self, N):
        """Sample using the prebuilt alias table. Returns (samples, pdf).

        Builds the alias table if it does not exist.
        """
        if getattr(self, '_alias_prob', None) is None or getattr(self, '_alias_idx', None) is None:
            self.build_alias()

        M = self._alias_prob.shape[0]
        device = self.device

        # Draw uniform base indices and uniform tests
        idx = torch.randint(0, M, (N,), device=device)
        u = torch.rand((N,), device=device)
        prob_idx = self._alias_prob[idx]
        choose = u < prob_idx
        flat_idx = torch.where(choose, idx, self._alias_idx[idx])

        x_idx = self._x_lookup[flat_idx]
        y_idx = self._y_lookup[flat_idx]
        
        u2 = torch.rand(N, device=device)
        v2 = torch.rand(N, device=device)
        x = self._edges_x_left[x_idx] + u2 * self._edges_x_width[x_idx]
        y = self._edges_y_left[y_idx] + v2 * self._edges_y_width[y_idx]

        samples = torch.stack([x, y], dim=-1)
        pdf = (self._probs_flat[flat_idx] / self._bin_area)
        return samples, pdf
    
    def filter(self):
        # Apply Gaussian filter to the histogram
        kernel_size = 5
        sigma = 1.0
        
        # Create Gaussian kernel
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=self.device)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        
        # Pad histogram to handle borders
        hist_padded = torch.nn.functional.pad(self.hist_sum, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode='constant', value=0)
        
        # Convolve histogram with Gaussian kernel
        hist_filtered = torch.nn.functional.conv2d(hist_padded.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=0)
        
        self.hist_sum = hist_filtered.squeeze()
    
class LightTracingGuiding:
    def __init__(self, bins=512, device="cuda"):
        self.hist_dirs = Histogram2D(bins=bins, device=device)
        self.hist_pos = Histogram2D(bins=bins, device=device)
    
    def add(self, dir_samples, pos_samples):
        self.hist_dirs.add(dir_samples)
        self.hist_pos.add(pos_samples)

    def filter(self):
        self.hist_dirs.filter()
        self.hist_pos.filter()
    
    def sample(self, N):
        return *self.hist_dirs.sample(N), *self.hist_pos.sample(N)
    
    def build_alias(self):
        self.hist_dirs.build_alias()
        self.hist_pos.build_alias()