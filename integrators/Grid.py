import torch
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

class Grid3D:
    def __init__(self, scene: mi.Scene, device: str, resolution: int = 256, n_features: int=32) -> None:
        self.res = resolution
        self.n_features = n_features
        self.device = device
        self.data = torch.zeros((self.res ** 3, n_features), device=device)
        self.counts = torch.zeros((self.res ** 3, 1), device=device)
        self.max_bounds = scene.bbox().max.torch().to(device) * 1.1
        self.min_bounds = scene.bbox().min.torch().to(device) * 1.1
        self.grid_spacing = (self.max_bounds - self.min_bounds) / (self.res - 1)
    
    def get_cell_count(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Given a batched tensor of 3D positions, return interpolated values from the grid using trilinear or nearest interpolation.

        Args:
            positions (torch.Tensor): A tensor of shape (B, 3) representing B 3D positions.

        Returns:
            torch.Tensor: A tensor of shape (B, n_features) containing the interpolated feature vectors.
        """
        # Normalize positions into grid coordinates
        grid_coords = (positions - self.min_bounds) / self.grid_spacing  # Shape: (B, 3)

        # Round to the nearest integer to find the closest grid index
        nearest_idx = torch.round(grid_coords).long().clamp(0, self.res - 1)  # Shape: (B, 3)

        # Compute the 1D indices for the flattened array
        flat_index = (nearest_idx[:, 2] * self.res ** 2) + (nearest_idx[:, 1] * self.res) + nearest_idx[:, 0]  # Shape: (B,)

        return self.counts[flat_index]
    
    def get_cell_value(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Given a batched tensor of 3D positions, return interpolated values from the grid using trilinear or nearest interpolation.

        Args:
            positions (torch.Tensor): A tensor of shape (B, 3) representing B 3D positions.

        Returns:
            torch.Tensor: A tensor of shape (B, n_features) containing the interpolated feature vectors.
        """
        return self.__get_nearest_cell_value(positions)
    
    def __get_nearest_cell_value(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Given a batched tensor of 3D positions, return the values of the nearest cells in the grid.

        Args:
            positions (torch.Tensor): A tensor of shape (B, 3) representing B 3D positions.

        Returns:
            torch.Tensor: A tensor of shape (B, n_features) containing the feature vectors for each input position.
        """
        # Normalize positions into grid coordinates
        grid_coords = (positions - self.min_bounds +  (torch.rand_like(positions) - .5) * self.grid_spacing) / self.grid_spacing  # Shape: (B, 3)

        # Round to the nearest integer to find the closest grid index
        nearest_idx = torch.round(grid_coords).long().clamp(0, self.res - 1)  # Shape: (B, 3)

        # Compute the 1D indices for the flattened array
        flat_index = (nearest_idx[:, 2] * self.res ** 2) + (nearest_idx[:, 1] * self.res) + nearest_idx[:, 0]  # Shape: (B,)

        return self.data[flat_index]
    
    def __get_interpolated_cell_value(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Given a batched tensor of 3D positions, return interpolated values from the grid using trilinear interpolation.

        Args:
            positions (torch.Tensor): A tensor of shape (B, 3) representing B 3D positions.

        Returns:
            torch.Tensor: A tensor of shape (B, n_features) containing the interpolated feature vectors.
        """
        B = positions.shape[0]

        # Normalize to grid coordinates
        grid_coords = (positions - self.min_bounds) / self.grid_spacing  # Shape: (B, 3)

        # Get integer part and fractional part
        idx0 = torch.floor(grid_coords).long().clamp(0, self.res - 2)  # Ensure we don't go out of bounds
        d = grid_coords - idx0.float()  # fractional part for interpolation

        # Compute 8 corner indices
        def compute_flat_idx(ix, iy, iz):
            return (iz * self.res ** 2) + (iy * self.res) + ix

        ix, iy, iz = idx0[:, 0], idx0[:, 1], idx0[:, 2]

        # Get values at all 8 corners
        c000 = self.data[compute_flat_idx(ix,     iy,     iz    )]
        c001 = self.data[compute_flat_idx(ix,     iy,     iz + 1)]
        c010 = self.data[compute_flat_idx(ix,     iy + 1, iz    )]
        c011 = self.data[compute_flat_idx(ix,     iy + 1, iz + 1)]
        c100 = self.data[compute_flat_idx(ix + 1, iy,     iz    )]
        c101 = self.data[compute_flat_idx(ix + 1, iy,     iz + 1)]
        c110 = self.data[compute_flat_idx(ix + 1, iy + 1, iz    )]
        c111 = self.data[compute_flat_idx(ix + 1, iy + 1, iz + 1)]

        # Interpolate along x
        c00 = c000 * (1 - d[:, 0:1]) + c100 * d[:, 0:1]
        c01 = c001 * (1 - d[:, 0:1]) + c101 * d[:, 0:1]
        c10 = c010 * (1 - d[:, 0:1]) + c110 * d[:, 0:1]
        c11 = c011 * (1 - d[:, 0:1]) + c111 * d[:, 0:1]

        # Interpolate along y
        c0 = c00 * (1 - d[:, 1:2]) + c10 * d[:, 1:2]
        c1 = c01 * (1 - d[:, 1:2]) + c11 * d[:, 1:2]

        # Interpolate along z
        c = c0 * (1 - d[:, 2:3]) + c1 * d[:, 2:3]

        return c
    
    def update_nearest_cell(self, positions: torch.Tensor, values: torch.Tensor):
            """
            Updates the floating mean of the nearest grid cells using batched data points.

            Args:
                positions (torch.Tensor): A tensor of shape (B, 3) containing B 3D positions.
                values (torch.Tensor): A tensor of shape (B, n_features) containing B new feature values.
            """
            # Normalize positions into grid coordinates
            grid_coords = (positions - self.min_bounds + (torch.rand_like(positions) - .5) * self.grid_spacing ) / self.grid_spacing  # Shape: (B, 3)

            # Round to the nearest integer to find the closest grid index
            nearest_idx = torch.round(grid_coords).long().clamp(0, self.res - 1)  # Shape: (B, 3)

            # Compute 1D indices for the flattened array
            flat_index = (nearest_idx[:, 2] * self.res ** 2) + (nearest_idx[:, 1] * self.res) + nearest_idx[:, 0]  # Shape: (B,)

            # Update running mean using incremental averaging
            self.counts[flat_index] += 1  # Increment count for each cell
            alpha = 1 / self.counts[flat_index]  # Compute weight for new value
            self.data[flat_index] = (1 - alpha) * self.data[flat_index] + alpha * values  # Running mean formula
    
    def print_data_size(self) -> None:
        """
        Prints the size of self.data in megabytes (MB).
        """
        num_elements = self.data.numel()  # Total number of elements
        bytes_per_element = self.data.element_size()  # Size of each element in bytes
        total_size_mb = (num_elements * bytes_per_element) / (1024 * 1024)  # Convert to MB
        print(f"Size of self.data: {total_size_mb:.2f} MB")
    
    def __getstate__(self):
        """Prepare object state for pickling."""
        state = self.__dict__.copy()
        # Move tensors to CPU before saving (ensures portability)
        state["data"] = self.data.cpu()
        state["counts"] = self.counts.cpu()
        return state

    def __setstate__(self, state):
        """Restore object state after unpickling."""
        self.__dict__.update(state)
        # Move tensors back to original device
        self.device = state["device"]
        self.data = self.data.to(self.device)
        self.counts = self.counts.to(self.device)
