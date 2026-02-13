import numpy as np
from collections import defaultdict



def generate_hydration_shell(positions, cell_size=3.0, padding=12.0, min_dist=3.0, shell_width=3.0):
        """
        Generate a grid approximation of the hydration shell using the linked-cell approach.

        Parameters
        ----------
        positions : np.ndarray, shape (N,3)
            Atom positions.
        cell_size : float
            Grid spacing (Å).
        padding : float
            Extra padding beyond molecule bounding box (Å).
        min_dist : float
            Minimum distance from any atom (Å).
        shell_width : float
            Width of hydration shell (Å).

        Returns
        -------
        shell_cells : np.ndarray, shape (M,3)
            Centers of the cells approximating the hydration shell.
        """
        # 1. Bounding box with padding
        min_corner = np.min(positions, axis=0) - padding
        max_corner = np.max(positions, axis=0) + padding

        # 2. Create 3D grid of cell centers
        x = np.arange(np.floor(min_corner[0]), np.ceil(max_corner[0]) + cell_size, cell_size)
        y = np.arange(np.floor(min_corner[1]), np.ceil(max_corner[1]) + cell_size, cell_size)
        z = np.arange(np.floor(min_corner[2]), np.ceil(max_corner[2]) + cell_size, cell_size)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        cell_centers = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        # 3. Assign atoms to cells
        cell_indices = np.floor((positions - min_corner) / cell_size).astype(int)
        cells = defaultdict(list)
        for idx, (i, j, k) in enumerate(cell_indices):
            cells[(i, j, k)].append(idx)

        n_cells = (np.ceil((max_corner - min_corner) / cell_size)).astype(int)

        # Helper: get 26 neighbors + self
        def neighbor_cells(i, j, k):
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    for dk in [-1, 0, 1]:
                        ni, nj, nk = i + di, j + dj, k + dk
                        # ignore out-of-bounds
                        if 0 <= ni < n_cells[0] and 0 <= nj < n_cells[1] and 0 <= nk < n_cells[2]:
                            neighbors.append((ni, nj, nk))
            return neighbors

        # 4. Filter cells
        shell_cells = []
        for idx, center in enumerate(cell_centers):
            i = int(np.floor((center[0] - min_corner[0]) / cell_size))
            j = int(np.floor((center[1] - min_corner[1]) / cell_size))
            k = int(np.floor((center[2] - min_corner[2]) / cell_size))

            # Collect atoms in this cell + neighbors
            neighbor_atoms = []
            for ci, cj, ck in neighbor_cells(i, j, k):
                neighbor_atoms.extend(cells.get((ci, cj, ck), []))
            if not neighbor_atoms:
                continue  # skip empty regions

            # Compute distances
            dists = np.linalg.norm(positions[neighbor_atoms] - center, axis=1)
            # Apply criteria
            if np.all(dists > min_dist) and np.any(dists < (min_dist + shell_width)):
                shell_cells.append(center)

        return np.array(shell_cells)