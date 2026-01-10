import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.neighbors import KDTree
from scipy.spatial import Delaunay


class AdvancedSpatialDataset(Dataset):
    """
    Advanced Spatial Dataset: Supports multi-scale and direction-aware spatial aggregation.
    """

    def __init__(self, sample, img_feats, gene_exprs, coords, labels,
                 scales=[3], directions=8):
        """
        Args:
            sample: Sample identifier.
            img_feats: Image features [N, D_img].
            gene_exprs: Gene expression features [N, D_gene].
            coords: Spatial coordinates [N, 2].
            labels: Labels [N].
            scales: List of neighbor counts for different scales.
            directions: Number of directional bins (e.g., 8 for 45-degree intervals).
        """
        self.original_img = img_feats
        self.original_gene = gene_exprs
        self.coords = coords
        self.labels = labels
        self.sample = sample
        self.scales = scales
        self.directions = directions

        self.orig_img_dim = img_feats.shape[1]
        self.orig_gene_dim = gene_exprs.shape[1]

        # Precompute spatial aggregates
        self.agg_img, self.agg_gene = self.calculate_spatial_aggregates()

    def calculate_spatial_aggregates(self):
        """Computes multi-scale, direction-aware spatial aggregates."""
        n_spots = len(self.coords)
        agg_img = {}
        agg_gene = {}

        # Handle edge case with insufficient spots
        if n_spots <= 1:
            for scale in self.scales:
                agg_img[scale] = self.original_img.copy()
                agg_gene[scale] = self.original_gene.copy()
            return agg_img, agg_gene

        tri = Delaunay(self.coords)
        simplices = tri.simplices

        # Build adjacency list
        adjacency = {i: set() for i in range(n_spots)}
        for simplex in simplices:
            for i in simplex:
                for j in simplex:
                    if i != j:
                        adjacency[i].add(j)

        # Aggregate for each scale
        for scale in self.scales:
            agg_img[scale] = np.zeros((n_spots, self.orig_img_dim + self.directions))
            agg_gene[scale] = np.zeros((n_spots, self.orig_gene_dim + self.directions))

            for i in range(n_spots):
                neighbors = list(adjacency[i])

                # Fill with KNN if insufficient Delaunay neighbors
                if len(neighbors) < scale:
                    tree = KDTree(self.coords)
                    _, knn_indices = tree.query([self.coords[i]], k=scale + 1)
                    neighbors = list(set(neighbors) | set(knn_indices[0][1:]))
                    neighbors = neighbors[:scale]

                if len(neighbors) > 0:
                    agg_img[scale][i], agg_gene[scale][i] = self.directional_aggregation(i, neighbors)
                else:
                    # Fallback: self-feature + zero direction weights
                    agg_img[scale][i] = np.concatenate([self.original_img[i], np.zeros(self.directions)])
                    agg_gene[scale][i] = np.concatenate([self.original_gene[i], np.zeros(self.directions)])
        return agg_img, agg_gene

    def directional_aggregation(self, center_idx, neighbor_indices):
        """Performs direction-aware spatial aggregation."""
        center = self.coords[center_idx]
        directions = [[] for _ in range(self.directions)]

        # Assign neighbors to directional bins
        for idx in neighbor_indices:
            vec = self.coords[idx] - center
            angle = np.arctan2(vec[1], vec[0])
            if angle < 0:
                angle += 2 * np.pi

            dir_index = int(angle / (2 * np.pi / self.directions)) % self.directions
            directions[dir_index].append(idx)

        agg_img = np.zeros(self.orig_img_dim)
        agg_gene = np.zeros(self.orig_gene_dim)
        dir_weights = np.zeros(self.directions)

        total_neighbors = len(neighbor_indices)
        for dir_idx, dir_group in enumerate(directions):
            if len(dir_group) == 0:
                continue

            # Average features within the directional bin
            dir_img = np.mean(self.original_img[dir_group], axis=0)
            dir_gene = np.mean(self.original_gene[dir_group], axis=0)

            # Weight by density (number of neighbors in this direction)
            weight = len(dir_group) / total_neighbors

            agg_img += dir_img * weight
            agg_gene += dir_gene * weight
            dir_weights[dir_idx] = weight

        # Concatenate features and direction weights
        agg_img = np.concatenate([agg_img, dir_weights])
        agg_gene = np.concatenate([agg_gene, dir_weights])

        return agg_img, agg_gene

    def __len__(self):
        return len(self.original_img)

    def __getitem__(self, idx):
        item = {
            'orig_img': torch.tensor(self.original_img[idx], dtype=torch.float32),
            'orig_gene': torch.tensor(self.original_gene[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

        for scale in self.scales:
            item[f'agg_img_{scale}'] = torch.tensor(self.agg_img[scale][idx], dtype=torch.float32)
            item[f'agg_gene_{scale}'] = torch.tensor(self.agg_gene[scale][idx], dtype=torch.float32)

        return item


def spatial_collate_fn(batch):
    """Custom collate function for handling multi-scale spatial data."""
    collated = {
        'orig_img': [],
        'orig_gene': [],
        'label': []
    }

    # Identify active scales
    scales = [key.split('_')[-1] for key in batch[0].keys() if 'agg_img_' in key]

    for scale in scales:
        collated[f'agg_img_{scale}'] = []
        collated[f'agg_gene_{scale}'] = []

    for item in batch:
        collated['orig_img'].append(item['orig_img'])
        collated['orig_gene'].append(item['orig_gene'])
        collated['label'].append(item['label'].unsqueeze(0))

        for scale in scales:
            collated[f'agg_img_{scale}'].append(item[f'agg_img_{scale}'])
            collated[f'agg_gene_{scale}'].append(item[f'agg_gene_{scale}'])

    # Stack tensors
    for key in collated:
        collated[key] = torch.stack(collated[key])

        if key == 'label':
            collated[key] = collated[key].squeeze(-1)  # Ensure shape [B]

    return collated