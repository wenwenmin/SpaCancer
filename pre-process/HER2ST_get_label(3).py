import os
import pandas as pd
import anndata as ad
import numpy as np
from pathlib import Path


def process_slice(slice_name):
    """Process individual slice data: align coordinates and map labels."""
    h5ad_path = f"HB/her2st/{slice_name}/{slice_name}.h5ad"
    cnts_path = f"HB/her2st/{slice_name}/ut_{slice_name}_stdata_filtered.tsv"
    spatial_coords_path = f"HB/her2st/ST-spotfiles/{slice_name}_selection.tsv"
    label_coords_path = f"HB/her2st/lbl/{slice_name}_labeled_coordinates.tsv"

    output_dir = "HB/her2st/label3"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    label_output_path = f"{output_dir}/{slice_name}-labels3.txt"

    print(f"Processing slice {slice_name}...")
    try:
        cnts_df = pd.read_csv(cnts_path, sep='\t')
    except Exception as e:
        raise ValueError(f"Failed to read expression file: {str(e)}")

    spot_ids = cnts_df.iloc[:, 0].tolist()
    if not spot_ids:
        raise ValueError(f"No spot IDs found in {cnts_path}")

    # Load and validate spatial coordinates
    try:
        spatial_df = pd.read_csv(spatial_coords_path, sep='\t')
    except Exception as e:
        raise ValueError(f"Failed to read spatial file: {str(e)}")

    spatial_required = ['x', 'y', 'pixel_x', 'pixel_y']
    for col in spatial_required:
        if col not in spatial_df.columns:
            raise ValueError(f"Spatial file missing column: {col}")

    # Handle NaNs and Infinite values
    for col in spatial_required:
        if spatial_df[col].isna().any() or np.isinf(spatial_df[col]).any():
            print(f"Warning: {col} contains NaNs or Infs, filling with 0.")
            spatial_df[col] = spatial_df[col].fillna(0)
            spatial_df[col] = spatial_df[col].replace([np.inf, -np.inf], 0)

    # Round coordinates for matching
    spatial_df['x_rounded'] = spatial_df['x'].round().astype(int)
    spatial_df['y_rounded'] = spatial_df['y'].round().astype(int)
    spatial_df['pixel_x_rounded'] = spatial_df['pixel_x'].round().astype(int)
    spatial_df['pixel_y_rounded'] = spatial_df['pixel_y'].round().astype(int)

    # Load and validate labels
    try:
        label_df = pd.read_csv(label_coords_path, sep='\t')
    except Exception as e:
        raise ValueError(f"Failed to read label file: {str(e)}")

    label_required = ['x', 'y', 'label']
    for col in label_required:
        if col not in label_df.columns:
            raise ValueError(f"Label file missing column: {col}")

    label_df['x_rounded'] = label_df['x'].round().astype(int)
    label_df['y_rounded'] = label_df['y'].round().astype(int)

    adata = ad.read_h5ad(h5ad_path)

    new_spatial = []
    labels = []

    # Match spots to coordinates and labels
    for idx, spot_id in enumerate(spot_ids):
        if 'x' not in str(spot_id):
            raise ValueError(f"Invalid spot ID format: {spot_id}")

        parts = str(spot_id).split('x')
        if len(parts) != 2:
            raise ValueError(f"Invalid spot ID format: {spot_id}")

        try:
            a = int(parts[0])
            b = int(parts[1])
        except ValueError:
            raise ValueError(f"Cannot parse spot ID: {spot_id}")

        # 1. Retrieve pixel coordinates
        spatial_mask = (spatial_df['x_rounded'] == a) & (spatial_df['y_rounded'] == b)
        if not spatial_mask.any():
            raise ValueError(f"Coordinates not found for spot ID {spot_id} (a={a}, b={b})")
        spatial_info = spatial_df[spatial_mask].iloc[0]
        new_spatial.append([spatial_info['pixel_x_rounded'], spatial_info['pixel_y_rounded']])

        # 2. Retrieve and map label
        label_mask = (label_df['x_rounded'] == a) & (label_df['y_rounded'] == b)
        if not label_mask.any():
            raise ValueError(f"Label not found for spot ID {spot_id} (a={a}, b={b})")
        label_info = label_df[label_mask].iloc[0]
        label = str(label_info['label']).lower()

        # Map string labels to integers
        if "invasive cancer" in label or "cancer in situ" in label:
            labels.append(2)  # Cancer
        elif "connective tissue" in label or "adipose tissue" in label or "breast glands" in label:
            labels.append(0)  # Normal
        elif "immune infiltrate" in label:
            labels.append(1)  # Immune
        elif "undetermined" in label:
            labels.append(-1) # Undetermined
        else:
            raise ValueError(f"Unrecognized label in {slice_name}: {label_info['label']}")

    # Update adata and save
    adata.obsm['spatial'] = np.array(new_spatial)
    adata.write_h5ad(h5ad_path)

    np.savetxt(label_output_path, np.array(labels), fmt='%d')
    print(f"Finished {slice_name}, processed {len(spot_ids)} spots.")


if __name__ == "__main__":
    slices = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']
    for slice_name in slices:
        try:
            process_slice(slice_name)
        except Exception as e:
            print(f"Error processing {slice_name}: {str(e)}")
    print("All slices processed.")