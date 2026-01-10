import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
import os
import numpy as np
import random
from datetime import datetime
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
import warnings

# Imports aligned with main_STHBC.py structure
from model import SpatialCancerClassifier as CancerPredictionModel
from Dataset import AdvancedSpatialDataset, spatial_collate_fn
from CAPT import (
    find_best_val_threshold,
    fit_distribution,
    get_test_threshold,
    normalize_probs
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_all_seeds(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_training_slices(npz_path="", label_dir="", scales=[3, 6, 12], directions=4):
    """
    Load CRC training slices.
    """
    data = np.load(npz_path, allow_pickle=True)
    all_samples = ['SN048_1', 'SN048_2', 'SN048_3', 'SN048_4', 'SN84_1', 'SN84_2',
                   'SN123_1', 'SN123_2', 'SN123_4', 'SN124_1', 'SN124_2', 'SN124_4']

    slices = []

    for i, sample in enumerate(all_samples):
        img_feats = data['img_feats'][i].astype(np.float64)
        gene_exprs = data['gene_feats'][i].astype(np.float64)
        coords = data['positions'][i].astype(np.float64)

        labels_path = os.path.join(label_dir, f"{sample}-labels3.txt")
        labels = np.loadtxt(labels_path, dtype=int)

        # Filter valid indices
        valid_indices = labels != -1
        img_feats = img_feats[valid_indices]
        gene_exprs = gene_exprs[valid_indices]
        coords = coords[valid_indices]
        labels = labels[valid_indices]

        # Binarize labels: Label 2 is Positive (1), others are Negative (0)
        labels = np.where(labels == 2, 1, 0)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        dataset = AdvancedSpatialDataset(
            sample=sample,
            img_feats=img_feats,
            gene_exprs=gene_exprs,
            coords=coords,
            labels=labels,
            scales=scales,
            directions=directions
        )

        slices.append({
            'sample': sample,
            'dataset': dataset,
            'idx': i,
            'img_dim': img_feats.shape[1],
            'gene_dim': gene_exprs.shape[1]
        })

    return slices, all_samples


def load_test_datasets(npz_path="", label_base_path="", scales=[3, 6, 12], directions=4):
    """
    Load ST-colon test datasets (4 slices).
    Note: Handles splitting a single large array based on predefined spot counts.
    """
    test_data = np.load(npz_path, allow_pickle=True)

    test_img = test_data['img_feats'].astype(np.float64)
    test_expr = test_data['gene_feats'].astype(np.float64)
    test_coords = test_data['positions'].astype(np.float64)

    # Predefined spot counts for the 4 slices
    spot_counts = [3313, 4174, 4007, 3902]
    total_spots = sum(spot_counts)

    # Adjust last slice if total count mismatch (sanity check)
    if len(test_img) != total_spots:
        print(f"Warning: Test set spot count mismatch. Expected: {total_spots}, Actual: {len(test_img)}")
        spot_counts[-1] = len(test_img) - sum(spot_counts[:-1])

    test_datasets = []
    slice_names = ['ST-colon1', 'ST-colon2', 'ST-colon3', 'ST-colon4']

    start_idx = 0
    for i, (slice_name, spot_count) in enumerate(zip(slice_names, spot_counts)):
        end_idx = start_idx + spot_count
        end_idx = min(end_idx, len(test_img))

        slice_img = test_img[start_idx:end_idx]
        slice_expr = test_expr[start_idx:end_idx]
        slice_coords = test_coords[start_idx:end_idx]

        # Load labels dynamically
        label_path = label_base_path.format(num=i + 1)
        slice_labels = np.loadtxt(label_path, dtype=int)

        # Handle label length mismatch padding if necessary
        if len(slice_labels) != len(slice_img):
            print(f"Warning: Label count mismatch for {slice_name}.")
            if len(slice_labels) > len(slice_img):
                slice_labels = slice_labels[:len(slice_img)]
            else:
                padding = np.zeros(len(slice_img) - len(slice_labels), dtype=int)
                slice_labels = np.concatenate([slice_labels, padding])

        if slice_labels.ndim == 1:
            slice_labels = slice_labels.reshape(-1, 1)

        test_dataset = AdvancedSpatialDataset(
            sample=slice_name,
            img_feats=slice_img,
            gene_exprs=slice_expr,
            coords=slice_coords,
            labels=slice_labels,
            scales=scales,
            directions=directions
        )

        test_datasets.append({
            'name': slice_name,
            'dataset': test_dataset,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'spot_count': len(slice_img)
        })

        print(f"Loaded test slice {slice_name} with {len(slice_img)} spots.")
        start_idx = end_idx

    return test_datasets


def train_model(model, train_loader, val_loader, device, epochs=40, lr=1e-4, dist_type="beta"):
    """
    Train model and determine optimal threshold/distribution on validation set.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_auc = -1.0
    best_gamma = 0.5
    val_dist_params = None
    val_q = 0.5

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        cls_loss_total = 0.0
        cl_loss_total = 0.0
        batch_count = 0

        # Training Loop
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['label']
            batch_size = labels.size(0)

            optimizer.zero_grad()
            pred, total_loss, cls_loss, cl_loss = model(batch, labels=labels)
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * batch_size
            cls_loss_total += cls_loss.item() * batch_size
            cl_loss_total += cl_loss.item() * batch_size
            batch_count += batch_size

        avg_train_loss = train_loss / batch_count
        avg_cls_loss = cls_loss_total / batch_count
        avg_cl_loss = cl_loss_total / batch_count

        # Validation Loop
        model.eval()
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch['label'].cpu().numpy()
                outputs, _ = model(batch, labels=None)
                outputs = outputs.cpu().numpy()
                val_probs.append(outputs)
                val_labels.append(labels)

        val_probs = np.concatenate(val_probs).flatten()
        val_probs = normalize_probs(val_probs)
        val_labels = np.concatenate(val_labels).flatten()

        if len(np.unique(val_labels)) > 1:
            val_auc = roc_auc_score(val_labels, val_probs)
            val_ap = average_precision_score(val_labels, val_probs)
            current_gamma, current_best_f1 = find_best_val_threshold(val_probs, val_labels)
        else:
            val_auc, val_ap = 0.5, 0.5
            current_gamma, current_best_f1 = 0.5, 0.5

        # Save Best Model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_gamma = current_gamma

            # Fit validation distribution
            val_dist, val_dist_params = fit_distribution(val_probs, dist_type=dist_type)
            # Calculate quantile q corresponding to best_gamma
            val_q = val_dist.cdf(best_gamma)

            torch.save({
                'model_state_dict': model.state_dict(),
                'best_gamma': best_gamma,
                'val_dist_type': dist_type,
                'val_dist_params': val_dist_params,
                'val_q': val_q,
                'val_f1': current_best_f1
            }, "temp_best_CRC-to-CRC.pth")

            print(
                f"Saved Best Model (Val AUC={val_auc:.4f}, Val F1={current_best_f1:.4f}, Gamma={best_gamma:.4f}, q={val_q:.4f})")

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Cls Loss: {avg_cls_loss:.4f} | "
              f"CL Loss: {model.cl_weight * avg_cl_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

    # Load best model
    checkpoint = torch.load("temp_best_CRC-to-CRC.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['best_gamma'], checkpoint['val_dist_type'], checkpoint['val_dist_params'], checkpoint[
        'val_q']


def evaluate_combined_test_set(model, seed, test_datasets, device, val_dist_type, val_dist_params, val_q, best_gamma,
                               batch_size=128):
    """
    Evaluate on combined test datasets using Quantile Calibration.
    """
    model.eval()
    base_dir = "../predict_files/my_model/ST_colon/From_CRC"
    os.makedirs(base_dir, exist_ok=True)

    # Prepare combined dataset
    test_ds_list = [td['dataset'] for td in test_datasets]
    sample_names = [td['name'] for td in test_datasets]
    combined_test_dataset = ConcatDataset(test_ds_list)
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False,
                                      collate_fn=spatial_collate_fn)

    # Track slice boundaries for splitting results later
    slice_boundaries = []
    current_idx = 0
    for ds in test_ds_list:
        slice_boundaries.append((current_idx, current_idx + len(ds)))
        current_idx += len(ds)

    # Collect predictions
    all_probs_raw = []
    all_labels = []
    with torch.no_grad():
        for batch in combined_test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['label'].cpu().numpy()
            outputs, _ = model(batch, labels=None)
            outputs = outputs.cpu().numpy()
            all_probs_raw.append(outputs)
            all_labels.append(labels)

    y_probs_raw = np.concatenate(all_probs_raw).flatten()
    y_probs_norm = normalize_probs(y_probs_raw)
    y_true = np.concatenate(all_labels).flatten()

    # Fit Test Distribution
    test_dist, test_dist_params = fit_distribution(y_probs_norm, dist_type=val_dist_type)
    print(f"Val Dist Params ({val_dist_type}): {val_dist_params}")
    print(f"Test Dist Params ({val_dist_type}): {test_dist_params}")

    # Determine Test Threshold (Alpha)
    alpha = get_test_threshold(test_dist, val_q, val_dist_type)
    alpha = np.clip(alpha, 0.001, 0.95)
    print(f"Calibration: Val Gamma={best_gamma:.4f} (q={val_q:.4f}) -> Test Alpha={alpha:.4f}")

    # Generate Predictions
    y_pred = (y_probs_norm > alpha).astype(int)

    # Calculate Overall Metrics
    auc = roc_auc_score(y_true, y_probs_norm) if len(np.unique(y_true)) > 1 else 0.5
    ap = average_precision_score(y_true, y_probs_norm) if len(np.unique(y_true)) > 1 else 0.5
    f1 = f1_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Save per-slice results
    for i, (start_idx, end_idx) in enumerate(slice_boundaries):
        slice_name = sample_names[i]

        # Save calibration details
        calib_file = os.path.join(base_dir, f"{slice_name}_calibration_seed={seed}.txt")
        with open(calib_file, 'w') as f:
            f.write(f"Val Optimal Gamma: {best_gamma:.6f}\n")
            f.write(f"Gamma Quantile (q): {val_q:.6f}\n")
            f.write(f"Test Threshold Alpha: {alpha:.6f}\n")
            f.write(f"Val Distribution Params: {val_dist_params}\n")
            f.write(f"Test Distribution Params: {test_dist_params}\n")

        # Save probabilities and predictions
        slice_probs_raw = y_probs_norm[start_idx:end_idx]
        np.savetxt(os.path.join(base_dir, f"{slice_name}_probs(raw)_seed={seed}.txt"), slice_probs_raw, fmt='%.6f')

        slice_preds = y_pred[start_idx:end_idx]
        np.savetxt(os.path.join(base_dir, f"{slice_name}_preds(Final)_seed={seed}.txt"), slice_preds, fmt='%d')

        print(f"Saved results for slice {slice_name}")

    print(f"Overall Metrics (N={len(y_true)}):")
    print(f"AUC: {auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}")
    print(f"Acc: {accuracy:.4f}, Sens: {sensitivity:.4f}, Spec: {specificity:.4f}")

    return {
        'auc': auc, 'ap': ap, 'f1': f1,
        'sensitivity': sensitivity, 'specificity': specificity, 'accuracy': accuracy,
        'threshold_alpha': alpha, 'val_gamma': best_gamma, 'val_q': val_q
    }


def run_experiment(seed, batch_size, epochs, scales, directions, cl_weight, train_slices, full_train_dataset,
                   test_datasets, dist_type="beta"):
    """Run a single experiment."""
    set_all_seeds(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'!' * 20} Starting Experiment (Seed={seed}) {'!' * 20}")
    print(f"Device: {device}, Dist Type: {dist_type}")

    # Split Train/Val
    val_size = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_set, val_set = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Data Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=spatial_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=spatial_collate_fn)

    # Initialize Model
    img_dim = train_slices[0]['img_dim']
    gene_dim = train_slices[0]['gene_dim']
    model = CancerPredictionModel(
        img_dim=img_dim,
        gene_dim=gene_dim,
        scales=scales,
        hidden_dim=512,
        directions=directions,
        cl_weight=cl_weight
    ).to(device)

    # Train
    print("\nStarting Training & Validation Calibration...")
    model, best_gamma, val_dist_type, val_dist_params, val_q = train_model(
        model, train_loader, val_loader, device, epochs=epochs, lr=1e-4, dist_type=dist_type
    )

    # Evaluate
    print("\nEvaluating on Combined Test Set...")
    test_results = evaluate_combined_test_set(
        model, seed, test_datasets, device,
        val_dist_type, val_dist_params, val_q, best_gamma,
        batch_size=batch_size
    )

    return test_results


def main():
    # Global Configuration
    seeds = [42]  # [24, 42, 123, 345, 567]
    batch_size = 128
    epochs = 30
    scales = [2, 4, 8]
    directions = 4
    cl_weight = 0.1
    dist_type = "beta"

    print(f"\n{'=' * 60}")
    print(f"{'CRC Spatial Transcriptomics Multi-modal Training (Quantile Calibration)':^60}")
    print(f"{'=' * 60}\n")

    # Paths (Generalized)
    train_npz = "../dataset/train_dataset_CRC_new.npz"
    train_label_dir = "../data/CRC/label3"
    test_npz = "../dataset/test_dataset_CRC_new.npz"
    test_label_base = "../data/CRC_new/ST-colon{num}/spot_labels.txt"

    # Load Data
    print("\n===== Loading Data =====")
    print("Loading training slices...")
    train_slices, _ = load_training_slices(
        npz_path=train_npz,
        label_dir=train_label_dir,
        scales=scales,
        directions=directions
    )
    all_train_data = [slice_data['dataset'] for slice_data in train_slices]
    full_train_dataset = ConcatDataset(all_train_data)
    print(f"Total Training Spots: {len(full_train_dataset)}")

    print("Loading test datasets...")
    test_datasets = load_test_datasets(
        npz_path=test_npz,
        label_base_path=test_label_base,
        scales=scales,
        directions=directions
    )
    print(f"Total Test Spots (4 slices): {sum(len(td['dataset']) for td in test_datasets)}")
    print("===== Data Loaded =====")

    # Run Experiments
    all_results = []
    for seed in seeds:
        results = run_experiment(
            seed=seed,
            batch_size=batch_size,
            epochs=epochs,
            scales=scales,
            directions=directions,
            cl_weight=cl_weight,
            train_slices=train_slices,
            full_train_dataset=full_train_dataset,
            test_datasets=test_datasets,
            dist_type=dist_type
        )
        all_results.append(results)

    # Statistics
    aucs = [r['auc'] for r in all_results]
    aps = [r['ap'] for r in all_results]
    f1s = [r['f1'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]
    alphas = [r['threshold_alpha'] for r in all_results]
    gammas = [r['val_gamma'] for r in all_results]

    print("\n" + "=" * 80)
    print("Final Performance (Mean ± Std):")
    print("=" * 80)
    print(f"AUC:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"AP:    {np.mean(aps):.4f} ± {np.std(aps):.4f}")
    print(f"F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Acc:   {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Alpha: {np.mean(alphas):.4f} ± {np.std(alphas):.4f}")
    print(f"Gamma: {np.mean(gammas):.4f} ± {np.std(gammas):.4f}")
    print("=" * 80)

    # Save Stats
    stats_df = pd.DataFrame({
        'auc_mean': [np.mean(aucs)], 'auc_std': [np.std(aucs)],
        'ap_mean': [np.mean(aps)], 'ap_std': [np.std(aps)],
        'f1_mean': [np.mean(f1s)], 'f1_std': [np.std(f1s)],
        'accuracy_mean': [np.mean(accuracies)], 'accuracy_std': [np.std(accuracies)],
        'alpha_mean': [np.mean(alphas)], 'alpha_std': [np.std(alphas)],
        'gamma_mean': [np.mean(gammas)], 'gamma_std': [np.std(gammas)]
    })

    output_path = "../predict_files/my_model/ST_colon/From_CRC/final_statistics.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stats_df.to_csv(output_path, index=False)
    print(f"\nFinal statistics saved to {output_path}")


if __name__ == "__main__":
    main()