import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
import os
import numpy as np
import random
from datetime import datetime
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

# Imports aligned with main_STHBC.py structure
from model import SpatialCancerClassifier as CancerPredictionModel
from Dataset import AdvancedSpatialDataset, spatial_collate_fn
from CAPT import (
    find_best_val_threshold,
    fit_distribution,
    get_test_threshold,
    normalize_probs
)


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


def load_original_train_as_test(npz_path="", label_dir="", scales=[3, 6, 12], directions=4):
    """
    Load the original training set (HER2ST) to be used as the TEST set for this experiment.
    """
    data = np.load(npz_path, allow_pickle=True)
    all_samples = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']

    test_datasets = []
    img_dims = []
    gene_dims = []

    for i, sample in enumerate(all_samples):
        img_feats = data['img_feats'][i]
        gene_exprs = data['gene_feats'][i]
        coords = data['coords'][i]

        # Load labels
        labels_path = os.path.join(label_dir, f"{sample}-labels3.txt")
        labels = np.loadtxt(labels_path, dtype=int)

        # Filter valid indices
        valid_indices = labels != -2
        img_feats = img_feats[valid_indices]
        gene_exprs = gene_exprs[valid_indices]
        coords = coords[valid_indices]
        labels = labels[valid_indices]

        # Binarize labels
        labels = np.where(labels == 2, 1, 0)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        print(f"Test slice {sample} dims: img={img_feats.shape}, gene={gene_exprs.shape}, labels={labels.shape}")

        test_dataset = AdvancedSpatialDataset(
            sample=sample,
            img_feats=img_feats,
            gene_exprs=gene_exprs,
            coords=coords,
            labels=labels,
            scales=scales,
            directions=directions
        )

        test_datasets.append(test_dataset)
        img_dims.append(img_feats.shape[1])
        gene_dims.append(gene_exprs.shape[1])

        print(f"Slice {sample}: Spots={len(test_dataset)}, Normal={np.sum(labels == 0)}, Cancer={np.sum(labels == 1)}")

    return test_datasets, all_samples, img_dims[0], gene_dims[0]


def load_original_test_as_train(npz_path="", label_path="", scales=[3, 6, 12], directions=4):
    """
    Load the original test set (ViHBC) to be used as the TRAINING set for this experiment.
    """
    train_data = np.load(npz_path, allow_pickle=True)
    train_img = train_data['img_feats']
    train_expr = train_data['gene_feats']
    train_coords = train_data['coords']

    train_label = np.loadtxt(label_path, dtype=int)
    if train_label.ndim == 1:
        train_label = train_label.reshape(-1, 1)

    print(f"Train set dims: img={train_img.shape}, gene={train_expr.shape}, labels={train_label.shape}")
    print(f"Train labels: Normal={np.sum(train_label == 0)}, Cancer={np.sum(train_label == 1)}")

    train_dataset = AdvancedSpatialDataset(
        sample="ViHBC_as_Train",
        img_feats=train_img,
        gene_exprs=train_expr,
        coords=train_coords,
        labels=train_label,
        scales=scales,
        directions=directions
    )

    img_dim = train_img.shape[1]
    gene_dim = train_expr.shape[1]

    return train_dataset, img_dim, gene_dim


def train_model(model, train_loader, val_loader, device, epochs=40, lr=1e-4, dist_type="beta"):
    """
    Train model and determine optimal threshold/distribution on validation set.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_auc = 0.0
    best_gamma = 0.5  # Optimal threshold on validation
    val_dist_params = None
    val_q = 0.5  # Quantile corresponding to gamma

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

        # Validation Metrics
        if len(np.unique(val_labels)) > 1:
            val_auc = roc_auc_score(val_labels, val_probs)
            val_ap = average_precision_score(val_labels, val_probs)
        else:
            val_auc, val_ap = 0.5, 0.5

        # Find best threshold (Gamma) on validation set based on F1
        current_gamma, current_best_f1 = find_best_val_threshold(val_probs, val_labels)

        # Save Best Model (Optimizing for AUC)
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
            }, "temp_best_ViHBC_to_HER2ST.pth")

            print(
                f"Saved Best Model (Val AUC={best_val_auc:.4f}, Val F1={current_best_f1:.4f}, Gamma={best_gamma:.4f}, q={val_q:.4f})")

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Cls Loss: {avg_cls_loss:.4f} | "
              f"CL Loss: {model.cl_weight * avg_cl_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

    # Load best model
    checkpoint = torch.load("temp_best_ViHBC_to_HER2ST.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['best_gamma'], checkpoint['val_dist_type'], checkpoint['val_dist_params'], checkpoint[
        'val_q']


def evaluate_combined_test_set(model, seed, test_datasets, sample_names, device, val_dist_type, val_dist_params, val_q,
                               best_gamma=0, batch_size=128):
    """
    Evaluate on combined test datasets using Quantile Calibration.
    """
    model.eval()
    combined_test_dataset = ConcatDataset(test_datasets)
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False,
                                      collate_fn=spatial_collate_fn)

    # Collect predictions
    all_probs_raw = []
    all_labels = []
    slice_boundaries = []
    current_idx = 0

    # Track slice boundaries
    for dataset in test_datasets:
        slice_boundaries.append((current_idx, current_idx + len(dataset)))
        current_idx += len(dataset)

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
    print(f"Calibration: Val Gamma={best_gamma:.4f} (q={val_q:.4f}) -> Test Alpha={alpha:.4f}")

    # Generate Predictions
    y_pred = (y_probs_norm > alpha).astype(int)

    # Calculate Overall Metrics
    auc = roc_auc_score(y_true, y_probs_norm)
    ap = average_precision_score(y_true, y_probs_norm)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Save Results
    base_dir = "../predict_files/my_model/her2st/From_ViHBC"
    os.makedirs(base_dir, exist_ok=True)

    # Save calibration details
    calib_file = os.path.join(base_dir, f"calibration_seed={seed}.txt")
    with open(calib_file, 'w') as f:
        f.write(f"Val Optimal Gamma: {best_gamma:.6f}\n")
        f.write(f"Gamma Quantile (q): {val_q:.6f}\n")
        f.write(f"Test Threshold Alpha: {alpha:.6f}\n")
        f.write(f"Val Distribution Params: {val_dist_params}\n")
        f.write(f"Test Distribution Params: {test_dist_params}\n")
    print(f"Calibration saved to: {calib_file}")

    # Save per-slice results
    for i, (start_idx, end_idx) in enumerate(slice_boundaries):
        slice_name = sample_names[i]

        slice_probs_raw = y_probs_norm[start_idx:end_idx]
        np.savetxt(os.path.join(base_dir, f"{slice_name}_probs(raw)_seed={seed}.txt"), slice_probs_raw, fmt='%.6f')

        slice_preds = y_pred[start_idx:end_idx]
        np.savetxt(os.path.join(base_dir, f"{slice_name}_preds(Final)_seed={seed}.txt"), slice_preds, fmt='%d')

        print(f"Saved results for {slice_name}")

    print(f"\nFinal Test Set Results (Seed={seed}):")
    print(f"AUC: {auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}")
    print(f"Acc: {accuracy:.4f}, Sens: {sensitivity:.4f}, Spec: {specificity:.4f}")

    return {
        'auc': auc, 'ap': ap, 'f1': f1,
        'sensitivity': sensitivity, 'specificity': specificity, 'accuracy': accuracy,
        'test_alpha': alpha, 'val_gamma': best_gamma
    }


def run_experiment(seed, batch_size, epochs, scales, directions, cl_weight=0.1, dist_type="beta"):
    """Run a single experiment."""
    set_all_seeds(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'!' * 20} Starting Experiment (Seed={seed}) {'!' * 20}")
    print(f"Device: {device}, Dist Type: {dist_type}")

    # Paths (Swapped for Reverse Transfer: ViHBC -> HER2ST)
    # Testing on Original Train (HER2ST), Training on Original Test (ViHBC)
    original_train_npz = "../dataset/train_dataset_VisiumHBC.npz"
    original_test_npz = "../dataset/test_dataset_VisiumHBC.npz"
    original_train_label_dir = "../data/HB/her2st/label3"
    original_test_label_path = "../data/HB/Visium_HBC/spot_labels.txt"

    # 1. Load Training Data (Original ViHBC Test Set)
    print("Loading Training Data (Original ViHBC Test Set)...")
    train_dataset, img_dim, gene_dim = load_original_test_as_train(
        npz_path=original_test_npz,
        label_path=original_test_label_path,
        scales=scales,
        directions=directions
    )

    # Split Train/Val
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_set, val_set = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # 2. Load Test Data (Original HER2ST Train Set)
    print("Loading Test Data (Original HER2ST Train Set)...")
    test_datasets, sample_names, _, _ = load_original_train_as_test(
        npz_path=original_train_npz,
        label_dir=original_train_label_dir,
        scales=scales,
        directions=directions
    )

    # Data Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=spatial_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=spatial_collate_fn)

    # Initialize Model
    model = CancerPredictionModel(
        img_dim=img_dim, gene_dim=gene_dim, scales=scales,
        hidden_dim=512, directions=directions, cl_weight=cl_weight
    ).to(device)

    # Train
    print("\nStarting Training & Calibration...")
    model, best_gamma, val_dist_type, val_dist_params, val_q = train_model(
        model, train_loader, val_loader, device, epochs=epochs, lr=1e-4, dist_type=dist_type
    )

    # Evaluate
    print("\nEvaluating on Test Set...")
    results = evaluate_combined_test_set(
        model, seed, test_datasets, sample_names, device,
        val_dist_type, val_dist_params, val_q, best_gamma, batch_size=batch_size
    )

    return results


def main():
    # Configuration
    seeds = [42]  # [24, 42, 123, 345, 567]
    batch_size = 128
    epochs = 30
    scales = [2, 4, 8]
    directions = 4
    cl_weight = 0.1
    dist_type = "beta"

    print(f"\n{'=' * 60}")
    print(f"{'Reverse Transfer Experiment: ViHBC -> HER2ST':^60}")
    print(f"{'=' * 60}\n")

    # Run Experiments
    all_results = []
    for seed in seeds:
        results = run_experiment(
            seed, batch_size, epochs, scales, directions,
            cl_weight=cl_weight, dist_type=dist_type
        )
        all_results.append(results)

    # Statistics
    def calc_stats(arr): return np.mean(arr), np.std(arr)

    aucs = [r['auc'] for r in all_results]
    aps = [r['ap'] for r in all_results]
    f1s = [r['f1'] for r in all_results]
    alphas = [r['test_alpha'] for r in all_results]
    gammas = [r['val_gamma'] for r in all_results]

    print("\n" + "=" * 80)
    print("Final Statistics (Mean ± Std):")
    print("=" * 80)
    print(f"AUC:   {calc_stats(aucs)[0]:.4f} ± {calc_stats(aucs)[1]:.4f}")
    print(f"AP:    {calc_stats(aps)[0]:.4f} ± {calc_stats(aps)[1]:.4f}")
    print(f"F1:    {calc_stats(f1s)[0]:.4f} ± {calc_stats(f1s)[1]:.4f}")
    print(f"Alpha: {calc_stats(alphas)[0]:.4f} ± {calc_stats(alphas)[1]:.4f}")
    print(f"Gamma: {calc_stats(gammas)[0]:.4f} ± {calc_stats(gammas)[1]:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()