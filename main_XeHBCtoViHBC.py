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
    normalize_probs,
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


def load_training_slices(img_npz_path="", gene_npz_path="", label_path="", scales=[3, 6, 12], directions=4):
    """
    Load training slices (XeHBC Source), reading features from potentially aligned sources.
    """
    # Load image features
    train_data1 = np.load(img_npz_path, allow_pickle=True)
    img_feats = train_data1['img_feats']

    # Load gene features and coordinates
    train_data2 = np.load(gene_npz_path, allow_pickle=True)
    gene_exprs = train_data2['gene_feats']
    coords = train_data2['coords']

    # Load labels
    labels = np.loadtxt(label_path, dtype=int)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    print(
        f"Training Data Dims: img={img_feats.shape}, gene={gene_exprs.shape}, coords={coords.shape}, labels={labels.shape}")

    # Validate dimensions
    assert img_feats.shape[0] == gene_exprs.shape[0] == coords.shape[0] == labels.shape[0], \
        f"Dimension Mismatch: img={img_feats.shape}, gene={gene_exprs.shape}, coords={coords.shape}, labels={labels.shape}"

    # Create Dataset
    dataset = AdvancedSpatialDataset(
        sample="XeHBC_train",
        img_feats=img_feats,
        gene_exprs=gene_exprs,
        coords=coords,
        labels=labels,
        scales=scales,
        directions=directions
    )

    # Return as slice list for compatibility
    slices = [{
        'sample': "XeHBC_train",
        'dataset': dataset,
        'idx': 0,
        'img_dim': img_feats.shape[1],
        'gene_dim': gene_exprs.shape[1]
    }]

    all_samples = ["XeHBC_train"]

    return slices, all_samples


def load_test_dataset(img_npz_path="", gene_npz_path="", label_path="", scales=[3, 6, 12], directions=4):
    """
    Load test dataset (ViHBC Target), reading features from aligned sources.
    """
    # Load image features
    test_data1 = np.load(img_npz_path, allow_pickle=True)
    test_img = test_data1['img_feats']

    # Load gene features and coordinates
    test_data2 = np.load(gene_npz_path, allow_pickle=True)
    test_expr = test_data2['gene_feats']
    test_coords = test_data2['coords']

    # Load labels
    test_label = np.loadtxt(label_path, dtype=int)
    if test_label.ndim == 1:
        test_label = test_label.reshape(-1, 1)

    print(
        f"Test Data Dims: img={test_img.shape}, gene={test_expr.shape}, coords={test_coords.shape}, labels={test_label.shape}")

    # Validate dimensions
    assert test_img.shape[0] == test_expr.shape[0] == test_coords.shape[0] == test_label.shape[0], \
        f"Dimension Mismatch: img={test_img.shape}, gene={test_expr.shape}, coords={test_coords.shape}, labels={test_label.shape}"

    # Create Dataset
    test_dataset = AdvancedSpatialDataset(
        sample="ViHBC_test",
        img_feats=test_img,
        gene_exprs=test_expr,
        coords=test_coords,
        labels=test_label,
        scales=scales,
        directions=directions
    )

    return test_dataset


def train_model(model, train_loader, val_loader, device, epochs=40, lr=1e-4, dist_type="beta"):
    """
    Train model and determine optimal threshold/distribution on validation set.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_auc = -1.0
    best_gamma = 0.5  # Optimal threshold on validation
    val_dist_params = None
    val_q = 0.5  # Quantile corresponding to gamma

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        cls_loss_total = 0.0
        cl_loss_total = 0.0
        batch_count = 0

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
            }, "temp_best_XetoVi.pth")

            print(
                f"Saved Best Model (Val AUC={val_auc:.4f}, Val F1={current_best_f1:.4f}, Gamma={best_gamma:.4f}, q={val_q:.4f})")

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Cls Loss: {avg_cls_loss:.4f} | "
              f"CL Loss: {model.cl_weight * avg_cl_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

    # Load best model
    checkpoint = torch.load("temp_best_XetoVi.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['best_gamma'], checkpoint['val_dist_type'], checkpoint['val_dist_params'], checkpoint[
        'val_q']


def evaluate_test_set(model, seed, test_loader, device, best_gamma, val_dist_type, val_dist_params, val_q):
    """
    Evaluate on Test Set using Quantile Calibration (Dynamic Threshold).
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['label'].cpu().numpy()
            outputs, _ = model(batch, labels=None)
            outputs = outputs.cpu().numpy()
            all_probs.append(outputs)
            all_labels.append(labels)

    y_probs = np.concatenate(all_probs).flatten()
    y_true = np.concatenate(all_labels).flatten()

    # Normalize and Fit Test Distribution
    y_probs = normalize_probs(y_probs)
    test_dist, test_dist_params = fit_distribution(y_probs, dist_type=val_dist_type)
    print(f"Val Dist Params ({val_dist_type}): {val_dist_params}")
    print(f"Test Dist Params ({val_dist_type}): {test_dist_params}")

    # Determine Test Threshold (Alpha)
    alpha = get_test_threshold(test_dist, val_q, val_dist_type)
    alpha = np.clip(alpha, 0.001, 0.95)
    print(f"Calibration: Val Gamma={best_gamma:.4f} (q={val_q:.4f}) -> Test Alpha={alpha:.4f}")

    # Predictions
    y_pred = (y_probs > alpha).astype(int)


    # Metrics
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_probs)
        ap = average_precision_score(y_true, y_probs)
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    else:
        auc, ap, f1 = 0.5, 0.5, 0.5
        sensitivity, specificity, accuracy = 0.0, 0.0, 0.0

    # Save Results
    base_dir = "../predict_files/my_model/ViHBC/From_XeHBC"
    os.makedirs(base_dir, exist_ok=True)

    # Save calibration details
    calib_file = f"{base_dir}/calibration_seed={seed}.txt"
    with open(calib_file, 'w') as f:
        f.write(f"Val Optimal Gamma: {best_gamma:.6f}\n")
        f.write(f"Gamma Quantile (q): {val_q:.6f}\n")
        f.write(f"Test Threshold Alpha: {alpha:.6f}\n")
        f.write(f"Val Distribution Params: {val_dist_params}\n")
        f.write(f"Test Distribution Params: {test_dist_params}\n")

    # Save files
    np.savetxt(f"{base_dir}/probs(raw)_seed={seed}.txt", y_probs, fmt='%.6f')
    np.savetxt(f"{base_dir}/preds(Final)_seed={seed}.txt", y_pred, fmt='%d')

    print(f"Results saved to {base_dir}")

    return {
        'auc': auc, 'ap': ap, 'f1': f1,
        'sensitivity': sensitivity, 'specificity': specificity, 'accuracy': accuracy,
        'threshold_alpha': alpha
    }


def run_experiment(seed, batch_size, epochs, scales, directions, cl_weight=0.1, dist_type="beta"):
    """Run a single experiment with a specific seed."""
    set_all_seeds(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'!' * 20} Starting Experiment (Seed={seed}) {'!' * 20}")
    print(f"Device: {device}, Dist Type: {dist_type}")

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Start Time: {start_time}")

    # Paths (Generalized)
    # Note: These paths assume specific aligned datasets for Transfer Learning
    train_img_path = "../dataset/test_dataset_XeniumHBC.npz"
    train_gene_path = "../dataset/XeniumHBC_(by_VisiumHBC).npz"
    train_label_path = "../data/HB/Xenium_HBC/spot_labels.txt"

    test_img_path = "../dataset/test_dataset_VisiumHBC.npz"
    test_gene_path = "../dataset/VisiumHBC_(by_XeniumHBC).npz"
    test_label_path = "../data/HB/Visium_HBC/spot_labels.txt"

    # Load Training Data
    print("Loading training slices...")
    train_slices, _ = load_training_slices(
        img_npz_path=train_img_path,
        gene_npz_path=train_gene_path,
        label_path=train_label_path,
        scales=scales,
        directions=directions
    )

    all_train_data = [slice_data['dataset'] for slice_data in train_slices]
    full_train_dataset = ConcatDataset(all_train_data)
    print(f"Total Training Spots: {len(full_train_dataset)}")

    # Split Train/Val
    val_size = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_set, val_set = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Load Test Data
    print("Loading test dataset...")
    test_dataset = load_test_dataset(
        img_npz_path=test_img_path,
        gene_npz_path=test_gene_path,
        label_path=test_label_path,
        scales=scales,
        directions=directions
    )
    print(f"Test Spots: {len(test_dataset)}")

    # Data Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=spatial_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=spatial_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=spatial_collate_fn)

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
    print("\nStarting Training...")
    model, best_gamma, val_dist_type, val_dist_params, val_q = train_model(
        model, train_loader, val_loader, device, epochs=epochs, dist_type=dist_type
    )

    # Evaluate
    print("\nEvaluating on Test Set...")
    test_results = evaluate_test_set(
        model, seed, test_loader, device,
        best_gamma, val_dist_type, val_dist_params, val_q
    )

    print("\n" + "=" * 60)
    print(f"Seed {seed} Results:")
    print("=" * 60)
    print(f"AUC: {test_results['auc']:.4f}")
    print(f"AP:  {test_results['ap']:.4f}")
    print(f"F1:  {test_results['f1']:.4f}")

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
    print(f"{'Transfer Learning: XeHBC -> ViHBC':^60}")
    print(f"{'=' * 60}\n")

    # Run Experiments
    all_results = []
    for seed in seeds:
        results = run_experiment(
            seed, batch_size, epochs, scales, directions,
            cl_weight=cl_weight, dist_type=dist_type
        )
        all_results.append(results)

    # Metrics Calculation
    aucs = [r['auc'] for r in all_results]
    aps = [r['ap'] for r in all_results]
    f1s = [r['f1'] for r in all_results]
    sensitivities = [r['sensitivity'] for r in all_results]
    specificities = [r['specificity'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]
    alphas = [r['threshold_alpha'] for r in all_results]

    # Output Detailed Results
    print("\n" + "=" * 60)
    print("Detailed Results per Seed:")
    print("=" * 60)
    for i, seed in enumerate(seeds):
        print(f"Seed {seed:4d}: AUC={aucs[i]:.4f}, AP={aps[i]:.4f}, F1={f1s[i]:.4f}, Alpha={alphas[i]:.4f}")

    # Output Statistics
    print("\n" + "=" * 60)
    print("Final Performance (Mean ± Std):")
    print("=" * 60)
    print(f"AUC:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"AP:    {np.mean(aps):.4f} ± {np.std(aps):.4f}")
    print(f"F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Sens:  {np.mean(sensitivities):.4f} ± {np.std(sensitivities):.4f}")
    print(f"Spec:  {np.mean(specificities):.4f} ± {np.std(specificities):.4f}")
    print(f"Acc:   {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Alpha: {np.mean(alphas):.4f} ± {np.std(alphas):.4f}")
    print("=" * 60)

    # Save Statistics
    stats_df = pd.DataFrame({
        'Metric': ['AUC', 'AP', 'F1', 'Sensitivity', 'Specificity', 'Accuracy', 'Threshold Alpha'],
        'Mean': [np.mean(aucs), np.mean(aps), np.mean(f1s), np.mean(sensitivities), np.mean(specificities),
                 np.mean(accuracies), np.mean(alphas)],
        'Std': [np.std(aucs), np.std(aps), np.std(f1s), np.std(sensitivities), np.std(specificities),
                np.std(accuracies), np.std(alphas)]
    })
    base_dir = "../predict_files/my_model/ViHBC/From_XeHBC"
    os.makedirs(base_dir, exist_ok=True)
    stats_df.to_csv(f"{base_dir}/final_statistics.csv", index=False)
    print(f"Statistics saved to: {base_dir}/final_statistics.csv")


if __name__ == "__main__":
    main()