import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
import os
import numpy as np
import random
from datetime import datetime
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


def load_training_slices(npz_path="", label_dir="", scales=[3, 6, 12], directions=4):
    """
    Load training slices (STHBC dataset) and construct spatial graph structures.
    """
    data = np.load(npz_path, allow_pickle=True)
    all_samples = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']

    slices = []

    for i, sample in enumerate(all_samples):
        img_feats = data['img_feats'][i]
        gene_exprs = data['gene_feats'][i]
        coords = data['coords'][i]

        # Load HER2ST format labels
        labels_path = os.path.join(label_dir, f"{sample}-labels3.txt")
        labels = np.loadtxt(labels_path, dtype=int)

        # Filter valid indices (exclude label -1)
        valid_indices = labels != -1
        img_feats = img_feats[valid_indices]
        gene_exprs = gene_exprs[valid_indices]
        coords = coords[valid_indices]
        labels = labels[valid_indices]

        # Binarize labels: Label 2 is Positive (1), others are Negative (0)
        labels = np.where(labels == 2, 1, 0)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        # Create dataset instance
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


def load_test_dataset(npz_path="", label_path="", scales=[3, 6, 12], directions=4):
    """
    Load the target test dataset (ViHBC) as a single large entity.
    """
    test_data = np.load(npz_path, allow_pickle=True)
    # print(test_data.files) # Optional debug
    test_img = test_data['img_feats']
    test_expr = test_data['gene_feats']
    test_coords = test_data['coords']

    # Load test labels
    test_label = np.loadtxt(label_path, dtype=int)
    if test_label.ndim == 1:
        test_label = test_label.reshape(-1, 1)

    # Create test dataset instance
    test_dataset = AdvancedSpatialDataset(
        sample="test",
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
    Train the model and determine optimal threshold/distribution on the validation set.
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
            val_auc = 0.5
            val_ap = 0.5

        # Find best threshold (Gamma) on validation set based on F1
        current_gamma, current_best_f1 = find_best_val_threshold(val_probs, val_labels)

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
            }, "temp_best_VisiumHBC.pth")

            print(
                f"Saved Best Model (Val AUC={best_val_auc:.4f}, Val F1={current_best_f1:.4f}, Gamma={best_gamma:.4f}, q={val_q:.4f})")

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Cls Loss: {avg_cls_loss:.4f} | "
              f"CL Loss: {model.cl_weight * avg_cl_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

    # Load best model
    checkpoint = torch.load("temp_best_VisiumHBC.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['best_gamma'], checkpoint['val_dist_type'], checkpoint['val_dist_params'], checkpoint[
        'val_q']


def evaluate_test_set(model, seed, test_loader, device, val_dist_type, val_dist_params, val_q, best_gamma):
    """
    Evaluate on the test set using dynamic threshold calibration (Quantile Matching).
    """
    model.eval()
    all_probs_raw = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
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

    # Determine Test Threshold (Alpha) using Quantile Matching
    alpha = get_test_threshold(test_dist, val_q, val_dist_type)
    alpha = np.clip(alpha, 0.001, 0.95)
    print(f"Calibration: Val Gamma={best_gamma:.4f} (q={val_q:.4f}) -> Test Alpha={alpha:.4f}")

    # Generate Predictions
    y_pred = (y_probs_norm > alpha).astype(int)

    # Calculate Metrics
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_probs_norm)
        ap = average_precision_score(y_true, y_probs_norm)
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    else:
        auc, ap, f1 = 0.5, 0.5, 0.5
        sensitivity, specificity, accuracy = 0.0, 0.0, 0.0

    # Save Results
    base_dir = "../predict_files/my_model/ViHBC/From_HER2ST"
    os.makedirs(base_dir, exist_ok=True)

    # Save calibration details
    calib_file = os.path.join(base_dir, f"calibration_seed={seed}.txt")
    with open(calib_file, 'w') as f:
        f.write(f"Val Optimal Gamma: {best_gamma:.6f}\n")
        f.write(f"Gamma Quantile (q): {val_q:.6f}\n")
        f.write(f"Test Threshold Alpha: {alpha:.6f}\n")
        f.write(f"Val Distribution Params: {val_dist_params}\n")
        f.write(f"Test Distribution Params: {test_dist_params}\n")

    # Save probabilities and predictions
    np.savetxt(os.path.join(base_dir, f"probs(raw)_seed={seed}.txt"), y_probs_raw, fmt='%.6f')
    np.savetxt(os.path.join(base_dir, f"probs(Norm)_seed={seed}.txt"), y_probs_norm, fmt='%.6f')
    np.savetxt(os.path.join(base_dir, f"preds(Final)_seed={seed}.txt"), y_pred, fmt='%d')

    print(f"Results saved to {base_dir}")

    return {
        'auc': auc, 'ap': ap, 'f1': f1,
        'sensitivity': sensitivity, 'specificity': specificity, 'accuracy': accuracy
    }


def run_experiment(seed, batch_size, epochs, scales, directions, cl_weight=0.1, dist_type="beta"):
    """Run a single experiment with a specific seed."""
    set_all_seeds(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'!' * 20} Starting Experiment (Seed={seed}) {'!' * 20}")
    print(f"Device: {device}, Dist Type: {dist_type}")

    # Paths (Ensure these match your actual directory structure)
    train_npz = "../dataset/train_dataset_VisiumHBC.npz"
    test_npz = "../dataset/test_dataset_VisiumHBC.npz"
    train_label_dir = "../data/HB/her2st/label3"
    test_label_path = "../data/HB/Visium_HBC/spot_labels.txt"

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Start Time: {start_time}")

    # Load Training Data
    print("Loading training slices...")
    train_slices, _ = load_training_slices(
        npz_path=train_npz,
        label_dir=train_label_dir,
        scales=scales,
        directions=directions
    )

    # Concat all training slices
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
        npz_path=test_npz,
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
        val_dist_type, val_dist_params, val_q, best_gamma
    )

    print("\n" + "=" * 60)
    print(f"Seed {seed} Results:")
    print("=" * 60)
    print(f"AUC: {test_results['auc']:.4f}")
    print(f"AP:  {test_results['ap']:.4f}")
    print(f"F1:  {test_results['f1']:.4f}")
    print(f"Acc: {test_results['accuracy']:.4f}")

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
    print(f"{'Spatial Transcriptomics Multi-modal Model Training':^60}")
    print(f"{'=' * 60}\n")

    # Run Experiments
    all_results = []
    for seed in seeds:
        results = run_experiment(
            seed, batch_size, epochs, scales, directions,
            cl_weight=cl_weight, dist_type=dist_type
        )
        all_results.append(results)

    # Aggregate Results
    aucs = [r['auc'] for r in all_results]
    aps = [r['ap'] for r in all_results]
    f1s = [r['f1'] for r in all_results]
    sensitivities = [r['sensitivity'] for r in all_results]
    specificities = [r['specificity'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]

    print("\n" + "=" * 60)
    print("Final Performance (Mean ± Std):")
    print("=" * 60)
    print(f"AUC:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"AP:   {np.mean(aps):.4f} ± {np.std(aps):.4f}")
    print(f"F1:   {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Sens: {np.mean(sensitivities):.4f} ± {np.std(sensitivities):.4f}")
    print(f"Spec: {np.mean(specificities):.4f} ± {np.std(specificities):.4f}")
    print(f"Acc:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()