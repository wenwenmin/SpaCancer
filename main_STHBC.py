import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import os
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
import random
from datetime import datetime

from model import (
    SpatialCancerClassifier as CancerPredictionModel,
)
from Dataset import (AdvancedSpatialDataset,
    spatial_collate_fn)
from CAPT import *



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



def load_her2st_slices(npz_path, label_dir, scales=[3, 6, 12], directions=4):
    """
    Load data and split by slices. Constructs spatial graph structures.
    """
    data = np.load(npz_path, allow_pickle=True)
    all_samples = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']

    slices = []
    for i, sample in enumerate(all_samples):
        img_feats = data['img_feats'][i]
        gene_exprs = data['gene_feats'][i]
        coords = data['coords'][i]

        labels_path = os.path.join(label_dir, f"{sample}-labels3.txt")
        labels = np.loadtxt(labels_path, dtype=int)

        # Filter valid indices (exclude label -1)
        valid_indices = labels != -1
        img_feats = img_feats[valid_indices]
        gene_exprs = gene_exprs[valid_indices]
        coords = coords[valid_indices]
        labels = labels[valid_indices]

        print(f"Slice {sample}: Raw spots: {len(data['img_feats'][i])}, Filtered: {len(img_feats)}")

        # Binarize labels: Label 2 is Positive (1), others are Negative (0)
        labels = np.where(labels == 2, 1, 0)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        unique_labels, label_counts = np.unique(labels, return_counts=True)
        print(f"--- Label Stats for {sample} ---")
        for label, count in zip(unique_labels, label_counts):
            print(f"  Label {label}: {count} spots")
        print("-" * 30)

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

        dataset.sample = sample
        slices.append({
            'sample': sample,
            'dataset': dataset,
            'idx': i,
            'img_dim': img_feats.shape[1],
            'gene_dim': gene_exprs.shape[1]
        })

    return slices, all_samples


def train_fold(model, train_loader, val_loader, device, epochs=30, lr=1e-4, dist_type="beta"):
    """
    Training function with quantile calibration logic.
    Finds the optimal threshold on validation set and maps it to a quantile.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_auc = -1.0
    best_gamma = 0.5  # Optimal threshold for validation
    val_dist_params = None  # Validation distribution parameters
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
            # Find optimal gamma (threshold) maximizing F1 on validation
            current_gamma, current_best_f1 = find_best_val_threshold(val_probs, val_labels)
        else:
            val_auc = 0.5
            val_ap = 0.5
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
                'val_auc': best_val_auc
            }, "temp_best-HER2ST11.pth")
            print(
                f"Saved Best Model (Val AUC={best_val_auc:.3f}, F1={current_best_f1:.3f}, Gamma={best_gamma:.3f}, q={val_q:.3f})")

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.3f} | "
              f"Cls Loss: {avg_cls_loss:.3f} | "
              f"CL Loss: {model.cl_weight * avg_cl_loss:.3f} | "
              f"Val AUC: {val_auc:.3f} | Val AP: {val_ap:.3f}")

    # Load best model
    checkpoint = torch.load("temp_best-HER2ST11.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['best_gamma'], checkpoint['val_dist_type'], checkpoint['val_dist_params'], checkpoint[
        'val_q']


def evaluate_fold(model, test_loader, device, seed, slice_name, base_dir, val_dist_type, val_dist_params, val_q,
                  best_gamma):
    """
    Evaluation function using quantile-based threshold calibration to prevent data leakage.
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
    y_probs_raw = normalize_probs(y_probs_raw)
    y_true = np.concatenate(all_labels).flatten()

    # Step 1: Fit Test Distribution (Unsupervised - no labels used)
    test_dist, test_dist_params = fit_distribution(y_probs_raw, dist_type=val_dist_type)
    print(f"Val Dist Params ({val_dist_type}): {val_dist_params}")
    print(f"Test Dist Params ({val_dist_type}): {test_dist_params}")

    # Step 2: Determine Test Threshold alpha using Quantile Matching
    # alpha = F_test^-1(q_val)
    alpha = get_test_threshold(test_dist, val_q, val_dist_type)
    alpha = np.clip(alpha, 0.001, 0.95)
    print(f"Calibration: Val Gamma={best_gamma:.4f} (q={val_q:.4f}) -> Test Alpha={alpha:.4f}")

    # Step 3: Generate Predictions
    y_pred = (y_probs_raw > alpha).astype(int)

    # Metrics
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_probs_raw)
        ap = average_precision_score(y_true, y_probs_raw)
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    else:
        auc, ap, f1 = 0.5, 0.5, 0.5
        sensitivity, specificity = 0.5, 0.5
        accuracy = 0.5

    # Save Results
    os.makedirs(base_dir, exist_ok=True)

    # Save calibration details
    calib_file = os.path.join(base_dir, f"{slice_name}_calibration_seed={seed}.txt")
    with open(calib_file, 'w') as f:
        f.write(f"Val Optimal Gamma: {best_gamma:.6f}\n")
        f.write(f"Gamma Quantile (q): {val_q:.6f}\n")
        f.write(f"Test Threshold Alpha: {alpha:.6f}\n")
        f.write(f"Val Distribution Params: {val_dist_params}\n")
        f.write(f"Test Distribution Params: {test_dist_params}\n")

    # Save raw probabilities and final predictions
    raw_file = os.path.join(base_dir, f"{slice_name}_probs(raw)_seed={seed}.txt")
    np.savetxt(raw_file, y_probs_raw, fmt='%.6f')

    pred_file = os.path.join(base_dir, f"{slice_name}_preds(Final)_seed={seed}.txt")
    np.savetxt(pred_file, y_pred, fmt='%d')

    print(f"Results saved for {slice_name}.")

    return auc, ap, f1, y_probs_raw, y_true, sensitivity, specificity, alpha, accuracy


def leave_one_slice_out_cv(slices, all_samples, device, batch_size=64, epochs=30, seed=42,
                           scales=[3, 6, 12], directions=8, cl_weight=0.1, dist_type="beta"):
    """
    Leave-One-Out Cross-Validation (LOOCV).
    Integrates quantile calibration to adapt thresholds across domains.
    """
    set_all_seeds(seed)
    num_slices = len(slices)
    fold_metrics = []
    base_dir = "../predict_files/my_model/her2st/LOOCV/"

    for test_idx in range(num_slices):
        test_sample = slices[test_idx]['sample']
        print(f"\n{'=' * 10} LOOCV: Testing Slice {test_sample} {'=' * 10}")

        # Prepare Training Data (All slices except the test slice)
        all_train_data = []
        img_dim = slices[0]['img_dim']
        gene_dim = slices[0]['gene_dim']
        for i in range(num_slices):
            if i != test_idx:
                all_train_data.append(slices[i]['dataset'])

        # Split Train/Val
        all_train_dataset = ConcatDataset(all_train_data)
        val_size = int(0.2 * len(all_train_dataset))
        train_size = len(all_train_dataset) - val_size
        train_set, val_set = random_split(
            all_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )

        # Data Loaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=spatial_collate_fn)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=spatial_collate_fn)
        test_loader = DataLoader(slices[test_idx]['dataset'], batch_size=batch_size, shuffle=False,
                                 collate_fn=spatial_collate_fn)

        print(
            f"Train Spots: {len(train_set)} | Val Spots: {len(val_set)} | Test Spots: {len(slices[test_idx]['dataset'])}")

        # Initialize Model
        model = CancerPredictionModel(
            img_dim=img_dim,
            gene_dim=gene_dim,
            scales=scales,
            hidden_dim=512,
            directions=directions,
            cl_weight=cl_weight
        ).to(device)

        # Train and get calibration parameters
        model, best_gamma, val_dist_type, val_dist_params, val_q = train_fold(
            model, train_loader, val_loader, device,
            epochs=epochs,
            dist_type=dist_type
        )

        # Evaluate on Test Slice
        auc, ap, f1, preds, labels, sensitivity, specificity, alpha, acc = evaluate_fold(
            model, test_loader, device,
            seed=seed,
            slice_name=test_sample,
            base_dir=base_dir,
            val_dist_type=val_dist_type,
            val_dist_params=val_dist_params,
            val_q=val_q,
            best_gamma=best_gamma
        )

        fold_metrics.append({
            'sample': test_sample,
            'auc': auc,
            'ap': ap,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': acc,
            'threshold_alpha': alpha
        })

        print(f"Result {test_sample}: AUC={auc:.3f}, AP={ap:.3f}, F1={f1:.3f}, "
              f"Sens={sensitivity:.3f}, Spec={specificity:.3f}, Acc={acc:.3f}, Alpha={alpha:.4f}")

    return fold_metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Hyperparameters
    seeds = [24,42,123,345,567]
    batch_size = 128
    epochs = 30
    data_path = "../dataset/HER2ST_dataset.npz"
    label_dir = "../data/HB/her2st/label3"
    scales = [2, 4, 8]
    directions = 4
    cl_weight = 0.1
    dist_type = "beta"

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Start Time: {start_time}")
    print(f"Distribution Type: {dist_type}")

    print("Loading slices...")
    slices, all_samples = load_her2st_slices(
        npz_path=data_path,
        label_dir=label_dir,
        scales=scales,
        directions=directions
    )
    print(f"Loaded {len(slices)} slices.")

    all_metrics = []

    for seed in seeds:
        print(f"\n{'!' * 20} Starting Cross-Validation (Seed={seed}) {'!' * 20}")
        fold_metrics = leave_one_slice_out_cv(
            slices, all_samples, device,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            scales=scales,
            directions=directions,
            cl_weight=cl_weight,
            dist_type=dist_type
        )
        all_metrics.append(fold_metrics)

    # Aggregating Results
    print("\n" + "=" * 60)
    print("Average Results per Slice (Mean ± Std):")
    print("=" * 60)

    slice_metrics = {sample: {'aucs': [], 'aps': [], 'f1s': [], 'acc': [], 'alphas': []} for sample in all_samples}
    for seed_metrics in all_metrics:
        for metrics in seed_metrics:
            sample = metrics['sample']
            slice_metrics[sample]['aucs'].append(metrics['auc'])
            slice_metrics[sample]['aps'].append(metrics['ap'])
            slice_metrics[sample]['f1s'].append(metrics['f1'])
            slice_metrics[sample]['acc'].append(metrics['accuracy'])
            slice_metrics[sample]['alphas'].append(metrics['threshold_alpha'])

    for sample in all_samples:
        auc_mean = np.mean(slice_metrics[sample]['aucs'])
        auc_std = np.std(slice_metrics[sample]['aucs'])
        ap_mean = np.mean(slice_metrics[sample]['aps'])
        ap_std = np.std(slice_metrics[sample]['aps'])
        f1_mean = np.mean(slice_metrics[sample]['f1s'])
        f1_std = np.std(slice_metrics[sample]['f1s'])
        alpha_mean = np.mean(slice_metrics[sample]['alphas'])
        alpha_std = np.std(slice_metrics[sample]['alphas'])
        acc_mean = np.mean(slice_metrics[sample]['acc'])
        acc_std = np.std(slice_metrics[sample]['acc'])

        print(f"{sample}: "
              f"AUC={auc_mean:.3f}±{auc_std:.3f}, "
              f"AP={ap_mean:.3f}±{ap_std:.3f}, "
              f"F1={f1_mean:.3f}±{f1_std:.3f}, "
              f"Alpha={alpha_mean:.4f}±{alpha_std:.4f}, "
              f"Acc={acc_mean:.3f}±{acc_std:.3f}")

    # Overall Averages
    all_aucs = [m['auc'] for seed_metrics in all_metrics for m in seed_metrics]
    all_aps = [m['ap'] for seed_metrics in all_metrics for m in seed_metrics]
    all_f1s = [m['f1'] for seed_metrics in all_metrics for m in seed_metrics]
    all_acc = [m['accuracy'] for seed_metrics in all_metrics for m in seed_metrics]
    all_alphas = [m['threshold_alpha'] for seed_metrics in all_metrics for m in seed_metrics]

    print("\n" + "=" * 60)
    print("Overall Performance:")
    print("=" * 60)
    print(f"AUC: {np.mean(all_aucs):.3f} ± {np.std(all_aucs):.3f}")
    print(f"AP:  {np.mean(all_aps):.3f} ± {np.std(all_aps):.3f}")
    print(f"F1:  {np.mean(all_f1s):.3f} ± {np.std(all_f1s):.3f}")
    print(f"ACC: {np.mean(all_acc):.3f} ± {np.std(all_acc):.3f}")
    print(f"Avg Threshold (Alpha): {np.mean(all_alphas):.4f} ± {np.std(all_alphas):.4f}")


if __name__ == "__main__":
    main()