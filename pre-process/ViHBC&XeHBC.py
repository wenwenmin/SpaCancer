import scanpy as sc
import numpy as np
import scipy.sparse
import torch
from torch.utils.data import Dataset
import timm
from timm.data import create_transform, resolve_data_config
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
import os
import pandas as pd
from huggingface_hub import login
import warnings

# Configuration
warnings.filterwarnings("ignore")
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
Image.MAX_IMAGE_PIXELS = None


def row_min_max_normalize(features: np.ndarray) -> np.ndarray:
    """
    Perform row-wise min-max normalization to scale values to [0, 1].
    :param features: shape = (N, D)
    :return: Normalized features with the same shape.
    """
    row_min = features.min(axis=1, keepdims=True)
    row_max = features.max(axis=1, keepdims=True)
    normalized = (features - row_min) / (row_max - row_min + 1e-8)  # Prevent division by zero
    return normalized


def custom_encode_texts(model, tokenizer, texts, device, max_seq_len=77, batch_size=32):
    """
    Text encoding function consistent with the HER2ST pipeline.
    Encodes gene symbols into feature vectors using the OmiCLIP model.
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Manual text truncation
            truncated_texts = []
            for text in batch_texts:
                words = text.split()
                if len(words) > max_seq_len - 2:
                    words = words[:max_seq_len - 2]
                truncated_texts.append(" ".join(words))

            text_inputs = tokenizer(truncated_texts)
            text_inputs = text_inputs.to(device)

            text_embeds = model.encode_text(text_inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            embeddings.append(text_embeds.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def read(ref_dir, tgt_dir, ref_name, tgt_name, n_genes_feats=768, n_genes_hvg=1536, preprocess=False):
    # Initialize storage lists
    ref = []
    ref_pos = []
    ref_geneEmb = []
    ref_expr_raw = []  # Store raw expression data (unfiltered)
    tgt_pos = []
    tgt_geneEmb = []
    tgt_expr_raw = None  # Store raw target expression data (unfiltered)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------- Step 0: Load Models --------------------------
    # NOTE: Please replace with your own HuggingFace token
    hf_token = "YOUR_HF_TOKEN"
    login(hf_token)
    print("HuggingFace login successful!")

    # Load OmiCLIP
    from loki.preprocess import generate_gene_df
    from loki.utils import load_model
    print("Loading pre-trained OmiCLIP model...")
    model_weight_path = "../OmiCLIP/checkpoint.pt"
    OmiCLIP, preprocess, tokenizer = load_model(model_path=model_weight_path, device=device)
    OmiCLIP = OmiCLIP.to(device)

    # Empty housekeeping genes list (placeholder)
    empty_hk = pd.DataFrame(columns=["genesymbol"])

    # -------------------------- Step 1: Data Loading --------------------------

    # Process Reference Dataset (XeniumHBC)
    print(f"Processing reference {ref_name}...")
    ref_adata = sc.read_h5ad(os.path.join(ref_dir, "spot level/transfered.h5ad"))
    ref_adata.var_names_make_unique()
    ref_position = ref_adata.obsm['spatial']

    ref_expr_raw.append(ref_adata.copy())  # Save raw unfiltered data
    ref.append(ref_adata)
    ref_pos.append(ref_position)
    print(f'Reference section {ref_name} processed.')

    # Process Target Dataset (VisiumHBC)
    print(f"Processing target {tgt_name}...")
    tgt_adata = sc.read_visium(tgt_dir)
    tgt_adata.var_names_make_unique()

    # Filter for in-tissue spots
    mask = tgt_adata.obs['in_tissue'] == 1
    tgt_adata._inplace_subset_obs(mask)
    tgt_position = tgt_adata.obsm['spatial']

    tgt_expr_raw = tgt_adata.copy()  # Save raw unfiltered data
    print(f'Target section {tgt_name} processed.')

    # -------------------------- Step 2: Normalization & HVG Intersection --------------------------
    print("\nStep 1: Perform one-time normalization + log transformation, select HVGs, and intersect.")

    # Collect all raw sample data
    all_samples = ref_expr_raw + [tgt_expr_raw]
    print(f"Total samples to process: {len(all_samples)}")

    all_hvg_sets = []
    all_samples_normed = []  # Store normalized + log transformed adata (done only once)

    for adata in all_samples:
        # One-time normalization and log1p transformation (reused in later steps)
        temp_adata = adata.copy()
        sc.pp.normalize_total(temp_adata, target_sum=1e4)
        sc.pp.log1p(temp_adata)
        all_samples_normed.append(temp_adata)

        # Select all genes as candidates for intersection logic (based on original logic)
        total_genes = len(temp_adata.var)
        n_top = int(1 * total_genes)

        print(f"Sample total genes: {total_genes}, Genes selected for HVG ranking: {n_top}")

        # Rank highly variable genes
        sc.pp.highly_variable_genes(temp_adata, flavor="seurat_v3", n_top_genes=n_top)
        hvg_set = set(temp_adata.var[temp_adata.var.highly_variable].index)
        all_hvg_sets.append(hvg_set)

    # Calculate intersection of HVGs across all samples
    intersection_genes = set.intersection(*all_hvg_sets)
    intersection_genes = list(intersection_genes)
    print(f"Number of overlapping genes across all samples: {len(intersection_genes)}")

    # Crop datasets to intersection genes using the pre-normalized data
    # Reference Datasets
    ref_normed_filtered = [
        adata[:, intersection_genes].copy() for adata in all_samples_normed[:len(ref_expr_raw)]
    ]
    # Target Dataset
    tgt_normed_filtered = all_samples_normed[len(ref_expr_raw)][:, intersection_genes].copy()
    print("All datasets filtered to intersection genes (using pre-normalized data).")

    # -------------------------- Step 3: OmiCLIP Feature Extraction --------------------------
    print("\nStep 2: Extract OmiCLIP embeddings for intersection genes (768 dim).")

    # Reference Gene Embeddings
    print(f"OmiCLIP processing reference {ref_name} genes...")
    ref_gene_df = generate_gene_df(
        ad=ref_normed_filtered[0],
        house_keeping_genes=empty_hk,
        num_genes=len(intersection_genes),
        todense=True
    )
    ref_gene_texts = ref_gene_df["label"].tolist()
    ref_gene_embeddings = custom_encode_texts(
        model=OmiCLIP, tokenizer=tokenizer, texts=ref_gene_texts,
        max_seq_len=77, device=device, batch_size=64
    )
    ref_gene_embeddings = row_min_max_normalize(ref_gene_embeddings)
    ref_geneEmb.append(ref_gene_embeddings)

    # Target Gene Embeddings
    print(f"OmiCLIP processing target {tgt_name} genes...")
    tgt_gene_df = generate_gene_df(
        ad=tgt_normed_filtered,
        house_keeping_genes=empty_hk,
        num_genes=len(intersection_genes),
        todense=True
    )
    tgt_gene_texts = tgt_gene_df["label"].tolist()
    tgt_gene_embeddings = custom_encode_texts(
        model=OmiCLIP, tokenizer=tokenizer, texts=tgt_gene_texts,
        max_seq_len=77, device=device, batch_size=64
    )
    tgt_gene_embeddings = row_min_max_normalize(tgt_gene_embeddings)

    # -------------------------- Step 4: Select 1536 HVGs for Prediction Target --------------------------
    print(f"\nStep 3: Selecting {n_genes_hvg} HVGs from intersection for gene expression targets.")

    # Select 1536 HVGs using the normalized reference data
    ref_1536 = ref_normed_filtered[0].copy()
    sc.pp.highly_variable_genes(ref_1536, flavor="seurat_v3", n_top_genes=n_genes_hvg, subset=True)
    hvg_genes_1536 = ref_1536.var_names

    # Filter and Row-Normalize (min-max)
    ref_expr_hvg_1536 = [d[:, hvg_genes_1536] for d in ref_normed_filtered]
    tgt_expr_hvg_1536 = tgt_normed_filtered[:, hvg_genes_1536]

    ref_expr_processed_1536 = [
        row_min_max_normalize(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X)
        for adata in ref_expr_hvg_1536
    ]
    tgt_expr_processed_1536 = row_min_max_normalize(
        tgt_expr_hvg_1536.X.toarray() if scipy.sparse.issparse(tgt_expr_hvg_1536.X) else tgt_expr_hvg_1536.X
    )

    # -------------------------- Step 5: Select 768 HVGs for Input Features --------------------------
    print(f"\nStep 4: Selecting {n_genes_feats} HVGs from intersection for input gene features.")

    # Select 768 HVGs using the normalized reference data
    ref_768 = ref_normed_filtered[0].copy()
    sc.pp.highly_variable_genes(ref_768, flavor="seurat_v3", n_top_genes=n_genes_feats, subset=True)
    hvg_genes_768 = ref_768.var_names

    # Filter and Row-Normalize (min-max)
    ref_expr_hvg_768 = [d[:, hvg_genes_768] for d in ref_normed_filtered]
    tgt_expr_hvg_768 = tgt_normed_filtered[:, hvg_genes_768]

    ref_expr_processed_768 = [
        row_min_max_normalize(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X)
        for adata in ref_expr_hvg_768
    ]
    tgt_expr_processed_768 = row_min_max_normalize(
        tgt_expr_hvg_768.X.toarray() if scipy.sparse.issparse(tgt_expr_hvg_768.X) else tgt_expr_hvg_768.X
    )

    # -------------------------- Step 6: Feature Concatenation & Saving --------------------------
    print("\nStep 5: Concatenating gene features (OmiCLIP embeddings + 768 HVG expression).")

    # Reference Features
    ref_combined_gene_features = [
        np.concatenate([emb, expr], axis=1)
        for emb, expr in zip(ref_geneEmb, ref_expr_processed_768)
    ]

    # Target Features
    tgt_combined_gene_features = np.concatenate([tgt_gene_embeddings, tgt_expr_processed_768], axis=1)

    print("\nStep 6: Saving results.")
    output_dir = "../dataset/"
    os.makedirs(output_dir, exist_ok=True)

    # Save Target Dataset (VisiumHBC treated as target)
    tgt_save_path = os.path.join(output_dir, "VisiumHBC_(by_XeniumHBC).npz")
    np.savez_compressed(
        tgt_save_path,
        gene_feats=np.array(tgt_combined_gene_features),  # 768 dim OmiCLIP + 768 HVG
        coords=np.array(tgt_position),
        gene_expr=np.array(tgt_expr_processed_1536)  # 1536 HVG targets
    )
    print(f"Target dataset saved: {tgt_save_path}")

    # Save Reference Dataset (XeniumHBC treated as reference)
    ref_save_path = os.path.join(output_dir, "XeniumHBC_(by_VisiumHBC).npz")
    np.savez_compressed(
        ref_save_path,
        gene_feats=np.array(ref_combined_gene_features[0]),
        coords=np.array(ref_position),
        gene_expr=np.array(ref_expr_processed_1536[0])
    )
    print(f"Reference dataset saved: {ref_save_path}")
    print("\nProcessing complete!")


if __name__ == "__main__":
    # Data paths and parameters
    ref_dir = "../data/HB/Xenium_HBC/"
    ref_name = 'Xenium_HBC'
    tgt_dir = "../data/HB/Visium_HBC/"
    tgt_name = 'VisiumHBC'

    # Note: n_genes_feats=768 for gene_feats concatenation, n_genes_hvg=1536 for prediction targets
    read(ref_dir, tgt_dir, ref_name, tgt_name, n_genes_feats=768, n_genes_hvg=1536)