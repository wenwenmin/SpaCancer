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
    :return: Normalized features with same shape.
    """
    row_min = features.min(axis=1, keepdims=True)
    row_max = features.max(axis=1, keepdims=True)
    normalized = (features - row_min) / (row_max - row_min + 1e-8)  # Prevent division by zero
    return normalized


def extract_UNI2h_features_from_spots(model, image_path, positions, crop_size=299, device=None):
    """
    Extract UNI2h features from spots.
    Logic is consistent with HER2ST processing pipeline.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = Image.open(image_path).convert('RGB')
    H, W, _ = np.array(image).shape

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    features = []
    for (x, y) in tqdm(positions, desc="Extracting UNI2h features"):
        left = int(x - crop_size // 2)
        upper = int(y - crop_size // 2)
        right = left + crop_size
        lower = upper + crop_size

        left = max(0, left)
        upper = max(0, upper)
        right = min(W, right)
        lower = min(H, lower)

        crop = image.crop((left, upper, right, lower))
        crop = transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(crop).view(-1).cpu().numpy()
            features.append(feat)

    return np.stack(features)


def extract_resnet_features_from_spots(image_path, positions, crop_size=299, device=None):
    """Extract features using ResNet-50."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')

    features = []
    for (x, y) in tqdm(positions, desc="Extracting ResNet features"):
        left = int(x - crop_size // 2)
        upper = int(y - crop_size // 2)
        right = left + crop_size
        lower = upper + crop_size

        left = max(0, left)
        upper = max(0, upper)
        right = min(image.width, right)
        lower = min(image.height, lower)

        crop = image.crop((left, upper, right, lower))
        crop = transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(crop).view(-1).cpu().numpy()
            features.append(feat)

    return np.stack(features)


class ROIDataset(Dataset):
    def __init__(self, img_list, transform):
        super().__init__()
        self.images_lst = img_list
        self.transform = transform

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        pil_image = Image.fromarray(self.images_lst[idx].astype('uint8'))
        return self.transform(pil_image)


def custom_encode_texts(model, tokenizer, texts, device, max_seq_len=77, batch_size=32):
    """Text encoding function consistent with HER2ST pipeline."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Truncate overly long text
            truncated_texts = []
            for text in batch_texts:
                words = text.split()
                if len(words) > max_seq_len - 2:
                    words = words[:max_seq_len - 2]
                truncated_texts.append(" ".join(words))

            text_inputs = tokenizer(truncated_texts).to(device)
            text_embeds = model.encode_text(text_inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # Normalize
            embeddings.append(text_embeds.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def read(ref_dir, tgt_dir, ref_name, tgt_name, n_genes=768, overlap=None, preprocess=False):
    # Initialize storage
    ref, ref_img, ref_img_resnet, ref_pos, ref_geneEmb = [], [], [], [], []
    ref_expr_raw = []  # Stores normalized reference adata (for later HVG selection)
    tgt_img, tgt_img_resnet, tgt_pos, tgt_geneEmb = [], [], [], []
    tgt_expr_raw = None  # Stores normalized target adata (for later HVG selection)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------- Step 0: Load Models --------------------------
    # NOTE: Please replace with your own HuggingFace token
    hf_token = "YOUR_HF_TOKEN"
    login(hf_token)
    print("HuggingFace login successful!")

    # Load UNI2h
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 5.33334,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU,
        'reg_tokens': 8, 'dynamic_img_size': True
    }
    model_UNI2h = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    print("UNI2-h model loaded.")
    model_UNI2h.eval().to(device)

    # Load OmiCLIP
    from loki.preprocess import generate_gene_df
    from loki.utils import load_model
    print("Loading OmiCLIP model...")
    model_weight_path = "../OmiCLIP/checkpoint.pt"
    OmiCLIP, preprocess, tokenizer = load_model(model_path=model_weight_path, device=device)
    OmiCLIP = OmiCLIP.to(device)

    empty_hk = pd.DataFrame(columns=["genesymbol"])

    # -------------------------- Step 1: Pre-process & Normalize Data --------------------------


    print("Step 1: Processing reference datasets...")
    ref_adata_hvg_list = []

    for r in ref_name:
        print(f"  - Reading reference dataset {r}...")
        adata = sc.read(os.path.join(ref_dir, r, f'{r}.h5ad'))
        adata.var_names_make_unique()

        # Filter in-tissue spots
        mask = adata.obs.get('in_tissue', np.ones(adata.shape[0], dtype=bool)) == 1
        adata._inplace_subset_obs(mask)

        # Primary Normalization (Executed only here)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)


        total_genes = adata.shape[1]
        top_percent = int(
            total_genes * 1)  # Note: Code implies 100% selection here based on * 1, but follows logic structure
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=top_percent, subset=False)

        hvg_pct = adata.var[adata.var['highly_variable']].index.tolist()
        adata_hvg = adata[:, hvg_pct].copy()

        ref_adata_hvg_list.append(adata_hvg)
        ref_expr_raw.append(adata)  # Keep full normalized adata for later


    print(f"\nStep 2: Processing target dataset {tgt_name}...")
    tgt_adata_initial = sc.read_h5ad(os.path.join(tgt_dir, "spot level/transfered.h5ad"))
    tgt_adata_initial.var_names_make_unique()

    # Primary Normalization (Executed only here)
    sc.pp.normalize_total(tgt_adata_initial, target_sum=1e4)
    sc.pp.log1p(tgt_adata_initial)


    total_genes_tgt = tgt_adata_initial.shape[1]
    top_percent_tgt = int(total_genes_tgt * 1)  # Note: Code implies 100% here
    sc.pp.highly_variable_genes(tgt_adata_initial, flavor="seurat_v3", n_top_genes=top_percent_tgt, subset=False)

    hvg_pct_tgt = tgt_adata_initial.var[tgt_adata_initial.var['highly_variable']].index.tolist()
    tgt_adata_hvg = tgt_adata_initial[:, hvg_pct_tgt].copy()
    tgt_expr_raw = tgt_adata_initial  # Keep full normalized adata

    # 3. Find Intersection of HVGs (Ref + Tgt)
    print("\nStep 3: Finding global gene intersection (ref + tgt HVGs)...")
    overlap_genes = set(ref_adata_hvg_list[0].var_names)
    for adata in ref_adata_hvg_list[1:]:
        overlap_genes &= set(adata.var_names)
    print(f"  - Intersection of reference HVGs: {len(overlap_genes)} genes.")

    overlap_genes &= set(tgt_adata_hvg.var_names)
    overlap_genes = list(overlap_genes)
    print(f"  - Final global intersection (Ref + Tgt HVGs): {len(overlap_genes)} genes.")

    # 4. Filter All Datasets by Intersection (Using Pre-normalized Data)
    print("\nStep 4: Filtering all datasets with the global gene intersection...")
    # Reference: Crop from pre-normalized adata
    ref_adata_filtered = [adata[:, overlap_genes] for adata in ref_expr_raw]
    # Target: Crop from pre-normalized adata
    tgt_adata_filtered = tgt_expr_raw[:, overlap_genes].copy()
    print("  - All datasets have been filtered.")

    # -------------------------- Step 2: Feature Extraction --------------------------
    # (Reusing pre-normalized data, no re-normalization needed)

    # Process Reference Datasets
    print("\nExtracting features for reference datasets...")
    for i, (r, adata) in enumerate(zip(ref_name, ref_adata_filtered)):
        print(f"\nProcessing reference section {r}...")
        position = adata.obsm['spatial']

        # Extract UNI2h Features
        image_path = os.path.join(ref_dir, r, 'he-raw.jpg')
        print(f"  - Extracting UNI2h features...")
        image_feats = extract_UNI2h_features_from_spots(model_UNI2h, image_path, position, crop_size=256, device=device)
        image_feats = row_min_max_normalize(image_feats)

        # Extract ResNet Features
        print(f"  - Extracting ResNet features...")
        resnet_feats = extract_resnet_features_from_spots(image_path, position, crop_size=256, device=device)
        resnet_feats = row_min_max_normalize(resnet_feats)

        # Generate Gene Embeddings
        print(f"  - Generating gene embeddings with OmiCLIP...")
        gene_df = generate_gene_df(ad=adata, house_keeping_genes=empty_hk, num_genes=len(overlap_genes), todense=True)
        gene_texts = gene_df["label"].tolist()
        gene_embeddings = custom_encode_texts(
            model=OmiCLIP, tokenizer=tokenizer, texts=gene_texts, max_seq_len=77, device=device, batch_size=32
        )
        gene_embeddings = row_min_max_normalize(gene_embeddings)

        ref.append(adata)
        ref_img.append(image_feats)
        ref_img_resnet.append(resnet_feats)
        ref_pos.append(position)
        ref_geneEmb.append(gene_embeddings)

    # Process Target Dataset
    print(f"\nExtracting features for target dataset {tgt_name}...")
    adata = tgt_adata_filtered
    tgt_position = adata.obsm['spatial']
    tgt_image_path = os.path.join(tgt_dir, 'spot level/image.jpg')

    # Extract UNI2h Features
    print(f"  - Extracting UNI2h features...")
    tgt_image_feats = extract_UNI2h_features_from_spots(
        model_UNI2h, tgt_image_path, tgt_position, crop_size=224, device=device
    )
    tgt_image_feats = row_min_max_normalize(tgt_image_feats)

    # Extract ResNet Features
    print(f"  - Extracting ResNet features...")
    tgt_resnet_feats = extract_resnet_features_from_spots(
        tgt_image_path, tgt_position, crop_size=224, device=device
    )
    tgt_resnet_feats = row_min_max_normalize(tgt_resnet_feats)

    # Generate Gene Embeddings
    print(f"  - Generating gene embeddings with OmiCLIP...")
    tgt_gene_df = generate_gene_df(
        ad=adata, house_keeping_genes=empty_hk, num_genes=len(overlap_genes), todense=True
    )
    tgt_gene_texts = tgt_gene_df["label"].tolist()
    tgt_gene_embeddings = custom_encode_texts(
        model=OmiCLIP, tokenizer=tokenizer, texts=tgt_gene_texts, max_seq_len=77, device=device, batch_size=32
    )
    tgt_gene_embeddings = row_min_max_normalize(tgt_gene_embeddings)

    # -------------------------- Step 3: Final HVG Selection & Saving --------------------------
    # Note: We reuse the already normalized adata (ref_adata_filtered/tgt_adata_filtered)

    print("\nSelecting final HVGs for model training...")
    print("  - Selecting top 768 HVGs from the global gene set...")

    ref_expr_filtered = ref_adata_filtered  # Already normalized
    tgt_expr_filtered = tgt_adata_filtered  # Already normalized

    # Use the first reference sample as the basis for HVG selection
    sc.pp.highly_variable_genes(ref_expr_filtered[0], flavor="seurat_v3", n_top_genes=n_genes, subset=True)
    hvg_genes = ref_expr_filtered[0].var_names

    ref_expr_hvg = [d[:, hvg_genes] for d in ref_expr_filtered]
    tgt_expr_hvg = tgt_expr_filtered[:, hvg_genes]

    # Convert to array and normalize (Min-Max)
    ref_expr_processed = [
        row_min_max_normalize(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X)
        for adata in ref_expr_hvg
    ]
    tgt_expr_processed = row_min_max_normalize(
        tgt_expr_hvg.X.toarray() if scipy.sparse.issparse(tgt_expr_hvg.X) else tgt_expr_hvg.X
    )

    # Select 1536 HVGs
    print("  - Selecting top 1536 HVGs from the global gene set...")
    ref_expr_filtered_1536 = [adata.copy() for adata in ref_adata_filtered]  # Copy normalized adata

    sc.pp.highly_variable_genes(ref_expr_filtered_1536[0], flavor="seurat_v3", n_top_genes=1536, subset=True)
    hvg_genes_1536 = ref_expr_filtered_1536[0].var_names
    ref_expr_hvg_1536 = [d[:, hvg_genes_1536] for d in ref_expr_filtered_1536]
    tgt_expr_hvg_1536 = tgt_adata_filtered[:, hvg_genes_1536]

    ref_expr_processed_1536 = [
        row_min_max_normalize(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X)
        for adata in ref_expr_hvg_1536
    ]
    tgt_expr_processed_1536 = row_min_max_normalize(
        tgt_expr_hvg_1536.X.toarray() if scipy.sparse.issparse(tgt_expr_hvg_1536.X) else tgt_expr_hvg_1536.X
    )

    # Combine Features and Save
    print("\nCombining features and saving datasets...")
    ref_combined_gene_features = [
        np.concatenate([emb, expr], axis=1)
        for emb, expr in zip(ref_geneEmb, ref_expr_processed)
    ]
    tgt_combined_gene_features = np.concatenate([tgt_gene_embeddings, tgt_expr_processed], axis=1)

    # Save Training Dataset
    np.savez_compressed(
        f"../dataset/train_dataset_{tgt_name}.npz",
        img_feats=np.array(ref_img, dtype=object),
        img_resnet=np.array(ref_img_resnet, dtype=object),
        gene_feats=np.array(ref_combined_gene_features, dtype=object),
        coords=np.array(ref_pos, dtype=object),
        gene_expr=np.array(ref_expr_processed_1536, dtype=object)
    )
    print(f"  - Training dataset saved.")

    # Save Test Dataset
    np.savez_compressed(
        f"../dataset/test_dataset_{tgt_name}.npz",
        img_feats=np.array(tgt_image_feats),
        img_resnet=np.array(tgt_resnet_feats),
        gene_feats=np.array(tgt_combined_gene_features),
        coords=np.array(tgt_position),
        gene_expr=np.array(tgt_expr_processed_1536)
    )
    print(f"  - Test dataset saved.")
    print("\nAll processing finished successfully!")


if __name__ == "__main__":
    ref_dir = "../data/HB/her2st/"
    ref_name = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']
    tgt_dir = "../data/HB/Xenium_HBC/"
    tgt_name = 'XeniumHBC'

    read(ref_dir, tgt_dir, ref_name, tgt_name)