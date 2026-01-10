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


def extract_UNI2h_features_from_spots(model, image_path, positions, crop_size=299, device=None):
    """Extract UNI2h features (consistent with HER2ST processing)."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    H, W, _ = image_np.shape

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
            feat = model(crop)
            feat = feat.view(-1).cpu().numpy()
            features.append(feat)

    return np.stack(features)


def extract_resnet_features_from_spots(image_path, positions, crop_size=299, device=None):
    """Extract features using ResNet-50."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(device)

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
            feat = model(crop)
            feat = feat.view(-1).cpu().numpy()
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
        image = self.transform(pil_image)
        return image


def custom_encode_texts(model, tokenizer, texts, device, max_seq_len=77, batch_size=32):
    """Text encoding function consistent with HER2ST pipeline."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

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


def read(ref_dir, tgt_dir, ref_name, tgt_name, n_genes=768, overlap=None, preprocess=False):
    # Initialize storage
    ref, ref_img, ref_img_resnet, ref_pos, ref_geneEmb, ref_expr_raw = [], [], [], [], [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------- Step 0: Load Models --------------------------
    # NOTE: Please replace with your own HuggingFace token
    hf_token = "YOUR_HF_TOKEN"
    login(hf_token)
    print("HuggingFace login successful!")

    # Load UNI2-h
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

    # -------------------------- Step 1: Process Ref & Target Data --------------------------

    # 1. Read Reference Datasets
    print("Step 1: Processing reference datasets...")
    ref_adata_list = []
    for r in ref_name:
        print(f"  - Reading reference dataset {r}...")
        adata = sc.read(os.path.join(ref_dir, r, f'{r}.h5ad'))
        adata.var_names_make_unique()
        # Filter spots based on tissue mask
        mask = adata.obs.get('in_tissue', np.ones(adata.shape[0], dtype=bool)) == 1
        adata._inplace_subset_obs(mask)
        ref_adata_list.append(adata)

    # 2. Read Target Dataset
    print(f"\nStep 2: Processing target dataset {tgt_name}...")
    tgt_adata_initial = sc.read_visium(tgt_dir)
    tgt_adata_initial.var_names_make_unique()
    mask = tgt_adata_initial.obs['in_tissue'] == 1
    tgt_adata_initial._inplace_subset_obs(mask)

    # 3. Select Top 80% HVGs for All Datasets
    print("\nStep 3: Selecting top 80% HVGs for all datasets...")
    ref_adata_hvg_list = []

    # Process Reference HVGs
    for i, (r, adata) in enumerate(zip(ref_name, ref_adata_list)):
        print(f"  - Reference {r}: Selecting HVGs...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        total_genes = adata.shape[1]
        top_80_percent = int(total_genes * 0.8)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=top_80_percent, subset=False)
        hvg_80pct = adata.var[adata.var['highly_variable']].index.tolist()
        ref_adata_hvg_list.append(adata[:, hvg_80pct])

    # Process Target HVGs
    print(f"  - Target {tgt_name}: Selecting HVGs...")
    sc.pp.normalize_total(tgt_adata_initial, target_sum=1e4)
    sc.pp.log1p(tgt_adata_initial)
    total_genes_tgt = tgt_adata_initial.shape[1]
    top_80_percent_tgt = int(total_genes_tgt * 0.8)
    sc.pp.highly_variable_genes(tgt_adata_initial, flavor="seurat_v3", n_top_genes=top_80_percent_tgt, subset=False)
    hvg_80pct_tgt = tgt_adata_initial.var[tgt_adata_initial.var['highly_variable']].index.tolist()
    tgt_adata_hvg = tgt_adata_initial[:, hvg_80pct_tgt]

    # 4. Find Global Gene Intersection
    print("\nStep 4: Finding global gene intersection (Reference + Target HVGs)...")
    overlap_genes = set(ref_adata_hvg_list[0].var_names)

    # Intersect with all reference datasets
    for adata in ref_adata_hvg_list[1:]:
        overlap_genes &= set(adata.var_names)
    print(f"  - Intersection of reference HVGs: {len(overlap_genes)} genes.")

    # Intersect with target dataset
    overlap_genes &= set(tgt_adata_hvg.var_names)
    overlap_genes = list(overlap_genes)
    print(f"  - Final global intersection (Ref + Tgt HVGs): {len(overlap_genes)} genes.")

    # 5. Filter Datasets with Intersection
    print("\nStep 5: Filtering all datasets with the global gene intersection...")
    ref_adata_filtered = [adata[:, overlap_genes] for adata in ref_adata_hvg_list]
    tgt_adata_filtered = tgt_adata_hvg[:, overlap_genes]
    print("  - All datasets have been filtered.")

    # -------------------------- Step 2: Extract Features (Reference) --------------------------
    print("\nExtracting features for reference datasets...")
    for i, (r, adata) in enumerate(zip(ref_name, ref_adata_filtered)):
        print(f"\nProcessing reference section {r}...")
        position = adata.obsm['spatial']

        # Extract UNI2h Features
        image_path = os.path.join(ref_dir, r, 'he-raw.jpg')
        image_feats = extract_UNI2h_features_from_spots(model_UNI2h, image_path, position, crop_size=256, device=device)
        image_feats = row_min_max_normalize(image_feats)

        # Extract ResNet Features
        resnet_feats = extract_resnet_features_from_spots(image_path, position, crop_size=256, device=device)
        resnet_feats = row_min_max_normalize(resnet_feats)

        # Generate Gene Embeddings
        gene_df = generate_gene_df(ad=adata, house_keeping_genes=empty_hk, num_genes=300, todense=True)
        gene_texts = gene_df["label"].tolist()
        gene_embeddings = custom_encode_texts(
            model=OmiCLIP, tokenizer=tokenizer, texts=gene_texts, max_seq_len=77, device=device, batch_size=32
        )
        gene_embeddings = row_min_max_normalize(gene_embeddings)

        ref_expr_raw.append(adata.copy())
        ref.append(adata)
        ref_img.append(image_feats)
        ref_img_resnet.append(resnet_feats)
        ref_pos.append(position)
        ref_geneEmb.append(gene_embeddings)

    # -------------------------- Step 3: Extract Features (Target) --------------------------
    print(f"\nExtracting features for target dataset {tgt_name}...")
    adata = tgt_adata_filtered
    tgt_position = adata.obsm['spatial']
    tgt_image_path = os.path.join(tgt_dir, 'he-raw.png')

    # Extract UNI2h Features
    tgt_image_feats = extract_UNI2h_features_from_spots(
        model_UNI2h, tgt_image_path, tgt_position, crop_size=224, device=device
    )
    tgt_image_feats = row_min_max_normalize(tgt_image_feats)

    # Extract ResNet Features
    tgt_resnet_feats = extract_resnet_features_from_spots(
        tgt_image_path, tgt_position, crop_size=224, device=device
    )
    tgt_resnet_feats = row_min_max_normalize(tgt_resnet_feats)

    # Generate Gene Embeddings
    tgt_gene_df = generate_gene_df(
        ad=adata, house_keeping_genes=empty_hk, num_genes=300, todense=True
    )
    tgt_gene_texts = tgt_gene_df["label"].tolist()
    tgt_gene_embeddings = custom_encode_texts(
        model=OmiCLIP, tokenizer=tokenizer, texts=tgt_gene_texts, max_seq_len=77, device=device, batch_size=32
    )
    tgt_gene_embeddings = row_min_max_normalize(tgt_gene_embeddings)

    tgt_expr_raw = adata.copy()

    # -------------------------- Step 4: Final HVG Selection & Saving --------------------------
    print("\nSelecting final HVGs for model training...")

    # Select 768 HVGs from global set (using first reference sample as basis)
    print("  - Selecting top 768 HVGs from the global gene set...")
    ref_expr_filtered = [r.copy() for r in ref_expr_raw]
    tgt_expr_filtered = tgt_expr_raw.copy()

    sc.pp.highly_variable_genes(ref_expr_filtered[0], flavor="seurat_v3", n_top_genes=n_genes, subset=True)
    hvg_genes = ref_expr_filtered[0].var_names

    ref_expr_hvg = [d[:, hvg_genes] for d in ref_expr_filtered]
    tgt_expr_hvg = tgt_expr_filtered[:, hvg_genes]

    ref_expr_processed = [
        row_min_max_normalize(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X)
        for adata in ref_expr_hvg
    ]
    tgt_expr_processed = row_min_max_normalize(
        tgt_expr_hvg.X.toarray() if scipy.sparse.issparse(tgt_expr_hvg.X) else tgt_expr_hvg.X
    )

    # Select 1536 HVGs
    print("  - Selecting top 1536 HVGs from the global gene set...")
    ref_expr_filtered_1536 = [r.copy() for r in ref_expr_raw]

    sc.pp.highly_variable_genes(ref_expr_filtered_1536[0], flavor="seurat_v3", n_top_genes=1536, subset=True)
    hvg_genes_1536 = ref_expr_filtered_1536[0].var_names
    ref_expr_hvg_1536 = [d[:, hvg_genes_1536] for d in ref_expr_filtered_1536]
    tgt_expr_hvg_1536 = tgt_expr_filtered[:, hvg_genes_1536]

    ref_expr_processed_1536 = [
        row_min_max_normalize(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X)
        for adata in ref_expr_hvg_1536
    ]
    tgt_expr_processed_1536 = row_min_max_normalize(
        tgt_expr_hvg_1536.X.toarray() if scipy.sparse.issparse(tgt_expr_hvg_1536.X) else tgt_expr_hvg_1536.X
    )

    # Combine Features
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
    tgt_dir = "../data/HB/Visium_HBC/"
    tgt_name = 'VisiumHBC'

    read(ref_dir, tgt_dir, ref_name, tgt_name)