import scanpy as sc
import numpy as np
import scipy.sparse
import torch
from torch.utils.data import Dataset
import timm
from timm.data import create_transform, resolve_data_config
from PIL import Image, ImageDraw
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


def get_coordinate_transform_func(slice_name):
    """Returns the coordinate transformation function for a specific slice."""
    if slice_name == "SN048_1":
        return lambda x, y: (int(x * 2.18) + 480, int(y * 2.18) + 230)
    elif slice_name == "SN048_2":
        return lambda x, y: (int(x * 2.18) + 300, int(y * 2.18) + 160)
    elif slice_name == "SN048_3":
        return lambda x, y: (int(x * 2.18) + 420, int(y * 2.18) + 80)
    elif slice_name == "SN048_4":
        return lambda x, y: (int(x * 2.18) + 200, int(y * 2.18) + 80)
    elif slice_name == "SN84_1":
        return lambda x, y: (int(x * 2.18) + 220, int(y * 2.18) + 200)
    elif slice_name in ["SN84_2", "SN124_4"]:
        return lambda x, y: (int(x * 2.18) + 220, int(y * 2.18) + 240)
    elif slice_name in ["SN123_1", "SN123_2", "SN123_4"]:
        return lambda x, y: (int(x * 2.18) + 270, int(y * 2.18) + 140)
    elif slice_name in ["SN124_1", "SN124_2"]:
        return lambda x, y: (int(x * 2.18) + 220, int(y * 2.18) + 280)
    else:
        return lambda x, y: (int(x * 2.18), int(y * 2.18))


def extract_resnet_features_from_spots_forCRC(image_path, positions, slice_name=' ', crop_size=52, device=None):
    """Extract ResNet-50 features for CRC reference samples."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    positions = [tuple(p) for p in positions]
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    H, W, _ = image_np.shape

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    coord_transform = get_coordinate_transform_func(slice_name)

    resnet50 = models.resnet50(pretrained=True)
    resnet_feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
    resnet_feature_extractor.to(device)
    resnet_feature_extractor.eval()

    features = []
    for (x, y) in tqdm(positions, desc="Extracting ResNet features"):
        x_translated, y_translated = coord_transform(x, y)
        left = int(x_translated - crop_size // 2)
        upper = int(y_translated - crop_size // 2)
        right = left + crop_size
        lower = upper + crop_size

        left = max(0, left)
        upper = max(0, upper)
        right = min(W, right)
        lower = min(H, lower)

        crop = image.crop((left, upper, right, lower))
        crop = transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = resnet_feature_extractor(crop)
            feat = feat.view(-1).cpu().numpy()
            features.append(feat)

    return np.stack(features)


def extract_resnet_features_from_spots_for_target(image_path, positions, crop_size=52, device=None):
    """Extract ResNet-50 features for target samples (specific coordinate transform)."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    positions = [tuple(p) for p in positions]
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    H, W, _ = image_np.shape

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    resnet50 = models.resnet50(pretrained=True)
    resnet_feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
    resnet_feature_extractor.to(device)
    resnet_feature_extractor.eval()

    features = []
    for (x, y) in tqdm(positions, desc="Extracting ResNet features for target"):
        # Specific scaling for target dataset
        x_translated = int(x * 3.3 + 20)
        y_translated = int(y * 3.3 + 20)

        left = int(x_translated - crop_size // 2)
        upper = int(y_translated - crop_size // 2)
        right = left + crop_size
        lower = upper + crop_size

        left = max(0, left)
        upper = max(0, upper)
        right = min(W, right)
        lower = min(H, lower)

        crop = image.crop((left, upper, right, lower))
        crop = transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = resnet_feature_extractor(crop)
            feat = feat.view(-1).cpu().numpy()
            features.append(feat)

    return np.stack(features)


def extract_UNI2h_features_from_spots_forCRC(model, image_path, positions, slice_name, crop_size=52, device=None):
    """Extract UNI2h features for CRC reference samples."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    H, W, _ = image_np.shape

    show_crop_image = image.copy()
    draw = ImageDraw.Draw(show_crop_image)

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    coord_transform = get_coordinate_transform_func(slice_name)

    features = []
    for (x, y) in tqdm(positions, desc="Extracting UNI2h features"):
        x_translated, y_translated = coord_transform(x, y)

        left = int(x_translated - crop_size // 2)
        upper = int(y_translated - crop_size // 2)
        right = left + crop_size
        lower = upper + crop_size

        left = max(0, left)
        upper = max(0, upper)
        right = min(W, right)
        lower = min(H, lower)

        draw.rectangle([left, upper, right, lower], outline='blue', width=2)

        crop = image.crop((left, upper, right, lower))
        crop = transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(crop)
            feat = feat.view(-1).cpu().numpy()
            features.append(feat)

    # Save visualization
    os.makedirs("../data/CRC/slices_image/show_crop/", exist_ok=True)
    show_crop_path = f"../data/CRC/slices_image/show_crop/{slice_name}_show_crop.png"
    show_crop_image.save(show_crop_path)
    print(f"Saved crop visualization to {show_crop_path}")

    return np.stack(features)


def extract_UNI2h_features_from_spots_for_target(model, image_path, positions, crop_size=20, device=None):
    """Extract UNI2h features for target samples."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    H, W, _ = image_np.shape

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    features = []
    for (x, y) in tqdm(positions, desc="Extracting UNI2h features for target"):
        # Specific scaling for target dataset
        x_translated = int(x * 3.3 + 20)
        y_translated = int(y * 3.3 + 20)

        left = int(x_translated - crop_size // 2)
        upper = int(y_translated - crop_size // 2)
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


def read(ref_dir, tgt_dirs, ref_name, tgt_name, n_genes=768, overlap=None, preprocess=False):
    # Initialize storage lists
    ref, ref_img, ref_img_resnet, ref_pos, ref_geneEmb = [], [], [], [], []
    ref_expr_raw = []  # Reference raw expression

    tgt_all_img, tgt_all_img_resnet, tgt_all_pos, tgt_all_geneEmb = [], [], [], []
    tgt_all_expr_raw = []  # Target raw expression

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

    # -------------------------- Step 1: Data Loading & HVG Selection --------------------------
    all_samples = []
    print("Step 1: Loading all samples and selecting top 80% HVGs...")

    # Process Reference Datasets
    ref_raw_list = []
    for r in ref_name:
        print(f"Reading reference sample {r} and selecting 80% HVGs...")
        adata = sc.read_visium(os.path.join(ref_dir, r))
        adata.var_names_make_unique()
        mask = adata.obs['in_tissue'] == 1
        adata._inplace_subset_obs(mask)

        # Calculate HVGs
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        n_total_genes = adata.var.shape[0]
        n_top_80 = int(n_total_genes * 0.8)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_80, subset=False)
        hvg_80 = adata.var[adata.var['highly_variable']].index.tolist()
        adata_hvg80 = adata[:, hvg_80].copy()

        ref_raw_list.append(adata_hvg80)
        all_samples.append(adata_hvg80)

    # Process Target Datasets
    tgt_raw_list = []
    for i, tgt_dir in enumerate(tgt_dirs):
        slice_name = f"ST-colon{i + 1}"
        print(f"Reading target sample {slice_name} and selecting 80% HVGs...")
        tgt_adata = sc.read_h5ad(os.path.join(tgt_dir, 'transformed.h5ad'))
        tgt_adata.var_names_make_unique()
        if 'in_tissue' in tgt_adata.obs.columns:
            mask = tgt_adata.obs['in_tissue'] == 1
            tgt_adata._inplace_subset_obs(mask)

        # Calculate HVGs
        sc.pp.normalize_total(tgt_adata, target_sum=1e4)
        sc.pp.log1p(tgt_adata)
        n_total_genes = tgt_adata.var.shape[0]
        n_top_80 = int(n_total_genes * 0.8)
        sc.pp.highly_variable_genes(tgt_adata, flavor="seurat_v3", n_top_genes=n_top_80, subset=False)
        hvg_80 = tgt_adata.var[tgt_adata.var['highly_variable']].index.tolist()
        tgt_adata_hvg80 = tgt_adata[:, hvg_80].copy()

        tgt_raw_list.append(tgt_adata_hvg80)
        all_samples.append(tgt_adata_hvg80)

    # -------------------------- Step 2: Intersection of Genes --------------------------
    print("Step 2: Calculating gene intersection across all samples...")
    common_genes = set(all_samples[0].var_names)
    for sample in all_samples[1:]:
        common_genes &= set(sample.var_names)
    common_genes = list(common_genes)
    print(f"Number of common genes (Ref + Target): {len(common_genes)}")

    # -------------------------- Step 3: Reference Feature Extraction --------------------------
    print("Step 3: Processing reference datasets (Features + Gene Embeddings)...")
    for i, r in enumerate(ref_name):
        print(f"Processing {r} image...")
        adata = ref_raw_list[i][:, common_genes].copy()
        position = adata.obsm['spatial']
        image_path = os.path.join("../data/CRC/slices_image/", f"{r}.png")

        # Extract UNI2h Features
        image_feats = extract_UNI2h_features_from_spots_forCRC(
            model_UNI2h, image_path, position, slice_name=r, crop_size=32
        )
        image_feats = row_min_max_normalize(image_feats)

        # Extract ResNet Features
        resnet_feats = extract_resnet_features_from_spots_forCRC(
            image_path, position, slice_name=r, crop_size=32, device=device
        )
        resnet_feats = row_min_max_normalize(resnet_feats)

        # Generate Gene Embeddings
        print(f"OmiCLIP processing {r} genes...")
        gene_df = generate_gene_df(
            ad=adata, house_keeping_genes=empty_hk, num_genes=300, todense=True
        )
        gene_texts = gene_df["label"].tolist()
        gene_embeddings = custom_encode_texts(
            model=OmiCLIP, tokenizer=tokenizer, texts=gene_texts,
            max_seq_len=77, device=device, batch_size=64
        )
        gene_embeddings = row_min_max_normalize(gene_embeddings)

        # Store
        ref.append(adata)
        ref_img.append(image_feats)
        ref_img_resnet.append(resnet_feats)
        ref_pos.append(position)
        ref_geneEmb.append(gene_embeddings)
        ref_expr_raw.append(adata.copy())
        print(f'Section {r} processed.')

    # -------------------------- Step 4: Target Feature Extraction --------------------------
    print("Step 4: Processing target datasets (Features + Gene Embeddings)...")
    for i, tgt_dir in enumerate(tgt_dirs):
        slice_name = f"ST-colon{i + 1}"
        print(f"Processing target slice {slice_name}...")

        tgt_adata = tgt_raw_list[i][:, common_genes].copy()
        tgt_position = tgt_adata.obsm['spatial']
        tgt_image_path = os.path.join(tgt_dir, 'tissue_hires_image.png')

        # Extract UNI2h Features
        print(f"UNI2h processing {tgt_image_path}...")
        tgt_image_feats = extract_UNI2h_features_from_spots_for_target(
            model_UNI2h, tgt_image_path, tgt_position, crop_size=20,
        )
        tgt_image_feats = row_min_max_normalize(tgt_image_feats)

        # Extract ResNet Features
        tgt_resnet_feats = extract_resnet_features_from_spots_for_target(
            tgt_image_path, tgt_position, crop_size=20, device=device
        )
        tgt_resnet_feats = row_min_max_normalize(tgt_resnet_feats)

        # Generate Gene Embeddings
        print(f"OmiCLIP processing {slice_name} genes...")
        tgt_gene_df = generate_gene_df(
            ad=tgt_adata, house_keeping_genes=empty_hk, num_genes=300, todense=True
        )
        tgt_gene_texts = tgt_gene_df["label"].tolist()
        tgt_gene_embeddings = custom_encode_texts(
            model=OmiCLIP, tokenizer=tokenizer, texts=tgt_gene_texts,
            max_seq_len=77, device=device, batch_size=64
        )
        tgt_gene_embeddings = row_min_max_normalize(tgt_gene_embeddings)

        # Store
        tgt_all_img.append(tgt_image_feats)
        tgt_all_img_resnet.append(tgt_resnet_feats)
        tgt_all_pos.append(tgt_position)
        tgt_all_geneEmb.append(tgt_gene_embeddings)
        tgt_all_expr_raw.append(tgt_adata.copy())
        print(f'Target slice {slice_name} processed.')

    # Merge Target Data
    print("Merging target slices...")
    tgt_merged_img = np.vstack(tgt_all_img)
    tgt_merged_img_resnet = np.vstack(tgt_all_img_resnet)
    tgt_merged_pos = np.vstack(tgt_all_pos)
    tgt_merged_geneEmb = np.vstack(tgt_all_geneEmb)
    tgt_merged_expr_raw = tgt_all_expr_raw[0].concatenate(tgt_all_expr_raw[1:])

    # -------------------------- Step 5: Final HVG Selection (768 & 1536) --------------------------
    print("Step 5: Selecting 768 and 1536 HVGs from common genes...")

    # Use first reference sample to determine final HVGs
    ref_expr_filtered = [r.copy() for r in ref_expr_raw]
    tgt_expr_filtered = tgt_merged_expr_raw.copy()

    # 1. Select 768 HVGs
    sc.pp.highly_variable_genes(ref_expr_filtered[0], flavor="seurat_v3", n_top_genes=768, subset=True)
    hvg_768 = ref_expr_filtered[0].var_names

    ref_expr_hvg768 = [d[:, hvg_768] for d in ref_expr_filtered]
    tgt_expr_hvg768 = tgt_expr_filtered[:, hvg_768]

    # Convert to dense & Normalize
    ref_expr_processed_768 = [
        row_min_max_normalize(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X)
        for adata in ref_expr_hvg768
    ]
    tgt_expr_processed_768 = row_min_max_normalize(
        tgt_expr_hvg768.X.toarray() if scipy.sparse.issparse(tgt_expr_hvg768.X) else tgt_expr_hvg768.X
    )

    # 2. Select 1536 HVGs
    ref_expr_filtered_1536 = [r.copy() for r in ref_expr_raw]
    sc.pp.highly_variable_genes(ref_expr_filtered_1536[0], flavor="seurat_v3", n_top_genes=1536, subset=True)
    hvg_1536 = ref_expr_filtered_1536[0].var_names

    ref_expr_hvg1536 = [d[:, hvg_1536] for d in ref_expr_filtered_1536]
    tgt_expr_hvg1536 = tgt_expr_filtered[:, hvg_1536]

    # Convert to dense & Normalize
    ref_expr_processed_1536 = [
        row_min_max_normalize(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X)
        for adata in ref_expr_hvg1536
    ]
    tgt_expr_processed_1536 = row_min_max_normalize(
        tgt_expr_hvg1536.X.toarray() if scipy.sparse.issparse(tgt_expr_hvg1536.X) else tgt_expr_hvg1536.X
    )

    # -------------------------- Step 6: Generate & Save Features --------------------------
    print("Step 6: Generating final feature sets...")

    # Concatenate Gene Features (Embedding + 768 HVG Expression)
    ref_gene_feats = [
        np.hstack((ref_geneEmb[i], ref_expr_processed_768[i]))
        for i in range(len(ref_geneEmb))
    ]
    tgt_gene_feats = np.hstack((tgt_merged_geneEmb, tgt_expr_processed_768))

    print("Saving results to disk...")
    # Save Reference Dataset
    np.savez(
        f"../dataset/train_dataset_{tgt_name}.npz",
        img_feats=np.array(ref_img, dtype=object),
        img_resnet=np.array(ref_img_resnet, dtype=object),
        gene_feats=np.array(ref_gene_feats, dtype=object),
        gene_expr=np.array(ref_expr_processed_1536, dtype=object),
        positions=np.array(ref_pos, dtype=object)
    )

    # Save Target Dataset
    np.savez(
        f"../dataset/test_dataset_{tgt_name}.npz",
        img_feats=np.array(tgt_merged_img, dtype=object),
        img_resnet=np.array(tgt_merged_img_resnet, dtype=object),
        gene_feats=np.array(tgt_gene_feats, dtype=object),
        gene_expr=np.array(tgt_expr_processed_1536, dtype=object),
        positions=np.array(tgt_merged_pos, dtype=object)
    )

    print("All processing done!")


if __name__ == "__main__":
    ref_dir = "../data/CRC/"
    ref_name = ['SN048_1', 'SN048_2', 'SN048_3', 'SN048_4', 'SN84_1', 'SN84_2',
                'SN123_1', 'SN123_2', 'SN123_4', 'SN124_1', 'SN124_2', 'SN124_4']

    tgt_dirs = [
        "../data/CRC_new/ST-colon1/",
        "../data/CRC_new/ST-colon2/",
        "../data/CRC_new/ST-colon3/",
        "../data/CRC_new/ST-colon4/"
    ]
    tgt_name = 'CRC_new'

    read(ref_dir, tgt_dirs, ref_name, tgt_name)