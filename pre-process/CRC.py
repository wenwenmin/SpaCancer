import scanpy as sc
import numpy as np
import scipy.sparse
import torch
from torch.utils.data import Dataset
import timm
from timm.data import create_transform, resolve_data_config
from PIL import Image, ImageDraw
from huggingface_hub import login
import os
import cv2
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
import pandas as pd
import warnings

# Configuration
warnings.filterwarnings("ignore")
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()


def row_min_max_normalize(features: np.ndarray) -> np.ndarray:
    """
    Perform row-wise min-max normalization to scale values to [0, 1].
    :param features: Input feature matrix (N, D)
    :return: Normalized feature matrix
    """
    row_min = features.min(axis=1, keepdims=True)
    row_max = features.max(axis=1, keepdims=True)
    normalized = (features - row_min) / (row_max - row_min + 1e-8)  # Prevent division by zero
    return normalized


def get_coordinate_transform_func(slice_name):
    """
    Returns the specific coordinate transformation function for a given slice.
    This handles alignment differences between spatial coordinates and image pixels.
    """
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


def extract_resnet_features_from_spots(image_path, positions, slice_name=' ', crop_size=32, device=None):
    """Extract ResNet-50 features from image patches around each spot."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    positions = [tuple(p) for p in positions]
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    H, W, _ = image_np.shape

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
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


def extract_UNI2h_features_from_spots(model, image_path, positions, slice_name, crop_size=32, device=None):
    """Extract UNI2-h features from image patches with coordinate alignment."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    H, W, _ = image_np.shape

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
    """Encode gene text labels using OmiCLIP."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            truncated_texts = []

            # Manually truncate texts to fit model limits
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


def read(ref_dir, ref_name, target_name=None, n_genes=768, preprocess=False):
    # Combine reference and target datasets (if any)
    all_samples = ref_name.copy()
    if target_name is not None:
        all_samples.extend(target_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------- Step 1: Load Models --------------------------
    # NOTE: Replace with your actual HuggingFace token
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

    # -------------------------- Step 2: Data Loading & HVG Selection --------------------------
    print("Step 2: Loading raw data and selecting top 80% HVGs per sample...")
    adata_list = []
    hvg_sets = []

    for sample in all_samples:
        adata = sc.read_visium(os.path.join(ref_dir, sample))
        adata.var_names_make_unique()

        # Filter spots outside of tissue
        mask = adata.obs['in_tissue'] == 1
        adata._inplace_subset_obs(mask)

        # Normalize and find HVGs
        temp_adata = adata.copy()
        sc.pp.normalize_total(temp_adata, target_sum=1e4)
        sc.pp.log1p(temp_adata)

        n_total_genes = temp_adata.n_vars
        n_hvg_80p = int(0.8 * n_total_genes)
        sc.pp.highly_variable_genes(temp_adata, flavor="seurat_v3", n_top_genes=n_hvg_80p, inplace=True)

        hvg_genes = temp_adata.var[temp_adata.var['highly_variable']].index.tolist()
        hvg_sets.append(set(hvg_genes))
        adata_list.append(adata)  # Store raw filtered data

    # Calculate intersection of HVGs across all samples
    common_genes = hvg_sets[0]
    for hvg_set in hvg_sets[1:]:
        common_genes &= hvg_set
    common_genes = list(common_genes)
    print(f"Number of overlapping HVGs across all samples: {len(common_genes)}")

    # -------------------------- Step 3: Feature Extraction --------------------------
    ref, ref_img, ref_resnet, ref_pos, ref_geneEmb, ref_expr_raw = [], [], [], [], [], []
    empty_hk = pd.DataFrame(columns=["genesymbol"])

    for idx, r in enumerate(ref_name):
        print(f"\nProcessing sample: {r}")
        adata = adata_list[idx]  # Use the pre-loaded data

        # Keep only overlapping genes
        adata = adata[:, common_genes].copy()
        print(f"Genes after filtering: {adata.n_vars}")
        position = adata.obsm['spatial']

        # Extract UNI2h Image Features
        image_path = os.path.join("../data/CRC/slices_image/", f"{r}.png")
        img_feat = extract_UNI2h_features_from_spots(model_UNI2h, image_path, position, slice_name=r, crop_size=32)
        ref_img.append(row_min_max_normalize(img_feat))

        # Extract ResNet Image Features
        resnet_feat = extract_resnet_features_from_spots(image_path, position, slice_name=r, crop_size=32,
                                                         device=device)
        ref_resnet.append(row_min_max_normalize(resnet_feat))

        # Generate Gene Embeddings (OmiCLIP)
        print(f"Generating OmiCLIP embeddings for {r}...")
        gene_df = generate_gene_df(
            ad=adata,
            house_keeping_genes=empty_hk,
            num_genes=300,
            todense=True
        )
        gene_texts = gene_df["label"].tolist()
        gene_embeddings = custom_encode_texts(
            model=OmiCLIP,
            tokenizer=tokenizer,
            texts=gene_texts,
            max_seq_len=77,
            device=device,
            batch_size=64
        )
        ref_geneEmb.append(row_min_max_normalize(gene_embeddings))

        # Save processed expression for final HVG selection
        ref_expr_raw.append(adata.copy())
        ref.append(adata)
        ref_pos.append(position)
        print(f"Sample {r} processed.")

    # -------------------------- Step 4: Final HVG Filtering (768 & 1536) --------------------------
    print("\nStep 4: Selecting final 768 and 1536 HVGs from overlapping genes...")

    # Use the first sample as reference for final HVG selection
    ref_expr_filtered = [r.copy() for r in ref_expr_raw]
    sc.pp.highly_variable_genes(ref_expr_filtered[0], flavor="seurat_v3", n_top_genes=768, subset=True)

    # Apply selection to all samples
    target_genes_768 = ref_expr_filtered[0].var_names
    ref_expr_processed = [
        row_min_max_normalize(
            d[:, target_genes_768].X.toarray() if scipy.sparse.issparse(d.X) else d[:, target_genes_768].X)
        for d in ref_expr_raw
    ]

    # Repeat for 1536 HVGs
    ref_expr_filtered_1536 = [r.copy() for r in ref_expr_raw]
    sc.pp.highly_variable_genes(ref_expr_filtered_1536[0], flavor="seurat_v3", n_top_genes=1536, subset=True)

    target_genes_1536 = ref_expr_filtered_1536[0].var_names
    ref_expr_processed_1536 = [
        row_min_max_normalize(
            d[:, target_genes_1536].X.toarray() if scipy.sparse.issparse(d.X) else d[:, target_genes_1536].X)
        for d in ref_expr_raw
    ]

    # -------------------------- Step 5: Save Results --------------------------
    # Concatenate: Gene Embeddings (768 dim) + Expression (768 dim)
    combined_gene_features = [
        np.concatenate([emb, expr], axis=1)
        for emb, expr in zip(ref_geneEmb, ref_expr_processed)
    ]

    output_path = "../dataset/CRC_dataset.npz"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(
        output_path,
        img_feats=np.array(ref_img, dtype=object),
        img_resnet=np.array(ref_resnet, dtype=object),
        gene_feats=np.array(combined_gene_features, dtype=object),
        coords=np.array(ref_pos, dtype=object),
        gene_expr=np.array(ref_expr_processed_1536, dtype=object)
    )

    print(f"Processing complete! Data saved to {output_path}")


if __name__ == "__main__":
    ref_dir = "../data/CRC/"
    # List of reference samples
    ref_name = [
        'SN048_1', 'SN048_2', 'SN048_3', 'SN048_4',
        'SN84_1', 'SN84_2',
        'SN123_1', 'SN123_2', 'SN123_3', 'SN123_4',
        'SN124_1', 'SN124_2', 'SN124_3', 'SN124_4'
    ]
    # Optional target samples
    target_name = []

    read(ref_dir, ref_name, target_name=target_name)