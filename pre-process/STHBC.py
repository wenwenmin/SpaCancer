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
import pandas as pd
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
import warnings

# Configure environment and suppress warnings
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
warnings.filterwarnings("ignore")

# Check environment
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")


def row_min_max_normalize(features: np.ndarray) -> np.ndarray:
    """
    Perform row-wise min-max normalization to scale values to [0, 1].
    :param features: shape = (N, D)
    :return: Normalized features with the same shape.
    """
    row_min = features.min(axis=1, keepdims=True)
    row_max = features.max(axis=1, keepdims=True)
    normalized = (features - row_min) / (row_max - row_min + 1e-8)
    return normalized


def extract_UNI2h_features_from_spots(model, image_path, positions, crop_size=250, device=None, slice_name=' '):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load image
    image = Image.open(image_path).convert('RGB')
    H, W, _ = np.array(image).shape

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    features = []

    for (x, y) in tqdm(positions, desc="Extracting UNI2h features"):
        # Calculate crop coordinates
        left = int(x - crop_size // 2)
        upper = int(y - crop_size // 2)
        right = left + crop_size
        lower = upper + crop_size

        # Boundary handling
        left = max(0, left)
        upper = max(0, upper)
        right = min(W, right)
        lower = min(H, lower)

        # Crop and transform
        crop = image.crop((left, upper, right, lower))
        crop = transform(crop).unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            feat = model(crop)
            feat = feat.view(-1).cpu().numpy()
            features.append(feat)

    return np.stack(features)  # shape: [N_spots, 2048]


def extract_resnet_features_from_spots(image_path, positions, crop_size=299, device=None):
    """
    Extract ResNet-50 features for image patches around each spot.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    positions = [tuple(p) for p in positions]

    # Load image
    image = Image.open(image_path).convert('RGB')
    H, W, _ = np.array(image).shape

    # Preprocessing for ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load ResNet-50
    resnet50 = models.resnet50(pretrained=True)
    resnet_feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
    resnet_feature_extractor.to(device)
    resnet_feature_extractor.eval()

    features = []

    for (x, y) in tqdm(positions, desc="Extracting ResNet features"):
        # Calculate crop coordinates
        left = int(x - crop_size // 2)
        upper = int(y - crop_size // 2)
        right = left + crop_size
        lower = upper + crop_size

        # Boundary handling
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


def custom_encode_texts(model, tokenizer, texts, device, max_seq_len=77, batch_size=32):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            truncated_texts = []

            # 1. Manual truncation to fit model max length
            for text in batch_texts:
                words = text.split()
                if len(words) > max_seq_len - 2:  # Reserve space for start/end tokens
                    words = words[:max_seq_len - 2]
                truncated_texts.append(" ".join(words))

            # 2. Tokenize and move to device
            text_inputs = tokenizer(truncated_texts)
            text_inputs = text_inputs.to(device)

            # 3. Encode
            text_embeds = model.encode_text(text_inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # Normalize
            embeddings.append(text_embeds.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


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


def read(ref_dir, ref_name):
    ref, ref_img, ref_resnet, ref_pos, ref_geneEmb, ref_expr_raw = [], [], [], [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------- Step 1: Load Models --------------------------
    # NOTE: Please replace with your own HuggingFace token
    hf_token = "YOUR_HF_TOKEN"
    login(hf_token)
    print("HuggingFace login successful!")

    # Load UNI2-h model
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 5.33334,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU,
        'reg_tokens': 8, 'dynamic_img_size': True
    }
    model_UNI2h = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model_UNI2h.eval().to(device)
    print("UNI2-h loaded.")

    # Load OmiCLIP model
    from loki.preprocess import generate_gene_df
    from loki.utils import load_model

    print("Loading OmiCLIP...")
    model_weight_path = "../../OmiCLIP/checkpoint.pt"
    OmiCLIP, preprocess, tokenizer = load_model(model_path=model_weight_path, device=device)
    OmiCLIP = OmiCLIP.to(device)

    # -------------------------- Step 2: Load Data & Select HVGs --------------------------
    print("Step 2: Loading raw data and selecting top 80% HVGs per sample...")
    Normal_adata_list = []
    hvg_sets = []

    for r in ref_name:
        print(f"Processing {r}...")
        adata = sc.read(os.path.join(ref_dir, r, f'{r}.h5ad'))

        temp_adata = adata.copy()
        sc.pp.normalize_total(temp_adata, target_sum=1e4)
        sc.pp.log1p(temp_adata)
        Normal_adata_list.append(temp_adata)

        total_genes = temp_adata.shape[1]
        n_top = int(total_genes * 0.9)
        sc.pp.highly_variable_genes(temp_adata, flavor="seurat_v3", n_top_genes=n_top)

        hvg = temp_adata.var_names[temp_adata.var.highly_variable].tolist()
        hvg_sets.append(set(hvg))

    # Calculate overlapping HVGs across all samples
    overlap_hvg = hvg_sets[0]
    for hvg_set in hvg_sets[1:]:
        overlap_hvg &= hvg_set
    overlap_hvg = list(overlap_hvg)
    print(f"Number of overlapping HVGs: {len(overlap_hvg)}")

    # -------------------------- Step 3: Process Overlapping Genes --------------------------
    empty_hk = pd.DataFrame(columns=["genesymbol"])

    for i, r in enumerate(ref_name):
        print(f"\nProcessing {r} with overlapping HVGs...")
        adata = Normal_adata_list[i]

        # Keep only overlapping genes
        adata_filtered = adata[:, overlap_hvg].copy()

        # Extract Image Features (UNI2h)
        position = adata_filtered.obsm['spatial']
        image_feats = extract_UNI2h_features_from_spots(
            model_UNI2h,
            os.path.join(ref_dir, r, 'he-raw.jpg'),
            position,
            crop_size=256,
            slice_name=r,
            device=device
        )

        # Extract Image Features (ResNet)
        resnet_feats = extract_resnet_features_from_spots(
            image_path=os.path.join(ref_dir, r, 'he-raw.jpg'),
            positions=position,
            crop_size=256,
            device=device
        )

        # Generate Gene Embeddings via OmiCLIP
        gene_df = generate_gene_df(
            ad=adata_filtered,
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
            device=device
        )
        gene_embeddings = row_min_max_normalize(gene_embeddings)

        # Append processed data
        ref.append(adata_filtered)
        ref_img.append(row_min_max_normalize(image_feats))
        ref_resnet.append(row_min_max_normalize(resnet_feats))
        ref_pos.append(position)
        ref_geneEmb.append(gene_embeddings)
        ref_expr_raw.append(adata_filtered.copy())

    # -------------------------- Step 4: Final HVG Selection --------------------------
    print("\nStep 4: Selecting final 768 and 1536 HVGs from overlapping genes...")
    first_adata = ref_expr_raw[0].copy()

    # Select 768 HVGs (for concatenation)
    sc.pp.highly_variable_genes(first_adata, flavor="seurat_v3", n_top_genes=768)
    hvg_768 = first_adata.var_names[first_adata.var.highly_variable].tolist()

    # Select 1536 HVGs (for expression target)
    sc.pp.highly_variable_genes(first_adata, flavor="seurat_v3", n_top_genes=1536)
    hvg_1536 = first_adata.var_names[first_adata.var.highly_variable].tolist()

    # -------------------------- Step 5: Process Expression Features --------------------------
    ref_expr_768 = []
    ref_expr_1536 = []

    for adata in ref_expr_raw:
        # Process 768 HVGs
        adata_768 = adata[:, hvg_768].copy()
        expr_768 = adata_768.X.toarray() if scipy.sparse.issparse(adata_768.X) else adata_768.X
        ref_expr_768.append(row_min_max_normalize(expr_768))

        # Process 1536 HVGs
        adata_1536 = adata[:, hvg_1536].copy()
        expr_1536 = adata_1536.X.toarray() if scipy.sparse.issparse(adata_1536.X) else adata_1536.X
        ref_expr_1536.append(row_min_max_normalize(expr_1536))

    # -------------------------- Step 6: Concatenate & Save --------------------------
    print("\nStep 6: Concatenating gene embeddings and HVG expression...")
    combined_gene_features = [
        np.concatenate([emb, expr], axis=1)  # 768 + 768 = 1536 dim
        for emb, expr in zip(ref_geneEmb, ref_expr_768)
    ]

    output_path = "../dataset/HER2ST_dataset.npz"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(
        output_path,
        img_feats=np.array(ref_img, dtype=object),
        img_resnet=np.array(ref_resnet, dtype=object),
        gene_feats=np.array(combined_gene_features, dtype=object),
        coords=np.array(ref_pos, dtype=object),
        gene_expr=np.array(ref_expr_1536, dtype=object)
    )

    print(f"Processing complete. Data saved to {output_path}")


if __name__ == "__main__":
    ref_dir = "../../data/HB/her2st/"
    ref_name = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']
    read(ref_dir, ref_name)