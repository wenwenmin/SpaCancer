import scanpy as sc
import numpy as np
import scipy.sparse
import torch
from matplotlib.patheffects import Normal
from torch.utils.data import Dataset, DataLoader
import timm
from timm.data import create_transform, resolve_data_config
from PIL import Image, ImageDraw
from huggingface_hub import login
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import pandas as pd
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 查看 PyTorch 版本
print(torch.__version__)
# 查看 PyTorch 是否能识别到 CUDA（即是否支持 GPU 加速）
print(torch.cuda.is_available())


def row_min_max_normalize(features: np.ndarray) -> np.ndarray:
    """
    对每一行进行 min-max 归一化，使其范围缩放到 [0, 1]
    :param features: shape = (N, 2048)
    :return: same shape, 每一行缩放到 [0,1]
    """
    row_min = features.min(axis=1, keepdims=True)
    row_max = features.max(axis=1, keepdims=True)
    normalized = (features - row_min) / (row_max - row_min + 1e-8)  # 防止除0
    return normalized


def extract_UNI2h_features_from_spots(model, image_path, positions, crop_size=250, device=None, slice_name=' '):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    H, W, _ = image_np.shape

    # 创建用于显示裁剪区域的图像副本
    show_crop_image = image.copy()
    draw = ImageDraw.Draw(show_crop_image)

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    features = []

    for (x, y) in tqdm(positions, desc="Extracting UNI2h features"):
        # 计算裁剪区域坐标
        left = int(x - crop_size // 2)
        upper = int(y - crop_size // 2)
        right = left + crop_size
        lower = upper + crop_size

        # 边界处理
        left = max(0, left)
        upper = max(0, upper)
        right = min(W, right)
        lower = min(H, lower)

        draw.rectangle([left, upper, right, lower], outline='blue', width=2)
        # 裁剪并处理图像
        crop = image.crop((left, upper, right, lower))
        crop = transform(crop).unsqueeze(0).to(device)

        # 提取特征
        with torch.no_grad():
            feat = model(crop)
            feat = feat.view(-1).cpu().numpy()
            features.append(feat)

    # # 保存显示裁剪区域的图像
    # os.makedirs("../data/HB/her2st/show_crop/", exist_ok=True)
    # show_crop_path = f"../data/HB/her2st/show_crop/{slice_name}_show_crop.png"
    # show_crop_image.save(show_crop_path)
    # print(f"Saved crop visualization to {show_crop_path}")

    return np.stack(features)  # 返回 shape: [N_spots, 2048]


def extract_UNI_features_from_spots(model, image_path, positions, crop_size=299, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    H, W, _ = image_np.shape

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    features = []

    for (x, y) in tqdm(positions, desc="Extracting UNI features"):
        # 计算裁剪区域坐标
        left = int(x - crop_size // 2)
        upper = int(y - crop_size // 2)
        right = left + crop_size
        lower = upper + crop_size

        # 边界处理
        left = max(0, left)
        upper = max(0, upper)
        right = min(W, right)
        lower = min(H, lower)

        # 裁剪并处理图像
        crop = image.crop((left, upper, right, lower))
        crop = transform(crop).unsqueeze(0).to(device)

        # 提取特征
        with torch.no_grad():
            feat = model(crop)
            feat = feat.view(-1).cpu().numpy()
            print("UNI2h featture dim:", feat.shape)
            features.append(feat)

    return np.stack(features)  # 返回 shape: [N_spots, 2048]


def extract_resnet_features_from_spots(image_path, positions, crop_size=299, device=None):
    """
    用ResNet-50提取每个spot周围区域的图像特征

    参数:
        image_path (str): 图像路径
        positions (np.ndarray or List[Tuple[int, int]]): 所有spot中心坐标 (x, y)，支持(n,2)的numpy数组
        crop_size (int): 每个patch裁剪大小（默认为299×299）
        device (torch.device or str): 指定设备，默认为自动选择

    返回:
        np.ndarray: shape = (N_spots, 2048)，每行是一个spot的特征
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 转为 List[Tuple[int, int]] 以支持 numpy 输入
    positions = [tuple(p) for p in positions]

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    H, W, _ = image_np.shape

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet输入大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载并构建ResNet-50特征提取器
    resnet50 = models.resnet50(pretrained=True)
    resnet_feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])  # 去掉fc层
    resnet_feature_extractor.to(device)
    resnet_feature_extractor.eval()

    features = []

    for (x, y) in tqdm(positions, desc="Extracting ResNet features"):
        # 计算裁剪区域坐标
        left = int(x - crop_size // 2)
        upper = int(y - crop_size // 2)
        right = left + crop_size
        lower = upper + crop_size

        # 边界处理
        left = max(0, left)
        upper = max(0, upper)
        right = min(W, right)
        lower = min(H, lower)

        # 裁剪并处理图像
        crop = image.crop((left, upper, right, lower))
        crop = transform(crop).unsqueeze(0).to(device)

        # 提取特征
        with torch.no_grad():
            feat = resnet_feature_extractor(crop)  # [1, 2048, 1, 1]
            feat = feat.view(-1).cpu().numpy()  # [2048]
            features.append(feat)

    return np.stack(features)  # 返回 shape: [N_spots, 2048]


def custom_encode_texts(model, tokenizer, texts, device, max_seq_len=77, batch_size=32):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # 1. 手动截断文本（适配模型最大长度）
            truncated_texts = []
            for text in batch_texts:
                words = text.split()
                if len(words) > max_seq_len - 2:  # 预留首尾标记位
                    words = words[:max_seq_len - 2]
                # print(len(words))  #每个spot取top len(words)个基因名作为序列进入tokenizer，源代码中长度为77，去除两个标记位，就是处理75个基因
                truncated_texts.append(" ".join(words))

            # print("(include HKg)gene tokens:", len(truncated_texts), truncated_texts)
            # 2. 调用tokenizer（返回的是Tensor而非字典）

            text_inputs = tokenizer(
                truncated_texts,  # 直接传递文本字符创长度
                # return_tensors="pt"
            )  # 此时text_inputs是一个Tensor（如input_ids）

            # 3. 核心修正：将Tensor直接移动到设备（无需遍历字典）
            text_inputs = text_inputs.to(device)

            # 4. 编码文本特征（模型接受Tensor作为输入）
            text_embeds = model.encode_text(text_inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # 归一化

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


def read(ref_dir, ref_name, overlap=None, preprocess=False):
    ref, ref_img, ref_resnet, ref_pos, ref_geneEmb, ref_expr_raw = [], [], [], [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------- 第一步：加载模型 --------------------------
    tokens = "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"
    login(tokens)
    print("huggingface load successful!")
    timm_kwargs = {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }
    model_UNI2h = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    print("hf-hub:MahmoodLab/UNI2-h loaded")
    model_UNI2h.eval()
    model_UNI2h.to(device)

    from loki.preprocess import generate_gene_df
    from loki.utils import load_model

    print("加载预训练OmiCLIP模型...")
    model_weight_path = "../OmiCLIP/checkpoint.pt"
    OmiCLIP, preprocess, tokenizer = load_model(model_path=model_weight_path, device=device)
    OmiCLIP = OmiCLIP.to(device)  # 确保模型在目标设备

    # -------------------------- 第二步：读取原始数据并筛选80%高变基因交集 --------------------------
    print("第一步：读取原始数据并筛选每个样本80%的高变基因...")
    Normal_adata_list = []  # 存储原始adata
    hvg_sets = []  # 存储每个样本的80%高变基因集合

    for r in ref_name:
        print(f"Processing {r} for initial HVG selection...")
        adata = sc.read(ref_dir + r + '/' + r + '.h5ad')
        # 预处理并筛选80%高变基因
        temp_adata = adata.copy()
        sc.pp.normalize_total(temp_adata, target_sum=1e4)
        sc.pp.log1p(temp_adata)
        Normal_adata_list.append(temp_adata)

        total_genes = temp_adata.shape[1]
        n_top = int(total_genes * 0.8)  # 80%的高变基因
        sc.pp.highly_variable_genes(temp_adata, flavor="seurat_v3", n_top_genes=n_top)


        hvg = temp_adata.var_names[temp_adata.var.highly_variable].tolist()
        hvg_sets.append(set(hvg))
        print(f"{r}: 原始基因数={total_genes}, 80%高变基因数={n_top}")

    # 计算所有样本高变基因的交集
    overlap_hvg = hvg_sets[0]
    for hvg_set in hvg_sets[1:]:
        overlap_hvg &= hvg_set
    overlap_hvg = list(overlap_hvg)
    print(f"\n所有样本共有的高变基因数: {len(overlap_hvg)}")

    # -------------------------- 第三步：基于交集基因进行后续处理 --------------------------
    empty_hk = pd.DataFrame(columns=["genesymbol"])  # 空管家基因列表

    for i, r in enumerate(ref_name):
        print(f"\nProcessing {r} with overlapping HVGs...")
        adata = Normal_adata_list[i]  # 读取原始adata

        # 只保留交集基因
        adata_filtered = adata[:, overlap_hvg].copy()
        print(f"{r}: 过滤后基因数={adata_filtered.shape[1]}")

        # 提取图像特征（UNI2h）
        position = adata_filtered.obsm['spatial']
        image_feats = extract_UNI2h_features_from_spots(
            model_UNI2h,
            ref_dir + r + '/' + 'he-raw.jpg',
            position,
            crop_size=256,
            slice_name=r,
            device=device
        )

        # 提取图像特征（ResNet）
        resnet_feats = extract_resnet_features_from_spots(
            image_path=ref_dir + r + '/' + 'he-raw.jpg',
            positions=position,
            crop_size=256,
            device=device
        )

        # OmiCLIP处理过滤后的基因（生成768维嵌入）
        print(f"OmiCLIP processing {r} genes (filtered)...")
        gene_df = generate_gene_df(
            ad=adata_filtered,  # 使用过滤后的adata
            house_keeping_genes=empty_hk,
            num_genes=300,
            todense=True
        )
        gene_texts = gene_df["label"].tolist()
        gene_embeddings = custom_encode_texts(
            model=OmiCLIP,
            tokenizer=tokenizer,
            texts=gene_texts,
            max_seq_len=77,#限制输入tokenizer的gene_texts中的基因数
            device=device,
            batch_size=32
        )

        gene_embeddings = row_min_max_normalize(gene_embeddings)  # 归一化

        # 保存处理后的数据
        ref.append(adata_filtered)
        ref_img.append(row_min_max_normalize(image_feats))  # UNI2h特征归一化
        ref_resnet.append(row_min_max_normalize(resnet_feats))  # ResNet特征归一化
        ref_pos.append(position)
        ref_geneEmb.append(gene_embeddings)
        ref_expr_raw.append(adata_filtered.copy())  # 保存过滤后的原始表达数据
        print(f'Section {r} processed: gene_emb shape={gene_embeddings.shape}, resnet shape={resnet_feats.shape}')

    # -------------------------- 第四步：在过滤后的基因中再次筛选HVG --------------------------
    print("\n第四步：在交集基因中筛选768个和1536个高变基因...")
    # 基于第一个样本筛选（确保所有样本使用相同的HVG集合）
    first_adata = ref_expr_raw[0].copy()
    # sc.pp.normalize_total(first_adata, target_sum=1e4)
    # sc.pp.log1p(first_adata)

    # 筛选768个HVG（用于拼接）
    sc.pp.highly_variable_genes(first_adata, flavor="seurat_v3", n_top_genes=768)
    hvg_768 = first_adata.var_names[first_adata.var.highly_variable].tolist()

    # 筛选1536个HVG（用于gene_expr）
    sc.pp.highly_variable_genes(first_adata, flavor="seurat_v3", n_top_genes=1536)
    hvg_1536 = first_adata.var_names[first_adata.var.highly_variable].tolist()

    print(f"筛选出768个HVG，1536个HVG")

    # -------------------------- 第五步：处理表达特征 --------------------------
    ref_expr_768 = []  # 768个HVG的表达特征
    ref_expr_1536 = []  # 1536个HVG的表达特征

    for adata in ref_expr_raw:
        # 处理768个HVG
        adata_768 = adata[:, hvg_768].copy()
        # sc.pp.normalize_total(adata_768, target_sum=1e4)
        # sc.pp.log1p(adata_768)
        expr_768 = adata_768.X.toarray() if scipy.sparse.issparse(adata_768.X) else adata_768.X
        ref_expr_768.append(row_min_max_normalize(expr_768))

        # 处理1536个HVG
        adata_1536 = adata[:, hvg_1536].copy()
        # sc.pp.normalize_total(adata_1536, target_sum=1e4)
        # sc.pp.log1p(adata_1536)
        expr_1536 = adata_1536.X.toarray() if scipy.sparse.issparse(adata_1536.X) else adata_1536.X
        ref_expr_1536.append(row_min_max_normalize(expr_1536))


    # -------------------------- 第六步：拼接特征并保存 --------------------------
    print("\n第六步：拼接基因嵌入和HVG表达特征...")
    combined_gene_features = [
        np.concatenate([emb, expr], axis=1)  # 768 + 768 = 1536维
        for emb, expr in zip(ref_geneEmb, ref_expr_768)
    ]

    # 保存结果
    np.savez_compressed(
        "../dataset/HER2ST_dataset(Final_77).npz",
        img_feats=np.array(ref_img, dtype=object),  # UNI2h提取的图像特征
        img_resnet=np.array(ref_resnet, dtype=object),  # ResNet提取的图像特征
        gene_feats=np.array(combined_gene_features, dtype=object),  # 1536维（768嵌入+768表达）
        coords=np.array(ref_pos, dtype=object),
        gene_expr=np.array(ref_expr_1536, dtype=object)  # 1536个HVG的表达
    )

    print("finish!!!")


ref_dir = "../data/HB/her2st/"
ref_name = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']

read(ref_dir, ref_name)