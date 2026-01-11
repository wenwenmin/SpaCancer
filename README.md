# SpaCancer:Large Foundation Model-Enhanced Histology–Spatial Transcriptomics Fusion for Cancer Region Identification

![Over](Fig1_overview.jpg)

## Project Overview
SpaCancer is a multimodal transfer learning framework designed for precise Cancer Tissue Region (CTR) detection, addressing the core need for accurate CTR identification in clinical oncology. By innovatively fusing histology images with Spatial Transcriptomics (ST) data, leveraging the strong feature extraction capability of Foundation Models (FMs) and the generalization advantage of transfer learning, SpaCancer effectively overcomes limitations of traditional methods—such as subjectivity, single-modal data constraints, ST data noise, and cross-batch/platform heterogeneity—providing reliable technical support for tumor heterogeneity analysis and clinical treatment.



## Requirements
All experiments were conducted on an NVIDIA RTX 3090 GPU. Before running SpaCancer, you need to create a conda environment and install the required packages:
```shell
conda create -n SpaCancer python==3.9.23
conda activate SpaCancer
pip install -r requirements.txt
```


## Core Features
1.Precise CTR Detection: Integrates morphological and gene expression multimodal information to achieve accurate segmentation of cancerous regions in breast cancer, colorectal cancer, and other malignancies.

2.Cross-Scenario Generalization: Supports CTR detection across batches and platforms, effectively mitigating performance loss caused by technical differences and batch effects.

3.Tumor Boundary & Risk Identification: Distinguishes tumor regions, boundary regions, and healthy regions, and predicts high-risk pathological regions with cancerous potential that are undetectable by experts.

4.Stability & Reliability: Outperforms 8 state-of-the-art (SOTA) methods in multi-dataset validation, demonstrating robustness for clinical research and applications.

## Datasets
The ST human HER2-positive breast tumor datasets (STHBC) are available in: [https://github.com/almaan/her2st](https://github.com/almaan/her2st).

The 10X Visium Human Breast Cancer Breast Cancer: Ductal Carcinoma In Situ, Lobular carcinoma In Situ, Invasive Carcinoma dataset (ViHBC) are available in: [https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0).

The Xenium Human Breast Cancer dataset (XeHBC) can be found at: [https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast).

The 10X Visium Human Breast Cancer: Ductal Carcinoma In Situ, Invasive Carcinoma dataset (DuCIS) are available in: [https://www.10xgenomics.com/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0](https://www.10xgenomics.com/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0).

The 10X Visium Colorectal Cancer Visium datasets (CRC) are available in: [https://zenodo.org/records/7760264](https://zenodo.org/records/7760264).

The 10X Visium Colorectal Cancer Visium datasets (ST_colon) are available in:[https://www.biosino.org/node/project/detail/OEP001756](https://www.biosino.org/node/project/detail/OEP001756).


## Pre-trained mode UNI-2h and OmiCLIP
Given the outstanding performance of large pre-trained general-purpose foundation models in clinical tasks, we use UNI-2h and OmiCLIP as the backbone feature extractor. Before using SpaCancer, you need to apply to the two model for permission to access the model weights: [https://huggingface.co/MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) and [https://huggingface.co/WangGuangyuLab/Loki](https://huggingface.co/WangGuangyuLab/Loki) .
Please follow these steps to integrate UNI and OmiCLIP into the pipeline:
- Log in to Hugging Face and obtain your **access token**.
- Paste your token into code in pre-process:
   ```shell
   login('The login you need to apply for')  # Replace with your actual token
   ```

🔒 Note: Do not share your token publicly. It is tied to your Hugging Face account and grants access to restricted models.

## Contact details
If you have any questions, please contact dunameng0607@163.com.
