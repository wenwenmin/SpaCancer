# SpaCancer:Large Foundation Model-Enhanced Histology–Spatial Transcriptomics Fusion for Cancer Region Segmentation

![Over](Fig1_overview.jpg)

## Project Overview
SpaCancer is a multimodal transfer learning framework designed for precise Cancer Tissue Region (CTR) detection, addressing the core need for accurate CTR identification in clinical oncology. By innovatively fusing histology images with Spatial Transcriptomics (ST) data, leveraging the strong feature extraction capability of Foundation Models (FMs) and the generalization advantage of transfer learning, SpaCancer effectively overcomes limitations of traditional methods—such as subjectivity, single-modal data constraints, ST data noise, and cross-batch/platform heterogeneity—providing reliable technical support for tumor heterogeneity analysis and clinical treatment.

## Core Features
1.Precise CTR Detection: Integrates morphological and gene expression multimodal information to achieve accurate segmentation of cancerous regions in breast cancer, colorectal cancer, and other malignancies.

2.Cross-Scenario Generalization: Supports CTR detection across batches and platforms, effectively mitigating performance loss caused by technical differences and batch effects.

3.Tumor Boundary & Risk Identification: Distinguishes tumor regions, boundary regions, and healthy regions, and predicts high-risk pathological regions with cancerous potential that are undetectable by experts.

4.Stability & Reliability: Outperforms 8 state-of-the-art (SOTA) methods in multi-dataset validation, demonstrating robustness for clinical research and applications.

## Datasets
Validation Datasets: 6 breast/colorectal cancer ST datasets, totaling 27 slices (3 multi-slice datasets: STHBC, CRC, ST colon; 3 single-slice datasets: 10XViHBC, XeHBC, DuCIS).

## Performance Advantages
Outperforms 8 SOTA methods (BLEEP, mclSTExp, SpaCell-plus, etc.) across three core metrics: AUC, AP, and F1-score, with an average AUC of 0.947 and F1-score of 0.831.
Maintains stable performance in cross-batch/platform experiments (e.g., XeHBC, DuCIS datasets), verifying strong generalization.
Ablation studies confirm: The UNI2h+OmiCLIP feature extractor combination, MSFA spatial aggregation module, and bidirectional cross-attention fusion all significantly contribute to performance.


## Data link
