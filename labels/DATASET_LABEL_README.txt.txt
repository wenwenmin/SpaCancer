Dataset Label Description
=========================

This document clarifies the label values found in the processed datasets, specifically for CRC and STHBC (Spatial Transcriptomics Human Breast Cancer).

1. Original Label Definitions
-----------------------------
The label files in these datasets utilize four distinct integer values to represent different tissue regions, based on the original annotations:

* **2**: Confirmed Cancer Regions.
* **1**: Immune Infiltration Regions.
* **0**: Healthy/Normal Tissue.
* **-1**: Uncertain/Indeterminate Regions (areas marked as ambiguous in the original dataset).

2. Binary Classification Mapping (Paper Setting)
------------------------------------------------
Since our proposed framework focuses on a **binary classification task** (Cancer vs. Non-Cancer), we map the original four labels into two categories for training and evaluation:

* **Positive Class (Cancer)**:
    - Original Label: **2**

* **Negative Class (Non-Cancer)**:
    - Original Labels: **0, 1, and -1**
    (Note: Immune infiltration regions and uncertain regions are treated as non-cancerous in our main experiments).

3. Handling Uncertain Regions (Label -1)
----------------------------------------
There are two approaches to handling the uncertain regions (label -1):

* **Approach A (Paper Protocol - Fair Comparison):**
  In our published results, we treated uncertain regions (-1) as **Non-Cancer**. We did *not* ignore or mask these regions during evaluation. This decision was made to ensure a strictly fair comparison with existing baseline methods, which also included these regions in their evaluation protocols.

* **Approach B (Optional - Precision Optimization):**
  You may choose to ignore/mask regions with label -1 during loss calculation and metric evaluation. Excluding these ambiguous areas typically allows the model to achieve higher precision and more accurate metrics.

**Note to Developers:**
When reproducing our model, you may choose either approach depending on your experimental goals. The code provided defaults to the Paper Protocol (Approach A).