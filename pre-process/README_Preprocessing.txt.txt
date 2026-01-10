# Data Preprocessing and Coordinate Alignment Notes

This document provides important details regarding the preprocessing scripts, specifically for the Colorectal Cancer (CRC) and ST_colon datasets.

## 1. Coordinate Transformation
In the preprocessing code for CRC and ST_colon datasets (e.g., `process_CRC_new.py`), you will notice specific coordinate transformation logic (scaling and offset adjustments). 

**Reason:** In the original raw data, the spatial coordinates of the spots do not perfectly align with the high-resolution tissue images. To ensure accurate feature extraction, we manually calibrated the coordinates to match the image pixels.

**Usage Note:** These transformations are flexible and highly dependent on the exact resolution and cropping of the histology images used.

## 2. Image Sources and Consistency

To ensure reproducibility, please use the specific image sources listed below. The coordinate transformation parameters in our code are calibrated **specifically** for these images.

### CRC Dataset
- **Image Source:** The images used in our code are converted directly from the image information embedded in the original `adata` (AnnData) files to PNG format.
- **Alignment:** If you extract images from the `adata` objects as we did, the provided coordinate transformations in the code will work directly.

### ST_colon Dataset (Target Dataset)
- **Image Source:** We utilized the `tissue_hires_image` provided by the MCGAE repository.
- **Link:** https://github.com/yiwen-yang/MCGAE/tree/main/colorectal%20cancer%20liver%20annotation
- **Alignment:** - If you download and use the images from the link above, you can use the coordinate transformation parameters exactly as they appear in our code.
    - **Important:** If you use images from a different source or with different resolutions/cropping, the provided transformations will likely misalign. In that case, you must manually adjust the scaling factors and offsets in the code to match your specific images.