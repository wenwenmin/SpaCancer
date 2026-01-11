import pandas as pd
import os
import re

base_path = "../data/CRC/label/3"
output_path = "./"
file_names = [
    "Pathologist_Annotations_SN048_A121573_Rep1.csv",
    "Pathologist_Annotations_SN048_A121573_Rep2.csv",
    "Pathologist_Annotations_SN048_A416371_Rep1.csv",
    "Pathologist_Annotations_SN048_A416371_Rep2.csv",
    "Pathologist_Annotations_SN84_A120838_Rep1.csv",
    "Pathologist_Annotations_SN84_A120838_Rep2.csv",
    "Pathologist_Annotations_SN123_A551763_Rep1.csv",
    "Pathologist_Annotations_SN123_A595688_Rep1.csv",
    "Pathologist_Annotations_SN123_A798015_Rep1.csv",
    "Pathologist_Annotations_SN123_A938797_Rep1_X.csv",
    "Pathologist_Annotations_SN124_A551763_Rep2.csv",
    "Pathologist_Annotations_SN124_A595688_Rep2.csv",
    "Pathologist_Annotations_SN124_A798015_Rep2.csv",
    "Pathologist_Annotations_SN124_A938797_Rep2.csv"
]

# Map original labels to integer classes
label_mapping = {
    # Tumor regions (2)
    "tumor": 2,
    "tumor&stroma_IC med to high": 2,
    "tumor&stroma": 2,
    "tumor&stroma IC med to high": 2,
    "tumor&stroma_IC low": 2,

    # Immune infiltrate regions (1)
    "stroma_fibroblastic_IC high": 1,
    "stroma_fibroblastic_IC_high": 1,
    "IC aggregate_submucosa": 1,
    "IC aggreagate_connective tissue": 1,
    "IC aggregate connective tissue": 1,
    "IC aggregregate_submucosa": 1,
    "IC aggregate_stroma or muscularis": 1,
    "IC aggregate submucosa": 1,
    "IC aggregate_muscularis or stroma": 1,
    "IC aggragate_stroma or muscularis": 1,
    "IC aggreates_stroma or muscularis": 1,

    # Normal regions (0)
    "epithelium&submucosa": 0,
    "stroma_fibroblastic_IC med": 0,
    "non neo epithelium": 0,
    "connective tissue_2_fibroblastic_IC low": 0,
    "connective tissue_3_fibroblastic_IC med": 0,
    "submucosa": 0,
    "connective tissue_1_edema": 0,
    "lamina propria": 0,
    "connective tissue_4_muscularis_IC low": 0,
    "stroma_desmoplastic_IC low": 0,
    "stroma_desmoplastic_IC med to high": 0,
    "connective tissue_6_hemosiderin?": 0,
    "stroma_fibroblastic_IC_med": 0,
    "stroma_fibroblastic_IC low": 0,
    "muscularis_IC med to high": 0,
    "stroma desmoplastic_IC low": 0,
    "glandular tissue": 0,
    "epitehlium&submucosa": 0,
    "stroma desmoplastic_IC med to high": 0,
    "squamous epithelium": 0,
    "epithelium&lam propria": 0,

    # Excluded (-1)
    "exclude": -1
}

os.makedirs(output_path, exist_ok=True)

missing_files = []
missing_columns = []
unmapped_labels = set()

for file_name in file_names:
    file_path = os.path.join(base_path, file_name)

    if not os.path.exists(file_path):
        missing_files.append(file_name)
        continue

    try:
        df = pd.read_csv(file_path)

        # Identify target column
        target_col = None
        if "Pathologist Annotations" in df.columns:
            target_col = "Pathologist Annotations"
        elif "Pathologist Annotation" in df.columns:
            target_col = "Pathologist Annotation"

        if not target_col:
            missing_columns.append(file_name)
            continue

        raw_labels = df[target_col].dropna().tolist()
        converted_labels = []

        for label in raw_labels:
            if label in label_mapping:
                converted_labels.append(str(label_mapping[label]))
            else:
                unmapped_labels.add(f"{file_name}: {label}")
                # Mark unmapped as -2
                converted_labels.append("-2")

        # Extract SN identifier for output filename
        match = re.search(r'SN.*?\.csv', file_name)
        if match:
            sn_part = match.group().replace('.csv', '')
            output_file_name = f"{sn_part}-label.txt"
        else:
            output_file_name = f"{os.path.splitext(file_name)[0]}-label.txt"

        output_file_path = os.path.join(output_path, output_file_name)

        with open(output_file_path, 'w') as f:
            f.write('\n'.join(converted_labels))

        print(f"Generated: {output_file_name} ({len(converted_labels)} labels)")

    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")

print("\n" + "=" * 60)
print("Processing Report")
print("=" * 60)

if missing_files:
    print(f"\nMissing files: {len(missing_files)}")
    for file in missing_files:
        print(f"  - {file}")

if missing_columns:
    print(f"\nFiles missing target column: {len(missing_columns)}")
    for file in missing_columns:
        print(f"  - {file}")

if unmapped_labels:
    print(f"\nUnmapped labels: {len(unmapped_labels)}")
    for label in unmapped_labels:
        print(f"  - {label}")

print("\nProcessing complete.")