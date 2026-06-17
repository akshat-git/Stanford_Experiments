import json, pathlib, re, nibabel as nib, torch
import pandas as pd, torch.nn.functional as F
from torch.utils.data import Dataset

IMG_SIZE  = 128
CT_DIR    = pathlib.Path("INSPECT/sample/CTPA")
LAB_TSV   = next(pathlib.Path("INSPECT/sample").glob("labels_20250611.tsv"))
IMP_TSV   = next(pathlib.Path("INSPECT/sample").glob("impressions_20250611.tsv"))

# ---------- load labels & impressions ---------------------------------
labels_df = pd.read_csv(LAB_TSV, sep="\t").set_index("impression_id")
imp_df = pd.read_csv(IMP_TSV, sep="\t").set_index("impression_id")

# regex patterns
RE_THROMB = re.compile(r"\bthrombolys", re.I)
RE_ANTICO = re.compile(r"\banticoag",   re.I)

class InspectDataset(Dataset):
    def __init__(self, split="train"):
        meta_raw = json.load(open("sample_index.json"))
        self.meta = []
        
        # Check if 'impressions' column exists once
        if 'impressions' not in imp_df.columns:
            raise ValueError(f"Column 'impressions' not found. Available columns: {imp_df.columns.tolist()}")
        
        # Debug counters
        total_samples = 0
        split_samples = 0
        found_impressions = 0
        thromb_count = 0
        antico_count = 0
        
        for m in meta_raw:
            total_samples += 1
            if m["split"] != split:
                continue
            split_samples += 1
            
            impression_id = m["impression_id"]
            
            # Check if impression_id exists in the dataframe
            if impression_id not in imp_df.index:
                continue
            
            txt = imp_df.loc[impression_id, 'impressions']
            
            # Ensure txt is a string
            if pd.isna(txt):
                continue
            txt = str(txt)
            found_impressions += 1
            
            if   RE_THROMB.search(txt):
                thromb_count += 1
                m["treat_vec"] = torch.tensor([0., 1.])   # thrombolysis
            elif RE_ANTICO.search(txt):
                antico_count += 1
                m["treat_vec"] = torch.tensor([1., 0.])   # anticoagulation
            else:
                continue  # skip if neither keyword present
            self.meta.append(m)
        
        print(f"Dataset '{split}' statistics:")
        print(f"  Total samples in JSON: {total_samples}")
        print(f"  Samples for split '{split}': {split_samples}")
        print(f"  Found impressions: {found_impressions}")
        print(f"  Thrombolysis samples: {thromb_count}")
        print(f"  Anticoagulation samples: {antico_count}")
        print(f"  Final dataset size: {len(self.meta)}")
        
        if len(self.meta) == 0:
            print("No samples found! Checking first few impression texts...")
            for m in meta_raw[:5]:  # Check first 5 samples
                if m["split"] == split and m["impression_id"] in imp_df.index:
                    txt = str(imp_df.loc[m["impression_id"], 'impressions'])
                    print(f"Sample text: {txt[:200]}...")  # First 200 chars

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        m       = self.meta[idx]
        img_path = CT_DIR / f"{m['image_id']}.nii.gz"

        # ----- load mid‑slice + resize -----------------------------------
        vol   = nib.load(img_path).get_fdata()
        slice2d = vol[:, :, vol.shape[2] // 2]
        img   = torch.from_numpy(slice2d).float().unsqueeze(0).unsqueeze(0)
        img   = F.interpolate(img, size=(IMG_SIZE, IMG_SIZE),
                              mode="bilinear", align_corners=False).squeeze(0)
        img   = img / img.max().clamp_min(1e-6)

        # ----- treatment vector (2) -------------------------------------
        treat = m["treat_vec"]                          # tensor([anticoag , thrombolysis])

        # ----- outcome labels (8) ---------------------------------------
        row = labels_df.loc[m["impression_id"]]
        lbl = torch.tensor([
            row["1_month_mortality"],
            row["6_month_mortality"],
            row["12_month_mortality"],
            row["1_month_readmission"],
            row["6_month_readmission"],
            row["12_month_readmission"],
            row["12_month_PH"],
            row["pe_acute"],
        ], dtype=torch.float32)

        return img, treat, lbl  # (image, condition, outcome labels)
