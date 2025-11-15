import os
import random
import numpy as np
import SimpleITK as sitk
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

parent_directory = os.path.dirname(os.getcwd())
BASE_DIR = os.path.join(parent_directory, 'data','manifest-1694710246744','Prostate-MRI-US-Biopsy')


def load_correct_study(patient_path):
    """
    Find subfolder containing exactly 60 DICOM files.
    """
    for root, dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if len(dcm_files) == 60:
            return root
    return None

def count_slices(patient_folder):
    study_folder = load_correct_study(patient_folder)
    if study_folder is None:
        return 0
    dcm_files = [f for f in os.listdir(study_folder) if f.lower().endswith(".dcm")]
    return len(dcm_files)

def load_patient_volume(patient_folder):

    study_folder = load_correct_study(patient_folder)
    #print(study_folder)
    if study_folder is None:
        return None

    # Read all DICOM slices in the folder
    dcm_files = sorted([os.path.join(study_folder, f) 
                        for f in os.listdir(study_folder) 
                        if f.lower().endswith('.dcm')])
    if len(dcm_files) < 3:
        # Not enough slices for triplet selection
        return None

    slices = []
    for f in dcm_files:
        img = sitk.ReadImage(f)
        arr = sitk.GetArrayFromImage(img)[0]  # (1,H,W) -> (H,W)
        slices.append(arr.astype(np.float32))

    # Stack into a volume (Z,H,W)
    volume_np = np.stack(slices, axis=0)

    return volume_np

def generate_consecutive_triplets(volume):
    """
    Generate overlapping triplets:
    (slice[i], slice[i+2]) -> slice[i+1]
    (slice[i], slice[i+4]) -> slice[i+2]
    """
    pre, post, mid = [], [], []

    # Distance 2
    for i in range(volume.shape[0] - 2):
        pre_slice = (volume[i] - volume[i].mean()) / (volume[i].std()+1e-6)
        mid_slice = (volume[i+1] - volume[i+1].mean()) / (volume[i+1].std()+1e-6)
        post_slice = (volume[i+2] - volume[i+2].mean()) / (volume[i+2].std()+1e-6)
        pre_slice  = torch.tensor(pre_slice).unsqueeze(0)
        mid_slice  = torch.tensor(mid_slice).unsqueeze(0)
        post_slice = torch.tensor(post_slice).unsqueeze(0)
        pre.append(pre_slice)
        post.append(post_slice)
        mid.append(mid_slice)

    # Distance 4
    for i in range(volume.shape[0] - 4):
        pre_slice = (volume[i] - volume[i].mean()) / (volume[i].std()+1e-6)
        mid_slice = (volume[i+1] - volume[i+2].mean()) / (volume[i+2].std()+1e-6)
        post_slice = (volume[i+4] - volume[i+4].mean()) / (volume[i+4].std()+1e-6)
        pre_slice  = torch.tensor(pre_slice).unsqueeze(0)
        mid_slice  = torch.tensor(mid_slice).unsqueeze(0)
        post_slice = torch.tensor(post_slice).unsqueeze(0)
        pre.append(pre_slice)
        post.append(post_slice)
        mid.append(mid_slice)

    return pre, post, mid

class PairedTransforms:
    def __call__(self, sample):
        pre = sample["pre"]
        post = sample["post"]
        mid = sample["target"]

        # Horizontal flip
        if random.random() < 0.5:
            pre  = TF.hflip(pre)
            post = TF.hflip(post)
            mid  = TF.hflip(mid)

        # Vertical flip
        if random.random() < 0.5:
            pre  = TF.vflip(pre)
            post = TF.vflip(post)
            mid  = TF.vflip(mid)

        # Rotation (use bilinear!)
        angle = random.uniform(-5, 5)
        pre  = TF.rotate(pre, angle, interpolation=TF.InterpolationMode.BILINEAR)
        post = TF.rotate(post, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mid  = TF.rotate(mid, angle, interpolation=TF.InterpolationMode.BILINEAR)

        return {"pre": pre, "post": post, "target": mid}
    

class TripletSliceDataset(Dataset):
    def __init__(self, patient_folders, transform=None):
        self.transform = transform
        self.patient_folders = patient_folders
        self.triplet_indices = []
        for pid, folder in enumerate(patient_folders):
            n_slices = count_slices(folder)
            if n_slices < 3:
                continue
            n_triplets = (n_slices - 2) + (n_slices - 4)
            for t in range(n_triplets):
                self.triplet_indices.append((pid, t))

    def __len__(self):
        return len(self.triplet_indices)

    def __getitem__(self, idx):
        patient_idx, triplet_idx = self.triplet_indices[idx]
        folder = self.patient_folders[patient_idx]
        vol = load_patient_volume(folder)  # only load when needed
        pre_list, post_list, mid_list = generate_consecutive_triplets(vol)

        pre = pre_list[triplet_idx]
        post = post_list[triplet_idx]
        mid = mid_list[triplet_idx]

        sample = {"pre": pre, "post": post, "target": mid}
        if self.transform:
            sample = self.transform(sample)

        return (sample["pre"], sample["post"]), sample["target"]


def build_dataloader(split="train",
                     batch_size=4,
                     augment=False,
                     num_workers=4):
    """
    base_dir: path to Prostate-MRI-US-Biopsy dataset folder
    split: "train", "val", "test"
    batch_size: dataloader batch size
    augment: whether to apply PairedTransforms
    """

    # ---------------------------------------------------------
    # Split patient folders
    # ---------------------------------------------------------

    patient_folders = sorted([
        f for f in os.listdir(BASE_DIR)
        if f.startswith("Prostate-MRI-US-Biopsy-")
    ])

    train_folders, test_val_folders = train_test_split(
        patient_folders, test_size=0.3, random_state=42
    )

    val_folders, test_folders = train_test_split(
        test_val_folders, test_size=0.6, random_state=42
    )

    if split == "train":
        folders = train_folders
    elif split == "val":
        folders = val_folders
    else:
        folders = test_folders

    # ---------------------------------------------------------
    # Read patient volumes + generate triplets
    # ---------------------------------------------------------
    patient_folders = []
    for pid in folders:
        patient_folders.append(os.path.join(BASE_DIR, pid))

    transform = PairedTransforms() if augment else None

    dataset = TripletSliceDataset(patient_folders, transform)

    # ---------------------------------------------------------
    # Build DataLoader
    # ---------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader

if __name__ == "__main__":
    BASE = "/path/to/data/manifest-1694710246744/Prostate-MRI-US-Biopsy"

    train_loader = build_dataloader(
        split="train",
        batch_size=4,
        augment=True
    )

    for batch in train_loader:
        (pre, post), mid = batch
        print(pre.shape, post.shape, mid.shape)
        break
