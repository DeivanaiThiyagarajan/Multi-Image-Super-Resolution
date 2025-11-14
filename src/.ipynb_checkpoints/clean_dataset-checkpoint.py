"""
Clean Dataset: Remove Ultrasound 3D Volumes and Keep Only MRI Data
Single script to remove unwanted data types from all patient folders
"""

import os
import shutil
from pathlib import Path
import pydicom
from tqdm import tqdm


def is_ultrasound_series(series_dir):
    """
    Check if series is ultrasound 3D (to be removed)
    Returns True if series should be deleted
    """
    try:
        dicom_files = list(Path(series_dir).glob("*.dcm"))
    except (OSError, PermissionError):
        return False
    
    if not dicom_files:
        return False
    
    try:
        dcm = pydicom.dcmread(str(dicom_files[0]))
        modality = getattr(dcm, "Modality", "").upper()
        
        # Remove Ultrasound data
        if modality == "US":
            return True
        
        # Remove 3D Rendering data (these are usually marked as US or have specific characteristics)
        series_desc = getattr(dcm, "SeriesDescription", "").upper()
        if "3D" in series_desc and "RENDERING" in series_desc:
            return True
        
        return False
    except Exception as e:
        return False


def scan_dataset(dataset_root):
    """
    Scan dataset and identify what would be deleted
    Returns list of series to be deleted
    """
    dataset_path = Path(dataset_root)
    patient_dirs = sorted([d for d in dataset_path.glob("Prostate-MRI-US-Biopsy-*") if d.is_dir()])
    
    print(f"Scanning {len(patient_dirs)} patient folders...\n")
    
    series_to_delete = []
    total_series = 0
    errors = 0
    
    for patient_dir in tqdm(patient_dirs, desc="Scanning patients"):
        patient_name = patient_dir.name
        
        try:
            # Iterate through studies
            for study_dir in patient_dir.iterdir():
                if not study_dir.is_dir():
                    continue
                
                try:
                    # Iterate through series
                    for series_dir in study_dir.iterdir():
                        if not series_dir.is_dir():
                            continue
                        
                        try:
                            total_series += 1
                            
                            # Check if this is ultrasound/3D rendering
                            if is_ultrasound_series(str(series_dir)):
                                series_to_delete.append({
                                    'path': series_dir,
                                    'patient': patient_name,
                                    'study': study_dir.name,
                                    'series': series_dir.name
                                })
                        except (OSError, PermissionError) as e:
                            errors += 1
                            tqdm.write(f"⚠️  Error accessing series: {e}")
                except (OSError, PermissionError) as e:
                    errors += 1
                    tqdm.write(f"⚠️  Error accessing study: {e}")
        except (OSError, PermissionError) as e:
            errors += 1
            tqdm.write(f"⚠️  Error accessing patient {patient_name}: {e}")
    
    if errors > 0:
        print(f"\n⚠️  Encountered {errors} errors during scanning (likely corrupted directories)")
    
    return series_to_delete, total_series


def clean_dataset(dataset_root, series_to_delete, total_series):
    """
    Delete the identified ultrasound and 3D rendering volumes
    Keep only MRI (MR modality) data
    """
    print(f"\nDeleting {len(series_to_delete)} series...\n")
    
    total_removed = 0
    
    for item in tqdm(series_to_delete, desc="Deleting series"):
        try:
            shutil.rmtree(item['path'])
            total_removed += 1
        except Exception as e:
            print(f"\n✗ Error removing {item['path']}: {e}")
    
    print("\n" + "=" * 70)
    print(f"✅ Cleanup complete!")
    print(f"Total series processed: {total_series}")
    print(f"Series removed: {total_removed}")
    print(f"Series kept: {total_series - total_removed}")
    print("=" * 70)
    print("✅ Dataset now contains only MRI data.\n")


if __name__ == "__main__":
    # Dataset path
    dataset_root = r"c:\Users\Aishu\Downloads\DL - MIA\Multi-Image-Super-Resolution\data\manifest-1694710246744\Prostate-MRI-US-Biopsy"
    
    # Verify path exists
    if not Path(dataset_root).exists():
        print(f"❌ Error: Dataset path not found: {dataset_root}")
        exit(1)
    
    print("\n" + "=" * 70)
    print("DATASET CLEANUP: Remove Ultrasound 3D & Keep Only MRI")
    print("=" * 70)
    print(f"Dataset root: {dataset_root}\n")
    
    # Step 1: Scan what will be deleted
    print("STEP 1: SCANNING DATASET")
    print("-" * 70)
    series_to_delete, total_series = scan_dataset(dataset_root)
    
    # Step 2: Show preview and ask for confirmation
    print("\n" + "=" * 70)
    print("STEP 2: PREVIEW")
    print("=" * 70)
    print(f"Total series found: {total_series}")
    print(f"Series to DELETE: {len(series_to_delete)}")
    print(f"Series to KEEP: {total_series - len(series_to_delete)}")
    
    if series_to_delete:
        print("\nFirst 5 series to be deleted:")
        for i, item in enumerate(series_to_delete[:5]):
            print(f"  {i+1}. {item['patient']}/{item['study']}/{item['series']}")
        if len(series_to_delete) > 5:
            print(f"  ... and {len(series_to_delete) - 5} more")
    
    print("\n" + "=" * 70)
    print("⚠️  BREAKPOINT: Review the above before proceeding!")
    print("=" * 70)
    
    # Confirm before deletion
    confirm = input("\nProceed with DELETION? This cannot be undone! (yes/no): ").strip().lower()
    
    if confirm != "yes":
        print("❌ Cancelled. No files were deleted.")
        exit(0)
    
    print()
    clean_dataset(dataset_root, series_to_delete, total_series)
    print("✅ Done!\n")
