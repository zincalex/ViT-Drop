import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def preprocess_h5_to_rgb(input_path, output_path):
    print(f"\n{'='*80}")
    print(f"Processing: {input_path}")
    print(f"{'='*80}")
    
    with h5py.File(input_path, 'r') as f_in:
        print("Loading sen2 data...")
        sen2 = f_in['sen2'][:]
        
        print("Loading labels...")
        labels_onehot = f_in['label'][:]
        
        num_samples = sen2.shape[0]
        print(f"Number of samples: {num_samples}")
        print(f"sen2 shape: {sen2.shape}")
        print(f"sen2 value range: [{sen2.min():.4f}, {sen2.max():.4f}]")
        
        print("\nNormalizing sen2 to [0, 255] using paper's method...")
        print(f"Original range: [{sen2.min():.4f}, {sen2.max():.4f}]")
        
        sen2_normalized = (sen2 * 255.0 / 2.8).astype(np.uint8)
        print(f"Normalized range: [{sen2_normalized.min()}, {sen2_normalized.max()}]")
        
        print("\nExtracting RGB channels (B4, B3, B2)...")
        rgb_images = sen2_normalized[:, :, :, [2, 1, 0]]
        print(f"RGB images shape: {rgb_images.shape}")
        
        print("\nConverting labels from one-hot to class indices...")
        labels = np.argmax(labels_onehot, axis=1).astype(np.int32)
        print(f"Labels shape: {labels.shape}")
        print(f"Number of classes: {len(np.unique(labels))}")
        print(f"Class distribution:")
        for cls in np.unique(labels):
            count = np.sum(labels == cls)
            percentage = (count / len(labels)) * 100
            print(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
        
        print(f"\nSaving to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f_out:
            f_out.create_dataset('images', data=rgb_images, compression='gzip', compression_opts=4)
            f_out.create_dataset('labels', data=labels, compression='gzip', compression_opts=4)
            
            f_out.attrs['num_samples'] = num_samples
            f_out.attrs['num_classes'] = len(np.unique(labels))
            f_out.attrs['image_shape'] = rgb_images.shape[1:]
            f_out.attrs['description'] = 'LCZ42 RGB images (Sentinel-2 B4,B3,B2) normalized to [0,255]'
        
        print(f"✓ Successfully saved {num_samples} samples")
        
        print("\nVerifying saved file...")
        with h5py.File(output_path, 'r') as f_verify:
            print(f"  Keys: {list(f_verify.keys())}")
            print(f"  images shape: {f_verify['images'].shape}")
            print(f"  images dtype: {f_verify['images'].dtype}")
            print(f"  images range: [{f_verify['images'][:100].min()}, {f_verify['images'][:100].max()}]")
            print(f"  labels shape: {f_verify['labels'].shape}")
            print(f"  labels dtype: {f_verify['labels'].dtype}")
            print(f"  labels range: [{f_verify['labels'][:].min()}, {f_verify['labels'][:].max()}]")
            print(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")
            print(f"  Metadata: {dict(f_verify.attrs)}")


def main():
    input_dir = Path("m1483140/m1483140")
    output_dir = Path("data/LCZ42")
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        exit(1)
    
    # SPLIT POLICY: Uses pre-existing train/val/test splits from original LCZ42 dataset
    splits = {
        'training.h5': 'train.h5',
        'validation.h5': 'val.h5',
        'testing.h5': 'test.h5'
    }
    
    for input_name, output_name in splits.items():
        input_path = input_dir / input_name
        output_path = output_dir / output_name
        
        if not input_path.exists():
            print(f"\nWarning: {input_path} not found, skipping...")
            continue
        
        preprocess_h5_to_rgb(input_path, output_path)
    
    print(f"\n{'='*80}")
    print("All files processed successfully!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
