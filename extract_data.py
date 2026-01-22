"""
MIR-SafetyBench Data Extraction Tool (High-Speed Pipeline Version)

This script loads the MIR-SafetyBench dataset from HuggingFace or a local path,
and converts it back to the original folder structure (including JSON files and images).

🚀 Performance Optimizations:
- [Group Acceleration] Pandas vectorized grouping, metadata-only extraction, avoids loading images (100+ times faster)
- [Global Thread Pool] All tasks share one thread pool, eliminating inter-group waiting (maximizing CPU and IO utilization)
- [Non-blocking JSON] JSON written immediately, no waiting for image saving (pipeline operation)
- [Memory Optimization] Dataset lazy loading + streaming processing, minimal memory footprint

Supports two data sources:
1. Direct download from HuggingFace (requires HUGGINGFACE_TOKEN environment variable)
2. Load from locally downloaded dataset (supports formats: parquet/json/csv/arrow)

Usage:
    # Download from HuggingFace (set environment variable first)
    export HUGGINGFACE_TOKEN=your_token_here
    python extract_data.py
    
    # Load from local path
    python extract_data.py --local-path /path/to/dataset
    
    # Specify output directory
    python extract_data.py --output ./my_output --local-path /path/to/dataset

Dependencies:
    pip install datasets pillow tqdm pandas
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import concurrent.futures  # Multi-threading acceleration
import pandas as pd  # Pandas grouping acceleration

# --- Configuration ---
HUGGINGFACE_REPO_ID = "thu-coai/MIR-SafetyBench"  # HuggingFace repository ID
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', '')  # Read HuggingFace Token from environment variable

DATASET_INFO = {
    "categories": [
        "Hate_Speech", "Violence", "Self-Harm", 
        "Illegal_Activities", "Harassment", "Privacy"
    ],
    "relationship_types": [
        "Analogy", "Causality", "Complementarity", "Decomposition", 
        "Relevance", "Spatial_Embedding", "Spatial_Juxtaposition", 
        "Temporal_Continuity", "Temporal_Jump"
    ]
}


def load_dataset_from_source(local_path=None):
    """Load dataset from HuggingFace or local path
    
    Args:
        local_path: Local dataset path (if None, download from HuggingFace)
    
    Returns:
        Dataset object, or None if failed
    """
    if local_path:
        # Load from local path
        print("📂 Loading dataset from local path...")
        print(f"   Path: {local_path}")
        
        try:
            local_path = Path(local_path)
            
            # Check if path exists
            if not local_path.exists():
                print(f"❌ Path does not exist: {local_path}")
                return None
            
            # ✅ Use load_dataset directly, let it auto-detect data format
            # Supports: datasets format (arrow), parquet, json, csv, etc.
            dataset = load_dataset(str(local_path))
            
            # If DatasetDict is returned, take the first split
            if hasattr(dataset, 'keys'):
                splits = list(dataset.keys())
                print(f"   Detected multiple splits: {splits}")
                first_split = splits[0]
                print(f"   Using first split: {first_split}")
                dataset = dataset[first_split]
            
            # Try to get dataset length
            try:
                dataset_len = len(dataset)  # type: ignore
                print(f"✅ Dataset loaded successfully! Total samples: {dataset_len}")
            except:
                print(f"✅ Dataset loaded successfully!")
            
            return dataset
            
        except Exception as e:
            print(f"❌ Loading failed: {e}")
            print("\nTips: Please check:")
            print("  1. Whether path contains valid data files (parquet/json/csv/arrow)")
            print("  2. Whether dataset files are complete")
            print("  3. Try viewing directory contents to confirm file format")
            return None
    else:
        # Download from HuggingFace
        print("📥 Downloading dataset from HuggingFace...")
        print(f"   Repository: {HUGGINGFACE_REPO_ID}")
        
        try:
            # Use token (if set in environment variable)
            token = HUGGINGFACE_TOKEN if HUGGINGFACE_TOKEN else None
            dataset = load_dataset(
                HUGGINGFACE_REPO_ID,
                token=token,
                split='train'  # Use 'train' or None if dataset has no splits
            )
            # Try to get dataset length, don't display if failed
            try:
                dataset_len = len(dataset)  # type: ignore
                print(f"✅ Dataset downloaded successfully! Total samples: {dataset_len}")
            except:
                print(f"✅ Dataset downloaded successfully!")
            return dataset
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("\nTips: Please check:")
            print("  1. Whether HuggingFace Token is correct (environment variable HUGGINGFACE_TOKEN)")
            print("  2. Whether repository ID is correct")
            print("  3. Whether network connection is working")
            return None


def organize_data_by_category_and_type(dataset):
    """
    Accelerate grouping with Pandas:
    Only extract text columns for groupby, get indices then map back to dataset, avoiding image loading.
    
    Performance optimizations:
    - Don't load images during grouping phase (avoid decoding large number of images)
    - Use Pandas vectorized operations for grouping (100+ times faster than loops)
    - Return Dataset subset (lazy loading), images only loaded when actually used
    
    Returns:
        data_dict: {category: {relationship_type: Dataset}}
    """
    print("\n📊 Starting data organization (Pandas accelerated version)...")
    
    # 1. Only extract columns for grouping and convert to Pandas DataFrame
    # This is extremely fast because it doesn't involve Image column decoding
    print("   -> Extracting metadata...")
    df_meta = dataset.select_columns(['category', 'relationship_type']).to_pandas()
    
    # 2. Use Pandas groupby to instantly complete grouping, get index list for each group
    # result format: {('Hate_Speech', 'Analogy'): Int64Index([0, 1, 5...]), ...}
    print("   -> Performing vectorized grouping...")
    grouped_indices = df_meta.groupby(['category', 'relationship_type']).groups
    
    # 3. Build result dictionary
    # Note: We store Dataset slices (Subset), not list of dicts
    # Huge benefit: images are still Lazy Loaded, only read when saving, greatly saving memory
    data_dict = defaultdict(lambda: defaultdict(list))
    
    print("   -> Mapping indices to Dataset...")
    for (category, rel_type), indices in tqdm(grouped_indices.items(), desc="  - Creating subsets", leave=False):
        # dataset.select() is zero-copy operation, very fast
        # It creates a new Dataset view containing only specific indices
        subset = dataset.select(indices.tolist())  # Convert Pandas Index to list
        data_dict[category][rel_type] = subset
        
    print(f"✅ Data grouping completed!")
    print(f"   Number of categories: {len(data_dict)}")
    print(f"   Total relationship type combinations: {len(grouped_indices)}")
    
    return data_dict


def save_single_image(args):
    """
    Single image save task for thread pool execution
    
    Args:
        args: (pil_image, save_path) tuple
    
    Returns:
        bool: Whether save was successful
    """
    pil_image, save_path = args
    try:
        # Optimization: compress_level=1 speeds up saving while maintaining quality
        pil_image.save(save_path, compress_level=1)
        return True
    except Exception as e:
        # Silently handle errors to avoid cluttering progress bar
        return False


def save_images_and_generate_json(data_dict, output_path):
    """
    High-speed pipeline version: Global thread pool + non-blocking JSON generation
    
    🚀 Core optimizations:
    1. Global thread pool: All tasks share one thread pool, eliminating inter-group waiting
    2. Non-blocking JSON: JSON written immediately, no waiting for image saving
    3. Pipeline operation: Main thread continuously submits tasks, background continuously saves images, maximizing CPU and IO utilization
    
    Execution flow:
    - Main thread: Traverse all data → decode images → submit save tasks → write JSON immediately → continue to next group
    - Background thread pool: Continuously execute image save tasks (32 threads concurrent)
    - Finally: Wait for all background tasks to complete
    """
    print("\n💾 Starting full-speed pipeline processing...")
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Global thread pool configuration
    max_workers = min(32, (os.cpu_count() or 4) * 4)
    print(f"   🚀 Enabling global thread pool (Workers: {max_workers})")
    
    # Collect all futures for final waiting
    all_futures = []
    
    # Calculate total groups for progress display
    total_groups = sum(len(rel_type_dict) for rel_type_dict in data_dict.values())
    
    # 🔥 Key: Global thread pool, lifecycle covers entire processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        
        # Main thread: Quickly traverse all categories, submit tasks and generate JSON
        with tqdm(total=total_groups, desc="📝 Generating JSON & submitting tasks", unit="group") as pbar:
            
            for category, rel_type_dict in data_dict.items():
                category_path = output_path / category
                category_path.mkdir(exist_ok=True)
                
                for rel_type, samples in rel_type_dict.items():
                    images_dir = category_path / "images" / rel_type
                    images_dir.mkdir(parents=True, exist_ok=True)
                    
                    json_data = []
                    
                    # Traverse samples, decode and submit tasks
                    for sample in samples:
                        # Robust ID handling
                        raw_id = sample['id']
                        try:
                            final_id = int(raw_id)
                        except (ValueError, TypeError):
                            final_id = raw_id
                        
                        sample_id_str = str(final_id)
                        num_images = sample['num_images']
                        image_paths = []
                        
                        for img_idx in range(num_images):
                            # Decode image (main thread, CPU intensive)
                            pil_image = sample['images'][img_idx]
                            
                            img_filename = f"{rel_type}_{sample_id_str}_{img_idx}.png"
                            img_save_path = images_dir / img_filename
                            
                            # 🔥 Submit immediately to global thread pool, no waiting for result
                            future = executor.submit(save_single_image, (pil_image, img_save_path))
                            all_futures.append(future)
                            
                            # Record relative path for JSON
                            relative_path = f"images/{rel_type}/{img_filename}"
                            image_paths.append(relative_path)
                        
                        # Build JSON entry
                        json_entry = {
                            "id": final_id,
                            "original_question": sample['original_question'],
                            "relationship_type": rel_type,
                            "revised_prompt": sample['revised_prompt'],
                            "image_descriptions": sample['image_descriptions'],
                            "image_keywords": sample['image_keywords'],
                            "images": image_paths,
                            "iteration": sample['iteration']
                        }
                        json_data.append(json_entry)
                    
                    # 🔥 Key: Images still saving in background, write JSON directly!
                    # Main thread doesn't wait, immediately process next group
                    json_filename = f"{rel_type}_final.json"
                    json_path = category_path / json_filename
                    
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=2)
                    
                    pbar.update(1)
        
        # At this point all JSON files are generated, main thread task completed
        print(f"\n✅ All JSON files generated!")
        print(f"⏳ Waiting for background to write remaining images (total {len(all_futures)} images)...")
        
        # Wait for all images to save, show progress
        failed_count = 0
        with tqdm(total=len(all_futures), desc="🖼️  Writing to disk", unit="img") as pbar:
            for future in concurrent.futures.as_completed(all_futures):
                if not future.result():
                    failed_count += 1
                pbar.update(1)
        
        if failed_count > 0:
            print(f"\n⚠️  Total {failed_count}/{len(all_futures)} images failed to save")
    
    print(f"\n🎉 All completed! Output directory: {output_path}")


def verify_structure(output_path):
    """Verify generated directory structure"""
    print("\n🔍 Verifying directory structure...")
    
    output_path = Path(output_path)
    total_json_files = 0
    total_images = 0
    
    for category in DATASET_INFO['categories']:
        category_path = output_path / category
        if not category_path.exists():
            print(f"  ⚠️  Missing category folder: {category}")
            continue
        
        # Count JSON files
        json_files = list(category_path.glob("*_final.json"))
        total_json_files += len(json_files)
        
        # Count images
        images_dir = category_path / "images"
        if images_dir.exists():
            image_files = list(images_dir.glob("**/*.png"))
            total_images += len(image_files)
            print(f"  ✅ {category}: {len(json_files)} JSON files, {len(image_files)} images")
        else:
            print(f"  ⚠️  {category}: Missing images folder")
    
    print(f"\n📈 Statistics:")
    print(f"   Total JSON files: {total_json_files}")
    print(f"   Total images: {total_images}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='MIR-SafetyBench High-Speed Extraction Tool - Global Thread Pool + Pipeline Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Download from HuggingFace and extract (requires setting HUGGINGFACE_TOKEN environment variable first)
  export HUGGINGFACE_TOKEN=your_token_here
  python extract_data.py
  
  # Load from local path and extract
  python extract_data.py --local-path /path/to/downloaded/dataset
  
  # Specify output directory
  python extract_data.py --output ./my_output --local-path /path/to/dataset
        """
    )
    
    parser.add_argument(
        '--local-path',
        type=str,
        default=None,
        help='Local dataset path (HuggingFace datasets format). If not specified, will download from HuggingFace'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./MIR-SafetyBench',
        help='Output directory path (default: ./MIR-SafetyBench)'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    print("=" * 60)
    print("🚀 MIR-SafetyBench Data Extraction Tool (High-Speed Pipeline Version)")
    print("=" * 60)
    
    # Display configuration information
    if args.local_path:
        print(f"\n📂 Data source: Local path")
        print(f"   {args.local_path}")
    else:
        print(f"\n📥 Data source: HuggingFace")
        print(f"   {HUGGINGFACE_REPO_ID}")
    
    print(f"\n📁 Output directory: {args.output}")
    print()
    
    # Step 1: Load dataset
    dataset = load_dataset_from_source(local_path=args.local_path)
    if dataset is None:
        return
    
    # Step 2: Organize data
    data_dict = organize_data_by_category_and_type(dataset)
    
    # Step 3: Save images and generate JSON
    save_images_and_generate_json(data_dict, args.output)
    
    # Step 4: Verify structure
    verify_structure(args.output)
    
    print("\n" + "=" * 60)
    print("🎉 Data extraction completed!")
    print("=" * 60)
    print(f"\n📁 Output directory: {args.output}")
    print("\nDirectory structure:")
    print("  final/")
    print("  ├── Hate_Speech/")
    print("  │   ├── images/")
    print("  │   │   ├── Analogy/")
    print("  │   │   │   ├── Analogy_1_0.png")
    print("  │   │   │   └── ...")
    print("  │   │   └── ...")
    print("  │   ├── Analogy_final.json")
    print("  │   └── ...")
    print("  ├── Violence/")
    print("  └── ...")


if __name__ == "__main__":
    main()

