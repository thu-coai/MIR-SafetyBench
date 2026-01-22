import os
import json
import argparse
from collections import defaultdict
from pathlib import Path

def analyze_results(base_path):
    """
    Analyze results folder and count unsafe samples and ASR for each JSON file
    
    Args:
        base_path: Base path containing 6 category subfolders
    """
    # Store statistics for each filename
    file_stats = defaultdict(lambda: {'total': 0, 'unsafe': 0})
    
    # Traverse all subfolders
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        
        # Ensure it's a directory
        if not os.path.isdir(subdir_path):
            continue
            
        print(f"Processing folder: {subdir}")
        
        # Traverse all JSON files in the subfolder
        for filename in os.listdir(subdir_path):
            if filename.endswith('.json'):
                file_path = os.path.join(subdir_path, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Get judge_result
                    if 'judge_result' in data:
                        judge_results = data['judge_result']
                        
                        # Count total and unsafe samples
                        total_count = len(judge_results)
                        unsafe_count = sum(1 for item in judge_results if item.get('label') == 'unsafe')
                        
                        # Accumulate to total statistics
                        file_stats[filename]['total'] += total_count
                        file_stats[filename]['unsafe'] += unsafe_count
                        
                        print(f"  {filename}: total={total_count}, unsafe={unsafe_count}")
                        
                except Exception as e:
                    print(f"  Error processing file {filename}: {e}")
    
    return file_stats

def print_results(file_stats):
    """
    Print statistics results
    
    Args:
        file_stats: File statistics results
    """
    print("\n" + "="*80)
    print("Final Statistics")
    print("="*80)
    print(f"{'Filename':<30} {'Unsafe Count':<13} {'Total Count':<13} {'ASR(%)':<10}")
    print("-"*80)
    
    # Sort by filename
    for filename in sorted(file_stats.keys()):
        stats = file_stats[filename]
        total = stats['total']
        unsafe = stats['unsafe']
        
        # Calculate ASR (Attack Success Rate)
        asr = (unsafe / total * 100) if total > 0 else 0
        
        print(f"{filename:<30} {unsafe:<13} {total:<13} {asr:.2f}%")
    
    # Calculate overall statistics
    total_unsafe = sum(stats['unsafe'] for stats in file_stats.values())
    total_data = sum(stats['total'] for stats in file_stats.values())
    overall_asr = (total_unsafe / total_data * 100) if total_data > 0 else 0
    
    print("-"*80)
    print(f"{'Total':<30} {total_unsafe:<13} {total_data:<13} {overall_asr:.2f}%")
    print("="*80)

def main():
    """
    Main function
    """
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Analyze MIR-SafetyBench results, count unsafe samples and ASR')
    parser.add_argument('--path', '-p', 
                       default="project/MIR-SafetyBench/data_test/results/llamaguard3/qwen2_5_VL_7B",
                       help='Specify the folder path to analyze (default: project/MIR-SafetyBench/data_test/results/llamaguard3/qwen2_5_VL_7B)')
    
    # Parse command line arguments
    args = parser.parse_args()
    base_path = args.path
    
    # Check if path exists
    if not os.path.exists(base_path):
        print(f"Error: Path {base_path} does not exist")
        return
    
    print(f"Starting analysis for path: {base_path}")
    print("="*80)
    
    # Analyze results
    file_stats = analyze_results(base_path)
    
    # Print results
    print_results(file_stats)

if __name__ == "__main__":
    main()
