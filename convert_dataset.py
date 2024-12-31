import os
import argparse
from datasets import load_dataset
import re
import os
import shutil

def process_dataset(args):
    """
    Load a dataset from a directory, remove original files, and save it back to disk.
    
    :param args: Argument object containing save_dir
    """
    # Load the dataset from the specified directory
    dataset = load_dataset(args.save_dir)

    # Remove all files in the save directory that start with 'data'
    os.system(f"rm -rf {args.save_dir}/data*")

    # Save the dataset back to disk
    dataset.save_to_disk(args.save_dir)
    
    print("Dataset processing completed!")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process and save dataset")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the processed dataset")

    # Parse arguments
    args = parser.parse_args()

    def get_cache_dir(rank=None):
        cache_pattern = re.compile(r"step_(\d+)_to_(\d+)_rank_(\d+)")
        if rank is None:
            return [os.path.join(args.save_dir, dir) for dir in os.listdir(args.save_dir) if cache_pattern.match(dir)]
        else:
            return [os.path.join(args.save_dir, dir) for dir in os.listdir(args.save_dir) \
                    if cache_pattern.match(dir) and cache_pattern.match(dir).group(3) == str(rank)]

    cache_dir = get_cache_dir()
    total_shards = 0
    for dir in cache_dir:
        arrow_files = [file for file in os.listdir(dir) if file.endswith(".arrow")]
        total_shards += len(arrow_files)
        
    current_shard_index = 0 
    for dir in cache_dir:
        arrow_files = [file for file in os.listdir(dir) if file.endswith(".arrow")]
        for file in arrow_files:
            new_filename = f"data0-{current_shard_index:05d}-of-{total_shards:05d}.arrow"
            current_shard_index += 1

            source_file_path = os.path.join(dir, file)
            destination_file_path = os.path.join(args.save_dir, new_filename)
            shutil.move(source_file_path, destination_file_path)
        shutil.rmtree(dir)

    # Call the main function
    process_dataset(args)