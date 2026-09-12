import os
import concurrent.futures
from pathlib import Path

def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Failed to delete {file_path}: {e}")

def delete_all_files_in_dir(directory, max_workers=8):
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        print("Invalid directory.")
        return

    files_to_delete = [str(f) for f in dir_path.iterdir() if f.is_file()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(delete_file, files_to_delete)

# 示例用法
if __name__ == "__main__":
    target_dir = "/xiangyux/AF3-30000"
    delete_all_files_in_dir(target_dir)
    print("fineshed")
