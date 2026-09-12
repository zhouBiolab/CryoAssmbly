import os
import shutil

def main(cif_dir, npz_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    npz_prefixes = set()
    for f in os.listdir(npz_dir):
        if f.endswith('.npz'):
            npz_prefixes.add(f[:4])

    copied = 0
    for prefix in npz_prefixes:
        folder_path = os.path.join(cif_dir, prefix)
        if os.path.isdir(folder_path):
            cif_file = os.path.join(folder_path, prefix + '.cif')
            if os.path.isfile(cif_file):
                shutil.copy2(cif_file, out_dir)
                copied += 1
                print(f"Copied: {cif_file}")
            else:
                print(f"Warning: {cif_file} not found")
        else:
            print(f"Warning: folder {folder_path} not found")

    print(f"\nDone. Copied {copied} files.")

if __name__ == '__main__':
    cif_dir = input("CIF directory: ").strip()
    npz_dir = input("NPZ directory: ").strip()
    out_dir = input("Output directory: ").strip()
    main(cif_dir, npz_dir, out_dir)