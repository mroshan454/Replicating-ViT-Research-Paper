import os
import random
import shutil
import zipfile

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Unzipped {zip_path} to {extract_to}")

def sample_images(src_dir: str, dst_dir: str, num_samples: int):
    os.makedirs(dst_dir, exist_ok=True)
    for class_folder in os.listdir(src_dir):
        class_path = os.path.join(src_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        dst_class_path = os.path.join(dst_dir, class_folder)
        os.makedirs(dst_class_path, exist_ok=True)

        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        sampled_images = random.sample(images, min(num_samples, len(images)))

        for img_name in sampled_images:
            shutil.copy2(os.path.join(class_path, img_name), os.path.join(dst_class_path, img_name))

def sample_dataset(original_dir, sampled_dir, train_samples_per_class, test_samples_per_class):
    train_src = os.path.join(original_dir, "train")
    test_src = os.path.join(original_dir, "test")

    train_dst = os.path.join(sampled_dir, "train")
    test_dst = os.path.join(sampled_dir, "test")

    print(f"Sampling {train_samples_per_class} images per class from train set...")
    sample_images(train_src, train_dst, train_samples_per_class)

    print(f"Sampling {test_samples_per_class} images per class from test set...")
    sample_images(test_src, test_dst, test_samples_per_class)

def zip_folder(folder_path: str, zip_path: str):
    shutil.make_archive(zip_path.replace('.zip',''), 'zip', folder_path)
    print(f"Created zip file: {zip_path}")

def main():
    original_zip = "happy_angry_sad.zip"
    extract_folder = "happy_angry_sad"
    sampled_folder = "happy_angry_sad_2"
    sampled_zip = "happy_angry_sad_2.zip"

    # Unzip original dataset if not already extracted
    if not os.path.exists(extract_folder):
        unzip_file(original_zip, extract_folder)
    else:
        print(f"{extract_folder} already exists, skipping unzip.")

    # Define how many images to sample per class (to total ~500 train, ~100 test)
    train_samples_per_class = 667  
    test_samples_per_class = 167  

    # Remove sampled_folder if it exists from previous runs
    if os.path.exists(sampled_folder):
        shutil.rmtree(sampled_folder)

    # Sample dataset
    sample_dataset(extract_folder, sampled_folder, train_samples_per_class, test_samples_per_class)

    # Zip the sampled dataset folder
    zip_folder(sampled_folder, sampled_zip)

if __name__ == "__main__":
    main()
