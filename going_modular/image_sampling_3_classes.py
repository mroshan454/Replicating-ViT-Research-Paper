import os
import random
import shutil
import zipfile

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Unzipped {zip_path} to {extract_to}")

def create_train_test_split(src_dir, dest_dir, train_count=500, test_count=150):
    # ✅ Pick only the 3 classes you want
    selected_classes = [
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_healthy"
    ]

    for class_name in selected_classes:
        class_path = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"⚠️ Warning: {class_name} not found in {src_dir}")
            continue

        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images)

        train_images = images[:train_count]
        test_images = images[train_count:train_count+test_count]

        # Paths to save
        train_dest = os.path.join(dest_dir, "train", class_name)
        test_dest = os.path.join(dest_dir, "test", class_name)
        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(test_dest, exist_ok=True)

        for img in train_images:
            shutil.copy2(os.path.join(class_path, img), os.path.join(train_dest, img))

        for img in test_images:
            shutil.copy2(os.path.join(class_path, img), os.path.join(test_dest, img))

        print(f"✅ Processed class: {class_name} | Train: {len(train_images)} | Test: {len(test_images)}")

def zip_folder(folder_path: str, zip_path: str):
    if not os.path.exists(folder_path):
        print(f"❌ Cannot zip — folder not found: {folder_path}")
        return
    shutil.make_archive(zip_path.replace('.zip',''), 'zip', folder_path)
    print(f"✅ Created zip file: {zip_path}")

def main():
    original_zip = "Plant_Disease.zip"
    extracted_dir = "Plant_Disease"
    output_dir = "Plant_Disease_for_ViT"
    output_zip = "Plant_Disease_for_ViT.zip"

    # Step 1: Unzip
    if not os.path.exists(extracted_dir):
        unzip_file(original_zip, extracted_dir)
    else:
        print(f"⚠️ {extracted_dir} already exists. Skipping unzip.")

    # Step 2: Remove previous output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Step 3: Create sampled dataset
    create_train_test_split(extracted_dir, output_dir, train_count=500, test_count=150)

    # Step 4: Zip the result
    zip_folder(output_dir, output_zip)

if __name__ == "__main__":
    main()
