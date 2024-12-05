from PIL import Image
import os

# Giriş ve çıkış dizinlerini tanımla
input_dirs = [
    "GRAY-cars_tanks/cars_tanks/train/cars",
    "GRAY-cars_tanks/cars_tanks/train/tanks",
    "GRAY-cars_tanks/cars_tanks/test/cars",
    "GRAY-cars_tanks/cars_tanks/test/tanks",
]
output_dir = "GRAY-cars_tanks/resized"

# Çıkış dizinini oluştur
os.makedirs(output_dir, exist_ok=True)


# Resimleri işlemek için fonksiyon
def resize_images(input_dir, output_subdir, size=(64, 64)):
    output_path = os.path.join(output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)  # Alt dizin oluştur

    for img_file in os.listdir(input_dir):
        if img_file.endswith((".jpg", ".jpeg", ".png")):  # Sadece resim dosyaları
            input_path = os.path.join(input_dir, img_file)
            output_path_file = os.path.join(output_path, img_file)

            try:
                with Image.open(input_path) as img:
                    img_resized = img.resize(
                        size, Image.Resampling.LANCZOS
                    )  # Yeniden boyutlandır
                    img_resized.save(output_path_file)  # Yeni dosyaya kaydet
                    print(f"Resized and saved: {output_path_file}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")


# Tüm dizinler için işlemi başlat
subdirs = ["train/cars", "train/tanks", "test/cars", "test/tanks"]

for input_dir, subdir in zip(input_dirs, subdirs):
    resize_images(input_dir, subdir, size=(64, 64))
