from PIL import Image
import os

# Giriş ve çıkış dizinlerini tanımla
input_dirs = [
    "20_20/0",
    "20_20/1",
]
output_dir = "20_20/resized"

# Çıkış dizinini oluştur
os.makedirs(output_dir, exist_ok=True)


# Resimleri işlemek için fonksiyon
def resize_images(input_dir, output_subdir, size=(32, 32)):
    output_path = os.path.join(output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)  # Alt dizin oluştur

    counter = 0  # Dosya isimlendirme için sayaç

    for img_file in os.listdir(input_dir):
        if img_file.endswith((".jpg", ".jpeg", ".png")):  # Sadece resim dosyaları
            input_path = os.path.join(input_dir, img_file)
            output_file_name = f"{counter}.jpg"  # Yeni dosya adı
            output_path_file = os.path.join(output_path, output_file_name)

            try:
                with Image.open(input_path) as img:
                    # Alfa kanalı varsa RGB'ye dönüştür
                    if img.mode == "RGBA":
                        img = img.convert("RGB")

                    img_resized = img.resize(
                        size, Image.Resampling.LANCZOS
                    )  # Yeniden boyutlandır
                    img_resized.save(
                        output_path_file, format="JPEG"
                    )  # Yeni dosyaya kaydet
                    print(f"Resized and saved: {output_path_file}")
                    counter += 1  # Sayaç artır
            except Exception as e:
                print(f"Error processing {input_path}: {e}")


# Tüm dizinler için işlemi başlat
subdirs = ["train/0", "train/1", "test/0", "test/1"]

for input_dir, subdir in zip(input_dirs, subdirs):
    resize_images(input_dir, subdir, size=(32, 32))
