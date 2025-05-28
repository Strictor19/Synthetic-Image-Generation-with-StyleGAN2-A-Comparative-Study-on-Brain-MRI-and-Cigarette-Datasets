import os
from PIL import Image

# Girdi klasörü (orijinal görüntüler)
input_folder = r"C:\Users\USER\Desktop\brain_tumor"
# Çıktı klasörü (yeniden boyutlandırılmış görüntüler)
output_folder = r"C:\Users\USER\Desktop\Beyin"

# Çıktı klasörünü oluştur
os.makedirs(output_folder, exist_ok=True)

# Tüm dosyaları döngüyle işle
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((512, 512), Image.LANCZOS)
        img_resized.save(os.path.join(output_folder, filename))
print("Tüm görseller 512x512 olarak yeniden boyutlandırıldı.")
