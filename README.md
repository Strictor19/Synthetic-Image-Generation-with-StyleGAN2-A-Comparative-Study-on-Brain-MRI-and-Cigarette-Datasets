# StyleGAN2 ile Farklı Veri Türlerinde Gerçekçi Yapay Görüntü Üretimi: Beyin MR ve Sigara Uygulamaları

---

## 📚 Proje Hakkında

Bu projede, modern derin öğrenme tabanlı üretici modellerden **StyleGAN2** mimarisi kullanılarak, iki farklı alan için (tıbbi: **beyin MR görüntüleri**, gündelik nesne: **sigara görselleri**) yapay (sentetik) görseller üretilmiştir. Projenin temel amacı, farklı veri türlerinin yapay görüntü üretimi üzerindeki etkisini göstermek, GAN tabanlı sentetik verinin avantajlarını ve kısıtlarını ortaya koymak, tıbbi ve nesne odaklı veri için deneysel sonuçları derinlemesine analiz etmektir.

### **Motivasyon ve Gerekçe**

- **Tıbbi Görüntülerde Veri Eksikliği ve Etik Sınırlar:** Tıbbi görüntülerin paylaşımı etik ve yasal nedenlerle sınırlı olduğundan, yüksek kaliteli sentetik veri üretimi tıp bilişiminde çok değerlidir.
- **Nesne Tespitinde Çeşitlilik:** Gerçek fotoğraf toplamak ve etiketlemek zordur. Sentetik veri, modelin çeşitlilik ve genelleme kabiliyetini artırır.
- **GAN’ların Potansiyeli ve Sınırları:** GAN tabanlı modeller teorik olarak “sonsuz” çeşitlilikte yeni görüntü üretebilir; ancak gerçek hayatta veri setinin yapısı, kaliteyi doğrudan etkiler. Proje boyunca, bu avantaj ve dezavantajlar somut örneklerle gösterilmiştir.

---

## 🔎 Kullanılan Yöntemler ve Temel Teknolojiler

- **Model:** StyleGAN2 (NVIDIA tarafından geliştirilmiş, yüksek kaliteli görüntü sentezi için en güncel GAN mimarilerinden biri)
- **Altyapı:** Python, TensorFlow 1.x, CUDA, RTX 4070 Ti Super GPU
- **Kod:** StyleGAN2 orijinal kaynak kodu + veri hazırlama ve analiz için ek Python betikleri
- **Eğitim Takibi:** TensorBoard ile FID, Loss fonksiyonları ve çıktı görsellerinin düzenli izlenmesi

### **Neden StyleGAN2?**
- Yüksek çözünürlükte (512x512 ve üstü) görsel üretebilme
- Stiller arası geçiş ve detay üzerinde kontrol
- Geliştirilmiş eğitim kararlılığı (önceki GAN’lara göre daha az artefakt)

---

## 📁 Veri Setleri ve Hazırlık

### **Beyin MR Görüntüleri**
- Kaynak: [Kaggle - Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset)
- Format: 2048 JPG, 512x512 piksel, gri ton, homojen ve hizalı veri
- **Avantaj:** Tüm görseller aynı boyut, benzer poz, arka plan sade → Modelin öğrenmesi hızlı ve stabil

### **Sigara Görselleri**
- Kaynak: Hazır veri setleri + el ile toplanan çeşitli internet görselleri
- Format: 2579 PNG, 512x512 piksel, renkli, arka planlar ve pozisyonlar aşırı çeşitli
- **Zorluk:** Çok değişken arka plan, ışık ve açı çeşitliliği → Model için karmaşık ve zorlayıcı bir problem

### **Veri Hazırlık Adımları**
- Tüm görseller 512x512 piksele yeniden boyutlandırıldı.
- Uygun format (PNG/JPG) ve normalize edilmiş pixel değerleri.
- StyleGAN2’nin beklediği zip formatına otomatik dönüştürüldü.

---

## 🧠 GAN ve StyleGAN2 Kısa Bilgi

- **GAN’lar (Üretici-Çekişmeli Ağlar):** Rastgele girdi (latent vektör) ile, gerçekçiliğe yakın sentetik görüntüler üretir. “Üretici” yeni görseller yaratır, “ayırt edici” ise bunların gerçek mi sahte mi olduğunu belirlemeye çalışır. Birbirlerine karşı rekabet ederler.
- **StyleGAN2:** Bu mekanizmaya ek olarak stil vektörlerini çok katmanlı şekilde manipüle ederek, daha detaylı ve “doğal” görünen görüntüler üretebilir. Özellikle insan yüzü, tıbbi görüntü gibi alanlarda çığır açmıştır.

---

## 🚀 Modelin Eğitimi ve Komutları

### **Temel Eğitim Komutları:**

```bash
python dataset_tool.py --source=./data/sigara --dest=./data/sigara.zip --resolution=512x512
python train.py --outdir=training-runs --data=./data/sigara.zip --gpus=1 --batch=16 --cfg=auto --mirror=1 --kimg=500 --snap=10
python generate.py --outdir=generated-images --trunc=1.0 --seeds=1-10 --network=training-runs/00001-my_dataset-auto1/network-snapshot-000500.pkl
tensorboard --logdir=training-runs --port=6006
