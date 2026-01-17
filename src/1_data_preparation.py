import os
import pandas as pd
import numpy as np
from scipy.io import arff
import urllib.request
import matplotlib.pyplot as plt  # Veri görselleştirme ve grafik çizimi için gerekli kütüphane

# ========================================================
# PROJE YAPILANDIRMASI VE DİZİN YÖNETİMİ
# ========================================================
# Veri akışını düzenlemek için 3 ana klasör tanımlanıyor:
# 1. DATA_DIR: Ham veri dosyalarının (indirilmiş/orijinal) durduğu yer.
# 2. PROCESSED_DIR: Temizlenmiş ve modele hazır CSV'lerin durduğu yer.
# 3. PLOT_DIR: Analiz sonucu üretilen grafik görsellerinin kaydedileceği yer.
DATA_DIR = "data"
PROCESSED_DIR = "processed_data"
PLOT_DIR = "plots"

# Klasörler dosya sisteminde yoksa hata vermeden (exist_ok=True) oluşturulur.
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("--- FAZ 1: VERİ HAZIRLIĞI VE ANALİZİ ---\n")

# ========================================================
# 1. VERİ SETİ: UCI PHISHING WEBSITES (Ana Veri Seti)
# ========================================================
# Bu blok, projenin ana veri seti olan Phishing verisini işler.
print("1. UCI Phishing Dataset İşleniyor...")
phishing_path = os.path.join(DATA_DIR, "Training Dataset.arff")

try:
    # ARFF formatındaki dosya yüklenir (data: ham veri, meta: sütun bilgileri).
    data, meta = arff.loadarff(phishing_path)
    df_phishing = pd.DataFrame(data)

    # --- VERİ TİPİ DÜZELTME ---
    # ARFF dosyalarından gelen veriler bazen 'byte string' (örn: b'1') formatında olur.
    # Bu döngü, object tipindeki sütunları bulur ve sayısal işlem yapılabilmesi için integer'a çevirir.
    for col in df_phishing.columns:
        if df_phishing[col].dtype == object:
            df_phishing[col] = df_phishing[col].str.decode('utf-8').astype(int)

    # --- HEDEF DEĞİŞKEN (TARGET) DÖNÜŞÜMÜ ---
    # Orijinal Veri: -1 (Phishing), 1 (Legitimate)
    # Model İçin Standart: 1 (Phishing/Pozitif), 0 (Legitimate/Negatif)
    # Bu dönüşüm, modelin performans metriklerinin (Precision/Recall) doğru yorumlanmasını sağlar.
    df_phishing['target'] = df_phishing['Result'].apply(lambda x: 1 if x == -1 else 0)

    # Eski etiket sütunu silinir.
    df_phishing = df_phishing.drop(columns=['Result'])

    # Temizlenmiş veriyi CSV olarak kaydetme.
    df_phishing.to_csv(os.path.join(PROCESSED_DIR, "phishing_clean.csv"), index=False)
    print(f"   -> Başarılı! Kayıt yeri: {PROCESSED_DIR}/phishing_clean.csv")

except FileNotFoundError:
    print(f"   HATA: '{phishing_path}' dosyası bulunamadı! Lütfen elindeki dosyayı 'data' klasörüne koy.")
    # Kodun ilerleyen kısımlarında (grafik çiziminde) çökmemesi için boş bir DataFrame oluşturulur.
    df_phishing = pd.DataFrame(columns=['target'])

# ========================================================
# 2. VERİ SETİ: UCI SPAMBASE (Doğrulama Veri Seti)
# ========================================================
# Bu blok, modelin genelleme yeteneğini test etmek için kullanılacak ikincil veri setini işler.
print("\n2. UCI Spambase Dataset İndiriliyor...")
spam_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
spam_path = os.path.join(DATA_DIR, "spambase.data")

# Dosya yoksa internetten indirilir, varsa indirme atlanır.
if not os.path.exists(spam_path):
    try:
        urllib.request.urlretrieve(spam_url, spam_path)
        print("   -> İnternetten indirildi.")
    except Exception as e:
        print(f"   -> İndirme Hatası: {e}")
else:
    print("   -> Dosya zaten mevcut.")

# Veri okuma ve işleme
try:
    # Bu veri setinde sütun başlıkları yoktur (header=None).
    df_spam = pd.read_csv(spam_path, header=None)

    # Son sütun (57. index) hedef değişkendir. İsmi 'target' yapılır.
    df_spam.rename(columns={57: 'target'}, inplace=True)

    # Özellik sütunlarına F0, F1... şeklinde isim verilir.
    feature_cols = [f"F{i}" for i in range(57)]
    df_spam.columns = feature_cols + ['target']

    # Kaydetme işlemi
    df_spam.to_csv(os.path.join(PROCESSED_DIR, "spam_clean.csv"), index=False)
    print(f"   -> Başarılı! Kayıt yeri: {PROCESSED_DIR}/spam_clean.csv")
except Exception as e:
    print(f"   -> Spam veri seti okunamadı: {e}")
    df_spam = pd.DataFrame(columns=['target'])

# ========================================================
# 3. RAPORLAMA (EDA ÖZET TABLOSU)
# ========================================================
# Veri setlerinin boyutlarını ve sınıf dağılımlarını konsola yazdırır.
print("\n" + "=" * 60)
print(f"{'VERİ SETİ':<20} | {'ÖRNEK SAYISI':<12} | {'ÖZELLİK':<8} | {'SINIF DAĞILIMI (1/0)'}")
print("=" * 60)

# Phishing İstatistikleri (DataFrame boş değilse hesapla)
if not df_phishing.empty:
    n_samples = df_phishing.shape[0]
    n_features = df_phishing.shape[1] - 1
    n_pos = df_phishing['target'].sum()  # 1 olanlar (Phishing)
    n_neg = n_samples - n_pos  # 0 olanlar (Safe)
    print(f"{'UCI Phishing':<20} | {n_samples:<12} | {n_features:<8} | {n_pos} (Phish) / {n_neg} (Safe)")

# Spam İstatistikleri (DataFrame boş değilse hesapla)
if not df_spam.empty:
    n_samples = df_spam.shape[0]
    n_features = df_spam.shape[1] - 1
    n_pos = df_spam['target'].sum()  # 1 olanlar (Spam)
    n_neg = n_samples - n_pos  # 0 olanlar (Safe)
    print(f"{'UCI Spambase':<20} | {n_samples:<12} | {n_features:<8} | {n_pos} (Spam)  / {n_neg} (Safe)")
print("=" * 60)

# ========================================================
# 4. GRAFİK OLUŞTURMA (GÖRSELLEŞTİRME MODÜLÜ)
# ========================================================
# Veri setlerindeki sınıf dengesizliğini (imbalance) görmek için bar grafikleri oluşturulur.
print("\n4. Veri Seti Grafikleri Oluşturuluyor...")


def plot_class_distribution(df, dataset_name, file_name, class_labels):
    """
    Verilen DataFrame'in hedef değişken dağılımını çizer ve kaydeder.
    Parametreler:
        df: Veri seti
        dataset_name: Grafiğin başlığında görünecek isim
        file_name: Kaydedilecek dosya adı (örn: grafik.png)
        class_labels: X ekseninde yazacak etiketler ['Güvenli', 'Tehdit']
    """
    if df.empty:
        return

    # Sınıf sayılarını hesapla (0 ve 1 kaç adet?)
    counts = df['target'].value_counts().sort_index()

    # Grafik alanını oluştur
    plt.figure(figsize=(8, 6))

    # Bar grafiği çizimi (Yeşil: Güvenli, Kırmızı: Tehdit)
    bars = plt.bar(counts.index, counts.values, color=['green', 'red'], alpha=0.7)

    # Başlık ve eksen isimlendirmeleri
    plt.title(f'{dataset_name} - Sınıf Dağılımı', fontsize=14)
    plt.xlabel('Sınıf', fontsize=12)
    plt.ylabel('Örnek Sayısı', fontsize=12)
    plt.xticks(counts.index, class_labels, fontsize=11)  # 0 ve 1 yerine etiket isimlerini yaz
    plt.grid(axis='y', linestyle='--', alpha=0.5)  # Arka plana yatay çizgiler ekle

    # Barların üzerine net sayıları yazdırma döngüsü
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Grafiği dosyaya kaydet ve belleği temizle
    save_path = os.path.join(PLOT_DIR, file_name)
    plt.savefig(save_path, dpi=300)  # Yüksek çözünürlük (300 dpi)
    plt.close()
    print(f"   -> Grafik kaydedildi: {save_path}")


# Phishing Grafiği Çizimi (0: Güvenli, 1: Phishing)
plot_class_distribution(df_phishing, "UCI Phishing", "phishing_dagilim.png", ["Güvenli (0)", "Phishing (1)"])

# Spam Grafiği Çizimi (0: Güvenli, 1: Spam)
plot_class_distribution(df_spam, "UCI Spambase", "spam_dagilim.png", ["Güvenli (0)", "Spam (1)"])

print(f"\nSONUÇ: Tüm veriler ve grafikler '{PROCESSED_DIR}' ve '{PLOT_DIR}' klasörlerine kaydedildi.")