import os
import pandas as pd
import numpy as np
from scipy.io import arff
import urllib.request

# ========================================================
# PROJE YAPILANDIRMASI VE DİZİN YÖNETİMİ
# ========================================================
# Verilerin indirileceği ham klasör (DATA_DIR) ve
# işlendikten sonra temiz hallerinin saklanacağı klasör (PROCESSED_DIR) belirlenir.
DATA_DIR = "data"
PROCESSED_DIR = "processed_data"

# Klasörler dosya sisteminde yoksa otomatik olarak oluşturulur.
# exist_ok=True parametresi, klasör zaten varsa hata vermesini engeller.
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("--- FAZ 1: VERİ HAZIRLIĞI VE ANALİZİ ---\n")

# ========================================================
# 1. VERİ SETİ: UCI PHISHING WEBSITES (Ana Veri Seti)
# ========================================================
# Bu bölüm, projenin ana odağı olan Phishing veri setini işler.
# .arff formatındaki dosya okunur, temizlenir ve CSV'ye dönüştürülür.
print("1. UCI Phishing Dataset İşleniyor...")
phishing_path = os.path.join(DATA_DIR, "Training Dataset.arff")

try:
    # ARFF dosyasını yükleme işlemi (data: veri, meta: veri hakkında bilgi)
    data, meta = arff.loadarff(phishing_path)
    df_phishing = pd.DataFrame(data)

    # --- VERİ TİPİ DÖNÜŞÜMÜ ---
    # ARFF dosyaları yüklenirken bazen sayısal değerler 'byte string' (örn: b'1') olarak gelir.
    # Bu döngü, object (string/byte) tipindeki sütunları tespit eder,
    # utf-8 formatında decode eder ve matematiksel işlem yapılabilmesi için integer'a çevirir.
    for col in df_phishing.columns:
        if df_phishing[col].dtype == object:
            df_phishing[col] = df_phishing[col].str.decode('utf-8').astype(int)

    # --- HEDEF DEĞİŞKEN (TARGET) DÜZENLEMESİ ---
    # Orijinal Veri Setinde: -1 (Phishing/Kötü), 1 (Legitimate/İyi) şeklindedir.
    # Makine öğrenmesi modelleri genellikle "tespit edilecek sınıfı" 1 olarak görmeyi sever.
    # Dönüşüm: -1 -> 1 (Phishing), diğerleri -> 0 (Legitimate)
    df_phishing['target'] = df_phishing['Result'].apply(lambda x: 1 if x == -1 else 0)

    # Eski 'Result' sütunu artık gereksiz olduğu için kaldırılır.
    df_phishing = df_phishing.drop(columns=['Result'])

    # Temizlenmiş veriyi CSV olarak kaydetme
    df_phishing.to_csv(os.path.join(PROCESSED_DIR, "phishing_clean.csv"), index=False)
    print(f"   -> Başarılı! Kayıt yeri: {PROCESSED_DIR}/phishing_clean.csv")

except FileNotFoundError:
    print(f"   HATA: '{phishing_path}' dosyası bulunamadı! Lütfen elindeki dosyayı 'data' klasörüne koy.")

# ========================================================
# 2. VERİ SETİ: UCI SPAMBASE (Doğrulama/Benchmark Veri Seti)
# ========================================================
# Bu bölüm, modelin genelleştirilebilirliğini test etmek için kullanılacak
# ikincil veri seti (Spam E-posta verisi) içindir.
print("\n2. UCI Spambase Dataset İndiriliyor...")
spam_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
spam_path = os.path.join(DATA_DIR, "spambase.data")

# --- OTOMATİK İNDİRME ---
# Eğer veri seti yerel diskte yoksa, UCI deposundan otomatik olarak indirilir.
if not os.path.exists(spam_path):
    urllib.request.urlretrieve(spam_url, spam_path)
    print("   -> İnternetten indirildi.")
else:
    print("   -> Dosya zaten mevcut.")

# Pandas ile okuma (Bu ham veride sütun isimleri (header) bulunmamaktadır)
df_spam = pd.read_csv(spam_path, header=None)

# --- SÜTUN İSİMLENDİRME ---
# Son sütun (57. indeks) hedef değişkendir (1: Spam, 0: Not Spam).
df_spam.rename(columns={57: 'target'}, inplace=True)

# Geriye kalan 57 özellik için (F0, F1...) şeklinde jenerik isimler atanır.
feature_cols = [f"F{i}" for i in range(57)]
df_spam.columns = feature_cols + ['target']

# Temizlenmiş veriyi CSV olarak kaydetme
df_spam.to_csv(os.path.join(PROCESSED_DIR, "spam_clean.csv"), index=False)
print(f"   -> Başarılı! Kayıt yeri: {PROCESSED_DIR}/spam_clean.csv")

# ========================================================
# 3. RAPORLAMA VE İSTATİSTİKLER (EDA ÖZETİ)
# ========================================================
# İşlenen veri setlerinin boyutlarını ve sınıf dağılımını (denge durumunu)
# gösteren bir özet tablo ekrana basılır. Bu çıktı Rapor/Makale Tablo 1 için kullanılır.

print("\n" + "=" * 60)
print(f"{'VERİ SETİ':<20} | {'ÖRNEK SAYISI':<12} | {'ÖZELLİK':<8} | {'SINIF DAĞILIMI (1/0)'}")
print("=" * 60)

# --- Phishing İstatistikleri ---
n_samples = df_phishing.shape[0]  # Toplam satır sayısı
n_features = df_phishing.shape[1] - 1  # Sütun sayısı eksi hedef sütun
n_pos = df_phishing['target'].sum()  # Hedef değeri 1 olanların sayısı (Phishing)
n_neg = n_samples - n_pos  # Hedef değeri 0 olanların sayısı (Legitimate)
print(f"{'UCI Phishing':<20} | {n_samples:<12} | {n_features:<8} | {n_pos} (Phish) / {n_neg} (Safe)")

# --- Spam İstatistikleri ---
n_samples = df_spam.shape[0]
n_features = df_spam.shape[1] - 1
n_pos = df_spam['target'].sum()  # Hedef değeri 1 olanların sayısı (Spam)
n_neg = n_samples - n_pos  # Hedef değeri 0 olanların sayısı (Safe)
print(f"{'UCI Spambase':<20} | {n_samples:<12} | {n_features:<8} | {n_pos} (Spam)  / {n_neg} (Safe)")
print("=" * 60)

print("\nSONUÇ: İki veri seti de başarıyla 'processed_data' klasörüne temizlenmiş olarak kaydedildi.")