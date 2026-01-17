import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pso_optimizer import BinaryPSO  # Kendi yazdığımız PSO sınıfını projeye dahil ediyoruz
import time
import os  # Dosya yolları ve klasör işlemleri için gerekli kütüphane

# ========================================================
# AYARLAR VE DİZİN YÖNETİMİ
# ========================================================
# Kodun farklı bilgisayarlarda hata vermeden çalışması için
# dosya yollarını (path) dinamik olarak alıyoruz.
current_dir = os.path.dirname(os.path.abspath(__file__))  # Şu anki dosyanın olduğu klasör
project_root = os.path.dirname(current_dir)  # Bir üst klasör (Proje ana dizini)
DATA_PATH = os.path.join(project_root, "processed_data", "phishing_clean.csv")

# PSO Algoritması için Hiperparametreler
N_PARTICLES = 20  # Sürüdeki parçacık (ajan) sayısı
N_ITERATIONS = 30  # Algoritmanın kaç tur döneceği


def load_data(path):
    """
    Veriyi okur, özellikleri (X) ve hedefi (y) ayırır ve
    0-1 aralığında normalize eder.
    """
    print(f"Veri Yükleniyor: {path}")

    # Dosya var mı kontrolü (Hata yönetimi)
    if not os.path.exists(path):
        raise FileNotFoundError(f"HATA: Veri dosyası bulunamadı -> {path}")

    # CSV dosyasını Pandas DataFrame olarak oku
    df = pd.read_csv(path)

    # Hedef değişkeni (y) ayır, geri kalanı özellik (X) yap
    X = df.drop(columns=['target']).values
    y = df['target'].values

    # --- NORMALİZASYON (ÖNEMLİ ADIM) ---
    # PSO ve KNN gibi mesafe temelli algoritmalar, sayısal değerlerin büyüklüğünden etkilenir.
    # Tüm özellikleri 0 ile 1 arasına sıkıştırarak (MinMax) modelin daha dengeli öğrenmesini sağlıyoruz.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Raporlama yaparken hangi özelliklerin seçildiğini ismen yazabilmek için sütun adlarını saklıyoruz.
    feature_names = df.drop(columns=['target']).columns
    return X_scaled, y, feature_names


# --- ANA AKIŞ (MAIN) ---
if __name__ == "__main__":
    # 1. Veriyi Hazırla
    try:
        X, y, feat_names = load_data(DATA_PATH)
    except Exception as e:
        print(e)
        exit()  # Hata varsa programı durdur

    print("-" * 50)
    print("DENEY BAŞLIYOR: Binary PSO Feature Selection")
    print("-" * 50)

    # 2. Modeli Kur (Sınıfımızdan nesne üretiyoruz)
    # BinaryPSO sınıfı, optimizasyon işlemini yönetecek olan beyni oluşturur.
    # w (Atalet): Parçacığın mevcut hızını koruma isteği (0.7 dengeli bir seçim)
    # c1 (Bilişsel): Parçacığın kendi en iyi konumuna dönme isteği
    # c2 (Sosyal): Parçacığın sürünün en iyi konumuna gitme isteği
    pso = BinaryPSO(n_particles=N_PARTICLES, n_iterations=N_ITERATIONS, w=0.7, c1=2.0, c2=2.0)

    # 3. Eğitimi Başlat (Optimizasyon Süreci)
    start_time = time.time()

    # fit() fonksiyonu en iyi özellik kombinasyonunu bulmak için iterasyonları çalıştırır.
    pso.fit(X, y)

    end_time = time.time()

    # 4. Sonuçları Raporla
    # pso.best_solution_ dizisi 1 ve 0'lardan oluşur. 1 olan indeksler seçilen özellikleri temsil eder.
    selected_indices = [i for i, val in enumerate(pso.best_solution_) if val == 1]
    selected_features = feat_names[selected_indices]

    print("\n" + "=" * 50)
    print(f"SONUÇ RAPORU (Süre: {end_time - start_time:.2f} saniye)")
    print("=" * 50)
    print(f"Maksimum Doğruluk (Accuracy): %{pso.best_score_ * 100:.2f}")
    print(f"Toplam Özellik Sayısı: {X.shape[1]}")
    print(f"Seçilen Özellik Sayısı: {len(selected_features)}")
    print(f"Seçilen Özellikler:\n{selected_features.tolist()}")

    # 5. Grafiği Çiz, Kaydet ve Göster
    # Yakınsama Eğrisi (Convergence Curve): Algoritmanın her iterasyonda çözümünü ne kadar iyileştirdiğini gösterir.
    plt.figure(figsize=(10, 6))
    plt.plot(pso.convergence_curve_, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title('Binary PSO Yakınsama Eğrisi (Phishing)', fontsize=14)
    plt.xlabel('İterasyon', fontsize=12)
    plt.ylabel('Doğruluk (Accuracy)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # --- Grafik Kaydetme İşlemleri ---
    plots_dir = os.path.join(project_root, "plots")  # Grafikler 'plots' klasörüne gidecek
    os.makedirs(plots_dir, exist_ok=True)  # Klasör yoksa oluştur

    save_path = os.path.join(plots_dir, "Experiment_1_Convergence.png")
    plt.savefig(save_path, dpi=300)  # dpi=300 ile yüksek çözünürlüklü (makale/tez standardı) kayıt
    print(f"\n[BİLGİ] Grafik şuraya kaydedildi: {save_path}")
    # -------------------------------

    plt.show()  # Grafiği ekrana bas