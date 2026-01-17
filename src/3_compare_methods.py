import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# Kendi yazdığımız modülden hem Standart hem de Kaotik PSO sınıflarını çağırıyoruz.
from pso_optimizer import BinaryPSO, ChaoticBinaryPSO
import time
import os

# ========================================================
# AYARLAR VE DİZİN YÖNETİMİ
# ========================================================
# Kodun farklı bilgisayarlarda (örn: Hoca'nın bilgisayarı) hata vermeden çalışması için
# dinamik dosya yolu bulma yöntemi kullanıyoruz.
current_dir = os.path.dirname(os.path.abspath(__file__))  # src klasörü
project_root = os.path.dirname(current_dir)  # Ana proje klasörü (src'nin bir üstü)
DATA_PATH = os.path.join(project_root, "processed_data", "phishing_clean.csv")

# Deney Parametreleri
# Her iki algoritma da adil karşılaştırma için aynı parçacık ve iterasyon sayısıyla çalıştırılır.
N_PARTICLES = 20
N_ITERATIONS = 40


def load_data(path):
    """
    Veriyi yükler, hedef değişkeni ayırır ve özellikleri 0-1 arasına normalize eder.
    """
    # Dosya var mı kontrol et (Hata Yönetimi)
    if not os.path.exists(path):
        raise FileNotFoundError(f"HATA: Veri dosyası bulunamadı! Aranan yer: {path}")

    df = pd.read_csv(path)

    # Özellikler (X) ve Hedef (y) ayrımı
    X = df.drop(columns=['target']).values
    y = df['target'].values

    # Normalizasyon: Verileri 0-1 aralığına sıkıştırır.
    # Bu işlem optimizasyon algoritmalarının daha hızlı yakınsamasına yardımcı olur.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


# --- ANA AKIŞ (BENCHMARK TESTİ) ---
if __name__ == "__main__":
    try:
        X, y = load_data(DATA_PATH)
        print(f"Veri başarıyla yüklendi: {DATA_PATH}")
    except FileNotFoundError as e:
        print(e)
        exit()

    print("\n--- METOTLAR KARŞILAŞTIRILIYOR ---\n")

    # --------------------------------------------------------
    # 1. STANDART PSO (BASELINE)
    # --------------------------------------------------------
    # İlk olarak klasik algoritmayı çalıştırıyoruz. Bu bizim referans noktamızdır.
    print("1. Standart Binary PSO Çalışıyor...")
    standard_pso = BinaryPSO(n_particles=N_PARTICLES, n_iterations=N_ITERATIONS)

    st_start = time.time()  # Süre ölçümü başlangıcı
    standard_pso.fit(X, y)
    st_time = time.time() - st_start  # Geçen süreyi hesapla

    print(f"Standart PSO Bitti. Başarı: %{standard_pso.best_score_ * 100:.2f}\n")

    # --------------------------------------------------------
    # 2. KAOTİK PSO (PROPOSED METHOD)
    # --------------------------------------------------------
    # İkinci olarak kaotik haritalarla güçlendirilmiş algoritmayı çalıştırıyoruz.
    # Beklentimiz: Yerel tuzaklardan (local optima) daha iyi kaçması ve daha yüksek doğruluk vermesi.
    print("2. Kaotik (Logistic) Binary PSO Çalışıyor...")
    chaotic_pso = ChaoticBinaryPSO(n_particles=N_PARTICLES, n_iterations=N_ITERATIONS)

    ch_start = time.time()
    chaotic_pso.fit(X, y)
    ch_time = time.time() - ch_start

    print(f"Kaotik PSO Bitti. Başarı: %{chaotic_pso.best_score_ * 100:.2f}\n")

    # ========================================================
    # 3. GRAFİK ÇİZİMİ VE KAYDETME (VISUALIZATION)
    # ========================================================
    # Her iki algoritmanın öğrenme sürecini (Convergence Curve) aynı grafik üzerinde gösteriyoruz.
    plt.figure(figsize=(12, 7))

    # Standart PSO Çizgisi (Kırmızı, kesik çizgi)
    plt.plot(standard_pso.convergence_curve_, color='red', linestyle='--', marker='o',
             linewidth=2, markersize=5,
             label=f'Standart PSO (En İyi: %{standard_pso.best_score_ * 100:.2f})')

    # Kaotik PSO Çizgisi (Mavi, düz çizgi - vurgulamak için)
    plt.plot(chaotic_pso.convergence_curve_, color='blue', linestyle='-', marker='s',
             linewidth=2, markersize=5,
             label=f'Kaotik PSO (En İyi: %{chaotic_pso.best_score_ * 100:.2f})')

    # Eksen etiketleri ve başlık
    plt.title('Standart vs Kaotik PSO Karşılaştırması (Phishing Tespiti)', fontsize=14)
    plt.xlabel('İterasyon Sayısı', fontsize=12)
    plt.ylabel('Doğruluk (Accuracy)', fontsize=12)
    plt.legend(fontsize=12)  # Hangi rengin ne olduğunu gösteren kutucuk
    plt.grid(True, linestyle='--', alpha=0.6)  # Arka plan ızgarası

    # --- GRAFİK KAYDETME İŞLEMİ ---
    # Plots klasörünü hedefle
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)  # Klasör yoksa oluştur

    # Dosya ismini belirle
    save_path = os.path.join(plots_dir, "Compare_Standard_vs_Chaotic_PSO.png")

    # Grafiği kaydet:
    # dpi=300: Yüksek çözünürlük (tez/makale standardı)
    # bbox_inches='tight': Grafiğin kenar boşluklarını otomatik kırpar, yazıların kesilmesini önler.
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"\n[BİLGİ] Karşılaştırma grafiği başarıyla kaydedildi: {save_path}")

    plt.show()  # Son olarak grafiği ekranda göster