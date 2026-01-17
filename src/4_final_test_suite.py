import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from pso_optimizer import BinaryPSO, ChaoticBinaryPSO

# ========================================================
# DENEY KONFİGÜRASYONU (AKADEMİK STANDARTLAR)
# ========================================================
# N_RUNS: PSO stokastik (rastgele tabanlı) bir algoritma olduğu için,
# tek bir sonuç yanıltıcı olabilir. Deneyi 10 kez tekrarlayıp ortalamasını alıyoruz.
N_RUNS = 10
N_PARTICLES = 20
N_ITERATIONS = 40
N_FOLDS = 5  # Veriyi 5 parçaya bölüp test edeceğiz (5-Fold Cross Validation)
RESULTS_FILE = "final_results_report.csv"  # Sonuçların yazılacağı Excel/CSV dosyası

# Dosya yollarını dinamik olarak bul
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
datasets = {
    "Phishing": os.path.join(project_root, "processed_data", "phishing_clean.csv"),
    "Spambase": os.path.join(project_root, "processed_data", "spam_clean.csv")
}
PLOT_DIR = os.path.join(project_root, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_data(path):
    """
    Belirtilen yoldan veriyi okur, X (Özellikler) ve y (Etiket) olarak ayırır.
    """
    if not os.path.exists(path):
        print(f"UYARI: {path} bulunamadı, bu veri seti atlanıyor.")
        return None, None
    df = pd.read_csv(path)
    X = df.drop(columns=['target']).values
    y = df['target'].values
    return X, y


#

def evaluate_with_cv(optimizer_class, X, y):
    """
    BU FONKSİYON PROJENİN KALBİDİR.
    Amacı: Data Leakage (Veri Sızıntısı) olmadan modelin başarısını ölçmek.

    Data Leakage Nedir?
    Eğer tüm veriyi en başta normalize edip sonra bölersek, test setindeki bilgi
    eğitim setine sızmış olur. Bu yüzden scaling (ölçekleme) işlemi
    HER FOLD İÇİNDE ayrı ayrı yapılmalıdır.
    """
    # StratifiedKFold: Sınıf dağılımını (örneğin %60 spam, %40 normal) her parçada korur.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_features = []

    # Her döngüde verinin %80'i eğitim (train), %20'si test (test) olur.
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # --- DATA LEAKAGE ÖNLEMİ ---
        # Scaler SADECE X_train verisine bakarak öğrenir (fit).
        # X_test verisi, X_train'den öğrenilen parametrelerle dönüştürülür (transform).
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- ÖZELLİK SEÇİMİ (FEATURE SELECTION) ---
        # Optimizer sadece eğitim setini görür. Test setini asla görmez.
        # cv=3 parametresi, PSO'nun kendi içindeki fitness hesaplaması içindir.
        optimizer = optimizer_class(n_particles=N_PARTICLES, n_iterations=N_ITERATIONS, cv=3)
        optimizer.fit(X_train_scaled, y_train)

        # PSO'nun seçtiği en iyi özellik maskesini (örn: [1, 0, 1, ...]) al.
        selected_mask = optimizer.best_solution_
        selected_indices = [i for i, val in enumerate(selected_mask) if val == 1]

        # Nadir durum kontrolü: Eğer PSO hiçbir özellik seçmezse bu turu başarısız say.
        if len(selected_indices) == 0:
            fold_accuracies.append(0.0)
            fold_features.append(0)
            continue

        # --- TEST AŞAMASI (DIŞ DÖNGÜ) ---
        # Veri setlerini sadece seçilen sütunlara indirge.
        X_train_selected = X_train_scaled[:, selected_indices]
        X_test_selected = X_test_scaled[:, selected_indices]

        # KNN Sınıflandırıcı ile final başarımı ölç.
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train_selected, y_train)
        y_pred = clf.predict(X_test_selected)

        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        fold_features.append(len(selected_indices))

    # Tüm fold'ların ortalamasını döndür (Genel Başarım)
    return np.mean(fold_accuracies), np.mean(fold_features)


def run_experiment(dataset_name, X, y):
    """
    Belirli bir veri seti için tüm algoritmaları N_RUNS kadar çalıştırır.
    İstatistikleri toplar.
    """
    results = []

    # Boxplot çizimi için ham verileri saklayacağımız liste
    raw_data_for_plot = []

    algorithms = {
        "Standard_PSO": BinaryPSO,
        "Chaotic_PSO": ChaoticBinaryPSO
    }

    print(f"\n>>> VERİ SETİ İŞLENİYOR: {dataset_name} <<<")

    for algo_name, AlgoClass in algorithms.items():
        print(f"   --- Algoritma: {algo_name} ({N_RUNS} Tekrar, {N_FOLDS}-Fold CV) ---")

        accuracies = []
        feature_counts = []
        times = []

        # İstatistiksel güvenilirlik için 10 tekrar (Runs)
        for run in range(N_RUNS):
            print(f"      Run {run + 1}/{N_RUNS}...", end="")

            start_time = time.time()

            # Tek bir run içinde aslında 5 kez (Fold) eğitim yapılır ve ortalaması alınır.
            acc, n_feats = evaluate_with_cv(AlgoClass, X, y)

            duration = time.time() - start_time

            accuracies.append(acc)
            feature_counts.append(n_feats)
            times.append(duration)

            # Grafik için ham veriyi kaydet (Her run'ın sonucu)
            raw_data_for_plot.append({
                "Dataset": dataset_name,
                "Method": algo_name,
                "Accuracy": acc,
                "Feature_Count": n_feats
            })

            print(f" Bitti. Ort. Acc: %{acc * 100:.2f} | Feat: {n_feats:.1f}")

        # İstatistikleri Hesapla (Ortalama, Standart Sapma, En İyi)
        stats = {
            "Dataset": dataset_name,
            "Method": algo_name,
            "Mean_Accuracy": np.mean(accuracies),
            "Best_Accuracy": np.max(accuracies),
            "Std_Dev": np.std(accuracies),  # Kararlılık ölçüsü (Düşük olması iyidir)
            "Mean_Features": np.mean(feature_counts),
            "Mean_Time(s)": np.mean(times)
        }
        results.append(stats)

    return results, raw_data_for_plot


def create_plots(raw_df):
    """
    Toplanan sonuçları Seaborn kütüphanesi ile görselleştirir.
    """
    sns.set_style("whitegrid")

    # 1. DOĞRULUK DAĞILIMI (BOXPLOT)
    # Boxplot, algoritmanın kararlılığını gösterir. Kutunun boyu kısaysa algoritma kararlıdır.
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Dataset", y="Accuracy", hue="Method", data=raw_df, palette="Set2")
    plt.title("Yöntemlerin Doğruluk Dağılımı (10 Tekrar)", fontsize=14)
    plt.ylabel("Doğruluk (Accuracy)", fontsize=12)
    plt.savefig(os.path.join(PLOT_DIR, "Final_Accuracy_Distribution.png"), dpi=300)
    plt.close()

    # 2. ÖZELLİK SAYISI KARŞILAŞTIRMASI (BARPLOT)
    # Hangi yöntemin daha az özellik seçerek başarılı olduğunu gösterir.
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Dataset", y="Feature_Count", hue="Method", data=raw_df, palette="viridis", errorbar="sd")
    plt.title("Seçilen Ortalama Özellik Sayısı", fontsize=14)
    plt.ylabel("Özellik Sayısı", fontsize=12)
    plt.savefig(os.path.join(PLOT_DIR, "Final_Feature_Reduction.png"), dpi=300)
    plt.close()

    print(f"\n[BİLGİ] Grafikler '{PLOT_DIR}' klasörüne kaydedildi.")


if __name__ == "__main__":
    all_results = []
    all_raw_data = []

    # Tanımlı veri setleri üzerinde döngü
    for ds_name, ds_path in datasets.items():
        X, y = load_data(ds_path)
        if X is not None:
            ds_results, ds_raw_data = run_experiment(ds_name, X, y)
            all_results.extend(ds_results)
            all_raw_data.extend(ds_raw_data)

    # Sonuçları DataFrame'e çevir
    final_df = pd.DataFrame(all_results)
    raw_df = pd.DataFrame(all_raw_data)  # Grafikler için ham veri

    # CSV Çıktısı için sütun sırasını düzenle
    cols = ["Dataset", "Method", "Mean_Accuracy", "Best_Accuracy", "Std_Dev", "Mean_Features", "Mean_Time(s)"]
    final_df = final_df[cols]

    # Sonuçları Kaydet
    save_path = os.path.join(project_root, RESULTS_FILE)
    final_df.to_csv(save_path, index=False)

    # Grafikleri oluştur
    create_plots(raw_df)

    print("\n" + "=" * 60)
    print("TÜM DENEYLER TAMAMLANDI!")
    print(f"Sonuçlar kaydedildi: {save_path}")
    print("=" * 60)
    print(final_df)