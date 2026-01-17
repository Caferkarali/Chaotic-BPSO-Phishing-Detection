import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt  # Grafik çizimi için eklendi
import os  # Klasör işlemleri için eklendi


class BinaryPSO:
    """
    Oltalama Tespiti için Binary Particle Swarm Optimization (BPSO) Sınıfı.
    Amaç: Sınıflandırma hatasını minimize eden en iyi öznitelik alt kümesini bulmak.
    """

    def __init__(self, n_particles=30, n_iterations=50, w=0.7, c1=2.0, c2=2.0, cv=3):
        # --- HİPERPARAMETRELERİN ANLAMI ---
        # n_particles: Sürüdeki ajan sayısı. Ne kadar çoksa arama alanı o kadar iyi taranır ama yavaşlar.
        # w (Atalet Ağırlığı): Parçacığın kendi hızını koruma eğilimi. Yüksekse 'keşif' (exploration), düşükse 'sömürü' (exploitation) yapar.
        # c1 (Bilişsel): "Benim bulduğum en iyi yer"e gitme isteği.
        # c2 (Sosyal): "Sürünün bulduğu en iyi yer"e gitme isteği.
        # cv: Cross-Validation katman sayısı (3 idealdir, 5 veya 10 yavaşlatabilir).

        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.cv = cv

        # Sonuçları saklamak için değişkenler
        self.convergence_curve_ = []  # Her iterasyondaki en iyi skoru tutar (Grafik için)
        self.best_solution_ = None  # En son bulunan özellik maskesi (Örn: [1, 0, 1, ...])
        self.best_score_ = 0.0  # Ulaşılan maksimum doğruluk

    def sigmoid(self, x):
        """


[Image of sigmoid function graph]

        Transfer Fonksiyonu: Hız değerini [0, 1] olasılığına dönüştürür.
        Standart PSO sürekli (ondalıklı) değerlerle çalışır. Ancak özellik seçimi VAR/YOK (1/0)
        olduğu için bu fonksiyon köprü görevi görür.
        """
        return 1 / (1 + np.exp(-x))

    def compute_fitness(self, position, X, y):
        """
        Amaç Fonksiyonu (Fitness Function):
        Bir parçacığın önerdiği özellik kombinasyonunun ne kadar 'kaliteli' olduğunu ölçer.
        Kalite = KNN Algoritmasının Doğruluk Oranı (Accuracy).
        """
        # Pozisyon vektöründeki 1 olan indeksleri al (Seçilen özellikler)
        selected_indices = [i for i, val in enumerate(position) if val == 1]

        # Eğer parçacık hiçbir özellik seçmediyse, ceza puanı ver (0.0).
        if len(selected_indices) == 0:
            return 0.0

        # Veri setini sadece seçilen sütunlara indirge
        X_subset = X[:, selected_indices]

        # Sınıflandırıcı olarak KNN kullanıyoruz (Hızlı olduğu için wrapper metodlarda standarttır).
        clf = KNeighborsClassifier(n_neighbors=5)

        # Cross Validation ile başarı ölçümü (Overfitting'i önlemek için veriyi 3'e bölüp test eder)
        scores = cross_val_score(clf, X_subset, y, cv=self.cv, scoring='accuracy')

        # 3 testin ortalamasını döndür
        return scores.mean()

    def fit(self, X, y):
        """
        Algoritmayı çalıştıran ana motor.
        """
        n_features = X.shape[1]

        # 1. BAŞLATMA (Initialization)
        # Konumlar: Rastgele 0 veya 1 (Özellik seçili mi değil mi?)
        particles_pos = np.random.randint(0, 2, (self.n_particles, n_features))

        # Hızlar: Parçacıkların değişim isteği (-1 ile 1 arası rastgele)
        particles_vel = np.random.uniform(-1, 1, (self.n_particles, n_features))

        # Hafıza: Her parçacığın kendi gördüğü en iyi konum (Pbest)
        pbest_pos = particles_pos.copy()
        pbest_score = np.zeros(self.n_particles)

        # Lider: Sürünün gördüğü en iyi konum (Gbest)
        gbest_pos = np.zeros(n_features)
        gbest_score = 0.0

        # Başlangıçta herkesin durumunu hesapla
        for i in range(self.n_particles):
            score = self.compute_fitness(particles_pos[i], X, y)
            pbest_score[i] = score

            if score > gbest_score:
                gbest_score = score
                gbest_pos = particles_pos[i].copy()

        # 2. İTERASYON DÖNGÜSÜ
        print(f"PSO Başlatıldı: {self.n_particles} Parçacık, {self.n_iterations} İterasyon")

        for t in range(self.n_iterations):
            for i in range(self.n_particles):
                #
                # a. Hız Güncelleme (Matematiksel Formül)
                # Yeni Hız = (Eski Hız * Atalet) + (Kendi En İyisine Çekim) + (Lidere Çekim)
                r1 = np.random.rand(n_features)  # Rastgelelik faktörü 1
                r2 = np.random.rand(n_features)  # Rastgelelik faktörü 2

                particles_vel[i] = (self.w * particles_vel[i] +
                                    self.c1 * r1 * (pbest_pos[i] - particles_pos[i]) +
                                    self.c2 * r2 * (gbest_pos - particles_pos[i]))

                # b. Konum Güncelleme (Sigmoid Transfer ile Binary Dönüşüm)
                # Hesaplanan hızı 0-1 arası bir olasılığa çevir -> S(v)
                sigmoid_v = self.sigmoid(particles_vel[i])

                # Eğer rastgele sayı olasılıktan küçükse özelliği seç (1), değilse seçme (0)
                particles_pos[i] = np.where(np.random.rand(n_features) < sigmoid_v, 1, 0)

                # c. Yeni Pozisyonun Değerlendirilmesi
                current_score = self.compute_fitness(particles_pos[i], X, y)

                # Kişisel rekoru kırıldı mı?
                if current_score > pbest_score[i]:
                    pbest_score[i] = current_score
                    pbest_pos[i] = particles_pos[i].copy()

                # Dünya rekoru (Global) kırıldı mı?
                if current_score > gbest_score:
                    gbest_score = current_score
                    gbest_pos = particles_pos[i].copy()

            # Her iterasyon sonunda en iyi skoru listeye ekle (Grafik için)
            self.convergence_curve_.append(gbest_score)
            print(f"İterasyon {t + 1}/{self.n_iterations} -> En İyi Başarı: %{gbest_score * 100:.2f}")

        # En son bulunan en iyi çözümü kaydet
        self.best_solution_ = gbest_pos
        self.best_score_ = gbest_score

        return self


class ChaoticBinaryPSO(BinaryPSO):
    """
    Standart Binary PSO'nun geliştirilmiş versiyonu.

    YENİLİK:
    Standart PSO başlangıç konumlarını tamamen rastgele (np.random) seçer.
    Bu bazen parçacıkların birbirine çok yakın başlamasına ve "Yerel Tuzaklara" (Local Minima) düşmesine neden olur.

    ÇÖZÜM:
    'Logistic Map' adı verilen Kaotik Harita kullanılarak başlangıç konumları belirlenir.
    Bu, arama uzayının çok daha homojen ve etkili taranmasını sağlar.
    """

    def __init__(self, n_particles=30, n_iterations=50, w=0.7, c1=2.0, c2=2.0, cv=3):
        # Üst sınıfın (BinaryPSO) özelliklerini miras al
        super().__init__(n_particles, n_iterations, w, c1, c2, cv)

    def fit(self, X, y):
        """
        Kaotik Başlatma (Chaotic Initialization) ile fit işlemi
        """
        n_features = X.shape[1]

        # --- KAOTİK HARİTA (LOGISTIC MAP) ÜRETİMİ ---
        #
        # Formül: x(t+1) = mu * x(t) * (1 - x(t))
        # mu = 4.0 olduğunda sistem tam kaotik davranış sergiler (rastgele görünür ama deterministiktir).

        chaotic_map = np.zeros((self.n_particles, n_features))
        x = np.random.rand(self.n_particles, n_features)  # İlk tohum değeri
        mu = 4.0

        # Kaotik seriyi ısıtma (Warm-up):
        # İlk üretilen sayılar bazen kararlı olabilir, kaosun oturması için ilk 100 adımı boşa çeviriyoruz.
        for _ in range(100):
            x = mu * x * (1 - x)

        # Gerçek başlangıç değerlerini üret
        # Kaotik değerler (0.0 - 1.0 arası) 0.5 eşiği ile 0 veya 1'e dönüştürülür.
        particles_pos = np.where(x > 0.5, 1, 0)

        # Hızları da kaotik başlatmak, parçacıklara ilk enerjiyi daha iyi verir.
        particles_vel = (x - 0.5) * 2  # Değerleri -1 ile 1 arasına çeker

        # --- BURADAN SONRASI STANDART PSO İLE AYNI ---
        # (Miras alınan yapı olduğu için mantık değişmez, sadece başlangıç noktaları değişmiştir)

        pbest_pos = particles_pos.copy()
        pbest_score = np.zeros(self.n_particles)

        gbest_pos = np.zeros(n_features)
        gbest_score = 0.0

        # İlk Fitness Değerleri
        print(f"Kaotik PSO Başlatıldı (Logistic Map): {self.n_particles} Parçacık")

        for i in range(self.n_particles):
            score = self.compute_fitness(particles_pos[i], X, y)
            pbest_score[i] = score
            if score > gbest_score:
                gbest_score = score
                gbest_pos = particles_pos[i].copy()

        # İterasyon Döngüsü
        for t in range(self.n_iterations):
            for i in range(self.n_particles):
                r1 = np.random.rand(n_features)
                r2 = np.random.rand(n_features)

                # Standart Hız Güncelleme
                particles_vel[i] = (self.w * particles_vel[i] +
                                    self.c1 * r1 * (pbest_pos[i] - particles_pos[i]) +
                                    self.c2 * r2 * (gbest_pos - particles_pos[i]))

                # Sigmoid ile Konum Güncelleme
                sigmoid_v = self.sigmoid(particles_vel[i])
                particles_pos[i] = np.where(np.random.rand(n_features) < sigmoid_v, 1, 0)

                # Fitness Kontrol
                current_score = self.compute_fitness(particles_pos[i], X, y)

                if current_score > pbest_score[i]:
                    pbest_score[i] = current_score
                    pbest_pos[i] = particles_pos[i].copy()

                if current_score > gbest_score:
                    gbest_score = current_score
                    gbest_pos = particles_pos[i].copy()
                    print(f"   -> Yeni Global En İyi (Gbest): %{gbest_score * 100:.2f} (İterasyon {t + 1})")

            self.convergence_curve_.append(gbest_score)

        self.best_solution_ = gbest_pos
        self.best_score_ = gbest_score
        return self


# ========================================================
# YARDIMCI GRAFİK FONKSİYONU
# ========================================================
def plot_convergence(optimizer, dataset_name="Dataset", algorithm_name="PSO", save_folder="plots"):
    """
    Bu fonksiyon, eğitilmiş modelin öğrenme eğrisini çizer ve kaydeder.
    Grafik yukarı doğru gidiyorsa model öğreniyor demektir.
    """
    if not hasattr(optimizer, 'convergence_curve_') or not optimizer.convergence_curve_:
        print("UYARI: Optimizer içinde grafik verisi bulunamadı. Model henüz eğitilmemiş olabilir.")
        return

    # Klasör kontrolü (Yoksa oluştur)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Grafik Çizimi
    plt.figure(figsize=(10, 6))
    plt.plot(optimizer.convergence_curve_, marker='o', linestyle='-', linewidth=2, markersize=4, color='blue')

    plt.title(f'{algorithm_name} - Yakınsama Eğrisi ({dataset_name})', fontsize=14)
    plt.xlabel('İterasyon Sayısı', fontsize=12)
    plt.ylabel('En İyi Doğruluk (Accuracy)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Dosya ismini güvenli hale getir (boşlukları _ yap)
    safe_name = f"{algorithm_name}_{dataset_name}_convergence.png".replace(" ", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(save_folder, safe_name)

    plt.savefig(save_path, dpi=300)
    plt.close()  # Belleği temizle

    print(f"   -> [GRAFİK KAYDEDİLDİ]: {save_path}")