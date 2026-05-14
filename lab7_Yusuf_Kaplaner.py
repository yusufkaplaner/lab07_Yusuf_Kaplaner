# ============================================================
# YZM0206 - Laboratuvar 7: KNN ile Müşteri Sınıflandırması
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. VERİYİ YÜKLE VE İNCELE
# -------------------------------------------------------
df = pd.read_csv('teleCust1000t.csv')
print("=== VERİ SETI ===")
print(df.head())
print(f"\nŞekil: {df.shape}")
print(f"\nSütunlar: {list(df.columns)}")

print("\n=== SINIF DAĞILIMI (custcat) ===")
print(df['custcat'].value_counts())
# Yorum: Sınıflar yaklaşık dengeli (~250'şer) → model için uygun bir veri seti

# -------------------------------------------------------
# 2. VERİYİ HAZIRLA
# -------------------------------------------------------
# Bağımsız değişkenler (features) ve bağımlı değişken (hedef)
X = df.drop('custcat', axis=1).values   # numpy array
y = df['custcat'].values                # numpy array

# Normalizasyon (KNN mesafe tabanlı çalıştığı için ÇOK ÖNEMLİ)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train / Test bölme (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4
)
print(f"\nEğitim seti: {X_train.shape[0]} örnek")
print(f"Test seti  : {X_test.shape[0]} örnek")

# -------------------------------------------------------
# 3a. MODEL — K = 4
# -------------------------------------------------------
print("\n" + "="*50)
print("K = 4 ile KNN Modeli")
print("="*50)

knn4 = KNeighborsClassifier(n_neighbors=4)
knn4.fit(X_train, y_train)

y_pred4 = knn4.predict(X_test)
acc4 = accuracy_score(y_test, y_pred4)
print(f"Test Accuracy (K=4): {acc4:.4f}")
print(classification_report(y_test, y_pred4))

# -------------------------------------------------------
# 3b. MODEL — K = 6
# -------------------------------------------------------
print("="*50)
print("K = 6 ile KNN Modeli")
print("="*50)

knn6 = KNeighborsClassifier(n_neighbors=6)
knn6.fit(X_train, y_train)

y_pred6 = knn6.predict(X_test)
acc6 = accuracy_score(y_test, y_pred6)
print(f"Test Accuracy (K=6): {acc6:.4f}")
print(classification_report(y_test, y_pred6))

print(f"\nK=4 Accuracy: {acc4:.4f}")
print(f"K=6 Accuracy: {acc6:.4f}")
print(f"Fark: {abs(acc4 - acc6):.4f} → {'K=4 daha iyi' if acc4 > acc6 else 'K=6 daha iyi' if acc6 > acc4 else 'Eşit'}")

# -------------------------------------------------------
# 4. OPTİMUM K DEĞERİNİ BUL (K = 1..10)
# -------------------------------------------------------
print("\n" + "="*50)
print("Optimum K Değeri Arama (1–10)")
print("="*50)

train_scores = []
test_scores  = []
k_range = range(1, 11)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, knn.predict(X_train)))
    test_scores.append(accuracy_score(y_test,  knn.predict(X_test)))

for k, tr, te in zip(k_range, train_scores, test_scores):
    print(f"K={k:2d}  |  Train: {tr:.4f}  |  Test: {te:.4f}")

best_k = k_range[test_scores.index(max(test_scores))]
print(f"\n★ Optimum K = {best_k}  (Test Accuracy: {max(test_scores):.4f})")

# Grafik
plt.figure(figsize=(9, 5))
plt.plot(k_range, train_scores, marker='o', label='Train Accuracy', linewidth=2)
plt.plot(k_range, test_scores,  marker='s', label='Test Accuracy',  linewidth=2)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Optimum K={best_k}')
plt.title('K Değerine Göre Model Doğruluğu')
plt.xlabel('K (Komşu Sayısı)')
plt.ylabel('Accuracy')
plt.xticks(k_range)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('k_vs_accuracy.png', dpi=150)
plt.show()
print("Grafik 'k_vs_accuracy.png' olarak kaydedildi.")

# -------------------------------------------------------
# 5. UZAKLIK ÖLÇÜTLERİ KARŞILAŞTIRMASI
# -------------------------------------------------------
print("\n" + "="*50)
print(f"Uzaklık Ölçütü Karşılaştırması (K={best_k})")
print("="*50)

metrics = {
    'Euclidean  (p=2)' : {'metric': 'minkowski', 'p': 2},
    'Manhattan  (p=1)' : {'metric': 'minkowski', 'p': 1},
    'Minkowski  (p=3)' : {'metric': 'minkowski', 'p': 3},
    'Chebyshev'        : {'metric': 'chebyshev'},
}

metric_results = {}
for name, params in metrics.items():
    knn_m = KNeighborsClassifier(n_neighbors=best_k, **params)
    knn_m.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn_m.predict(X_test))
    metric_results[name] = acc
    print(f"{name:22s}  →  Test Accuracy: {acc:.4f}")

best_metric = max(metric_results, key=metric_results.get)
print(f"\n★ En iyi uzaklık ölçütü: {best_metric}  ({metric_results[best_metric]:.4f})")

# Grafik
plt.figure(figsize=(8, 4))
plt.bar(metric_results.keys(), metric_results.values(), color=['steelblue','coral','mediumseagreen','orchid'])
plt.title(f'Uzaklık Ölçütüne Göre Test Accuracy (K={best_k})')
plt.ylabel('Accuracy')
plt.ylim(0.25, 0.55)
plt.tight_layout()
plt.savefig('metric_comparison.png', dpi=150)
plt.show()
print("Grafik 'metric_comparison.png' olarak kaydedildi.")

# -------------------------------------------------------
# 6. ÖZET RAPOR
# -------------------------------------------------------
print("\n" + "="*50)
print("ÖZET")
print("="*50)
print(f"K=4  Test Accuracy : {acc4:.4f}")
print(f"K=6  Test Accuracy : {acc6:.4f}")
print(f"Optimum K          : {best_k}  ({max(test_scores):.4f})")
print(f"En iyi uzaklık     : {best_metric}  ({metric_results[best_metric]:.4f})")