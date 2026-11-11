import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

CSV = sys.argv[1] if len(sys.argv) > 1 else "HousingData.csv"
MAX_CLUSTERS = 10          
SEED = 42
CATEGORICAS = ["CHAS"]     
NUMERICAS = ["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS",
             "RAD", "TAX", "PTRATIO", "B", "LSTAT"]   

df = pd.read_csv(CSV)

#  tratar nulos
medianas = df[NUMERICAS].median()
moda_chas = df["CHAS"].mode()[0]
df[NUMERICAS] = df[NUMERICAS].fillna(medianas)
df["CHAS"] = df["CHAS"].fillna(moda_chas)

# get_dummies
df["CHAS"] = df["CHAS"].map({0.0: "nao", 1.0: "sim"})
dummies = pd.get_dummies(df[CATEGORICAS], dtype=float)      
colunas_dummies = list(dummies.columns)

# normalizacao
medias = df[NUMERICAS].mean()
desvios = df[NUMERICAS].std(ddof=0)
X_num = (df[NUMERICAS] - medias) / desvios
X = pd.concat([X_num, dummies], axis=1)
colunas_modelo = list(X.columns)

# metodo do cotovelo
ks = list(range(1, MAX_CLUSTERS + 1))
inercias = [KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X).inertia_ for k in ks]

def achar_cotovelo(ks, ys):
    """k do ponto mais distante da reta entre o primeiro e o último ponto."""
    p = np.column_stack([ks, ys]).astype(float)
    p = (p - p.min(0)) / (p.max(0) - p.min(0))
    v, w = p[-1] - p[0], p - p[0]
    d = np.abs(v[0] * w[:, 1] - v[1] * w[:, 0]) / np.linalg.norm(v)
    return ks[int(np.argmax(d))]

k_otimo = achar_cotovelo(ks, inercias)
print("Inercia por k:", {k: round(i, 1) for k, i in zip(ks, inercias)})
print(f"Numero otimo de clusters (cotovelo): {k_otimo}")

plt.figure(figsize=(7, 4))
plt.plot(ks, inercias, "o-")
plt.axvline(k_otimo, color="red", ls="--", label=f"k otimo = {k_otimo}")
plt.xlabel("Numero de clusters (k)"); plt.ylabel("Inercia")
plt.title("Metodo do cotovelo"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("cotovelo.png", dpi=120)

# treino
kmeans = KMeans(n_clusters=k_otimo, n_init=10, random_state=SEED).fit(X)
medv = pd.read_csv(CSV).groupby(kmeans.labels_)["MEDV"].mean().round(2).to_dict()

joblib.dump({"kmeans": kmeans, "numericas": NUMERICAS, "categoricas": CATEGORICAS,
             "medias": medias, "desvios": desvios, "medianas": medianas,
             "moda_chas": moda_chas, "colunas_modelo": colunas_modelo,
             "medv_por_cluster": medv}, "modelo_kmeans.joblib")
print("Modelo salvo em modelo_kmeans.joblib | tamanhos:",
      {int(c): int(n) for c, n in zip(*np.unique(kmeans.labels_, return_counts=True))})
