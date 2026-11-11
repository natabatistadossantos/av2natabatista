import json
import sys
import joblib
import numpy as np
import pandas as pd

m = joblib.load("modelo_kmeans.joblib")
kmeans, NUM = m["kmeans"], m["numericas"]

def prever(dados: dict):
    nova = pd.DataFrame([dados])
    
    for c in NUM:
        nova[c] = pd.to_numeric(nova.get(c), errors="coerce")
    nova[NUM] = nova[NUM].fillna(m["medianas"])
    chas = dados.get("CHAS")
    chas = m["moda_chas"] if chas in (None, "") else float(chas)
    nova["CHAS"] = "sim" if chas == 1 else "nao"
    dummies = pd.get_dummies(nova[m["categoricas"]], dtype=float)
    X_num = (nova[NUM] - m["medias"]) / m["desvios"]
    X = pd.concat([X_num, dummies], axis=1).reindex(columns=m["colunas_modelo"], fill_value=0.0)
    cluster = int(kmeans.predict(X)[0])
    centro = pd.Series(kmeans.cluster_centers_[cluster], index=m["colunas_modelo"])
    centroide = centro[NUM] * m["desvios"] + m["medias"]
    centroide = pd.concat([centroide, centro.drop(NUM)])  
    return cluster, centroide

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dados = json.loads(sys.argv[1])
    else:
        dados = {}
        print("Dados da residencia (Enter = nao sei):")
        for c in NUM + ["CHAS"]:
            v = input(f"  {c}{' (0/1)' if c == 'CHAS' else ''}: ").strip().replace(",", ".")
            if v:
                dados[c] = float(v)
    cluster, centroide = prever(dados)
    print(f"\nA residencia pertence ao cluster {cluster}.")
    print("Centroide desnormalizado (CHAS_* = proporcao do cluster):")
    print(centroide.round(3).to_string())
    print(f"MEDV medio do cluster (x US$1000): {m['medv_por_cluster'][cluster]}")
