# -*- coding: utf-8 -*-
# ======================================================================
# Análisis simple, comentado y con K-Means para el escenario de 15 tiendas
# Requisitos: pandas, numpy, matplotlib, seaborn, scikit-learn
# Uso:
#   1) Coloca este archivo en la misma carpeta que 'escenario_tiendas_15.csv'
#   2) Ejecuta: python analisis_tiendas_simple_comentado_kmeans.py
#   3) Se guardarán imágenes y verás resultados/justificaciones en consola
# ======================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------------------------------------------------
# 1) Cargar datos
# -------------------------------------------------------------
df = pd.read_csv("escenario_gimnasios_15.csv", encoding="utf-8-sig")

print("Forma (filas, columnas):", df.shape)
print("\nTipos de datos:\n", df.dtypes)
print("\nValores faltantes por columna:\n", df.isna().sum())

# -------------------------------------------------------------
# 2) Estadística descriptiva básica + indicadores
# -------------------------------------------------------------
desc = df.select_dtypes("number").describe()
pd.set_option("display.max_rows", None)       
pd.set_option("display.max_columns", None) 
print("\nDescripción numérica:\n", desc)

var1 = "Ingresos_Mensuales"
var2 = "Precio_Membresia"
print("\n=== Indicadores de tendencia central ===")
print(f"{var1} -> media: {df[var1].mean(skipna=True):.2f}, mediana: {df[var1].median(skipna=True):.2f}, moda: {df[var1].mode(dropna=True).iloc[0] if not df[var1].mode(dropna=True).empty else None}")
print(f"{var2} -> media: {df[var2].mean(skipna=True):.2f}, mediana: {df[var2].median(skipna=True):.2f}, moda: {df[var2].mode(dropna=True).iloc[0] if not df[var2].mode(dropna=True).empty else None}")

# -------------------------------------------------------------
# 3) Gráficos básicos (boxplot/hist/boxplot extra/heatmap)
# -------------------------------------------------------------
plt.figure()
df[var1].plot(kind="box", title=f"Boxplot - {var1}")
plt.tight_layout(); plt.show(); plt.close()

plt.figure()
df[var1].plot(kind="hist", bins=10, title=f"Histograma - {var1}")
plt.xlabel("Valor"); plt.ylabel("Frecuencia")
plt.tight_layout(); plt.show(); plt.close()

plt.figure()
df[var2].plot(kind="box", title=f"Boxplot - {var2}")
plt.tight_layout(); plt.show(); plt.close()

num = df.select_dtypes("number")
corr = num.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", square=True)
plt.title("Heatmap de correlaciones (variables numéricas)")
plt.tight_layout(); plt.show(); plt.close()

# -------------------------------------------------------------
# 4) Preparación para clustering K-Means
# -------------------------------------------------------------
# ¿Qué variables usar? En clustering k-means conviene usar SOLO numéricas y
# evitar identificadores o categorías codificadas como texto.
# Además, podemos eliminar variables que no aporten o que estén altamente
# correlacionadas para reducir ruido.
#
# Regla simple:
#   - Quitamos columnas no numéricas o identificadores: 'Tienda', 'Region'
#   - Detectamos pares con |correlación|>=0.90 y eliminamos UNA de cada par
#     (nos quedamos con la primera y eliminamos la segunda del par)
#
features_all = df.select_dtypes("number").copy()

# Imputación simple de faltantes (media) para no complicar: k-means no permite NaN
features_all = features_all.fillna(features_all.mean(numeric_only=True))

# Encontrar pares altamente correlacionados
high_corr_threshold = 0.90
corr_matrix = features_all.corr()
to_drop = set()
cols = corr_matrix.columns.tolist()

for i, c1 in enumerate(cols):
    for c2 in cols[i+1:]:
        r = corr_matrix.loc[c1, c2]
        if pd.notna(r) and abs(r) >= high_corr_threshold:
            # marcamos c2 para eliminarlo (regla simple)
            to_drop.add(c2)

# Eliminamos variables: identificadores no numéricos ya no están,
# y eliminamos los de alta correlación definidos en to_drop
X = features_all.drop(columns=list(to_drop)) if to_drop else features_all.copy()

print("\n=== Selección de variables para clustering ===")
print("Columns usadas:", list(X.columns))
if to_drop:
    print("Eliminadas por alta correlación (|r|>=0.90):", list(to_drop))
else:
    print("No se eliminaron variables por alta correlación con el umbral 0.90.")

# -------------------------------------------------------------
# 5) Determinar un valor de k (Elbow + Silhouette)
# -------------------------------------------------------------
# Escalamos las variables (media 0, var 1) para que ninguna domine por magnitud
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# Probamos k de 2 a 6 dada la muestra pequeña (15 tiendas)
k_values = range(2, 6)
inertias = []
sil_scores = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    inertias.append(km.inertia_)  # suma de distancias cuadráticas dentro de clusters
    sil = silhouette_score(Xs, labels)
    sil_scores.append(sil)

# Guardamos gráficos de apoyo (codo e índice de silueta)
plt.figure()
plt.plot(list(k_values), inertias, marker="o")
plt.xticks(list(k_values))
plt.title("Método del codo (inertia vs k)")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.tight_layout(); plt.show(); plt.close()

plt.figure()
plt.plot(list(k_values), sil_scores, marker="o")
plt.xticks(list(k_values))
plt.title("Índice de Silueta vs k")
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.tight_layout(); plt.show(); plt.close()

# Elegimos k con el mejor silhouette (regla muy simple)
best_k = int(k_values[np.argmax(sil_scores)])
print(f"\nSugerencia automática de k (por máxima silueta en 2..6): k={best_k}")
print("Inertias por k:", dict(zip(k_values, inertias)))
print("Siluetas por k:", dict(zip(k_values, sil_scores)))

# -------------------------------------------------------------
# 6) K-Means con el k elegido y análisis de centros
# -------------------------------------------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(Xs)

# Centros en espacio estandarizado y en el espacio original
centers_std = kmeans.cluster_centers_
centers_orig = scaler.inverse_transform(centers_std)  # para interpretar en unidades originales

centers_df = pd.DataFrame(centers_orig, columns=X.columns)
centers_df.index = [f"Centro_{i}" for i in range(best_k)]
pd.set_option("display.max_rows", None)       
pd.set_option("display.max_columns", None) 
print("\n=== Centros de K-Means (en escala original) ===")
print(centers_df.round(2))

# Distancias euclidianas entre centros (en escala original para interpretación)
# Nota: también se puede usar la escala estandarizada para ver separación geométrica pura
def pairwise_distances(M):
    # M: matriz (k x d)
    # devuelve matriz (k x k) de distancias euclidianas
    diffs = M[:, None, :] - M[None, :, :]
    d = np.sqrt((diffs**2).sum(axis=2))
    return d

dist_centers = pairwise_distances(centers_orig)
dist_df = pd.DataFrame(dist_centers, index=centers_df.index, columns=centers_df.index)
print("\n=== Distancias entre centros (escala original) ===")
print(dist_df.round(2))

# -------------------------------------------------------------
# 7) Q&A Justificado basándose en resultados
# -------------------------------------------------------------
print("\n================= RESPUESTAS GUIADAS =================")

# ¿Crees que estos centros puedan ser representativos de los datos? ¿Por qué?
sil_best = max(sil_scores)
if sil_best >= 0.5:
    rep_txt = "Sí, la calidad de separación es razonable (silhouette >= 0.5)."
elif sil_best >= 0.35:
    rep_txt = "Medianamente; la separación es moderada (silhouette entre 0.35 y 0.5)."
else:
    rep_txt = "Limitado; la separación es débil (silhouette < 0.35)."

print(f"\n1) Representatividad de centros: {rep_txt} "
      f"Sugerencia: revisar variables y posibles outliers. Silhouette óptima: {sil_best:.2f} con k={best_k}.")

# ¿Cómo obtuviste el valor de k a usar?
print("\n2) Elección de k:")
print("- Se usó el método del codo (inertia vs k) y el índice de silueta.")
print(f"- Se eligió k={best_k} por máxima silueta en el rango 2..6.")

# ¿Centros más representativos con k más alto o más bajo?
print("\n3) ¿Más representativos con k más alto o más bajo?")
print("- En general, aumentar k reduce la inercia (clusters más compactos), pero puede sobre-segmentar.")
print("- El índice de silueta ayuda a balancear cohesión/separación; usar el pico de silueta evita sobreajuste.")

# ¿Qué distancia tienen los centros entre sí? ¿Alguno muy cercano?
min_offdiag = dist_df.replace(0, np.nan).min().min()
print("\n4) Distancias entre centros:")
print(dist_df.round(2))
print(f"- Distancia mínima entre centros (excluyendo 0): {min_offdiag:.2f}. "
      "Si es muy pequeña frente a otras, indica clusters similares/solapados.")

# ¿Qué pasa con outliers en boxplot?
print("\n5) Efecto de outliers:")
print("- K-Means minimiza la suma de distancias cuadráticas, por lo que es sensible a outliers.")
print("- Muchos outliers pueden arrastrar los centros y empeorar la silueta.")
print("- Recomendación: detectar/gestionar outliers (winsorizar, imputar, escalar robustamente) antes de agrupar.")

# ¿Qué se puede decir de los datos con base en los centros?
print("\n6) Interpretación de centros:")
print("- Revise 'centers_df' para ver el perfil de cada cluster (por ejemplo, tráfico alto, ticket alto, etc.).")
print("- Compare 'Ventas_Mensuales' promedio entre centros, junto con variables operativas, para caracterizar segmentos.")

# -------------------------------------------------------------
# 8) (Opcional) Visualización simple 2D con PCA para ver clusters
#     Mantenerlo simple: dibujamos puntos por cluster; sólo orientativo.
# -------------------------------------------------------------
try:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xs)

    plt.figure()
    for lab in np.unique(labels):
        idx = labels == lab
        plt.scatter(X2[idx, 0], X2[idx, 1], label=f"Cluster {lab}")
    plt.title("Clusters K-Means (PCA 2D)")
    plt.xlabel("Componente 1"); plt.ylabel("Componente 2")
    plt.legend()
    plt.tight_layout(); plt.show(); plt.close()
except Exception as e:
    print("PCA opcional no disponible:", e)

print("\nFin.")