# -*- coding: utf-8 -*-
# =============================================================
# Análisis muy simple y comentado para 15 tiendas
# Requisitos: pandas, matplotlib, seaborn
# Uso:
#   1) Coloca este archivo en la misma carpeta que 'escenario_tiendas_15.csv'
#   2) Ejecuta: python analisis_tiendas_simple_comentado.py
#   3) Se guardarán las imágenes y verás las métricas impresas en consola
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# 1) Cargar datos
# -------------------------------------------------------------
# Leemos el CSV con los datos del caso (15 tiendas, variables explícitas)
df = pd.read_csv("escenario_tiendas_15.csv", encoding="utf-8-sig")

# Vistazo rápido al tamaño de la tabla y a los tipos de datos
print("Forma (filas, columnas):", df.shape)
print("\nTipos de datos:\n", df.dtypes)

# Conteo de valores faltantes (NaN) por columna
print("\nValores faltantes por columna:\n", df.isna().sum())

# -------------------------------------------------------------
# 2) Estadística descriptiva básica
# -------------------------------------------------------------
# describe() nos da conteo, media, std, min, cuartiles y max para variables numéricas
desc = df.select_dtypes("number").describe()
pd.set_option("display.max_rows", None)       
pd.set_option("display.max_columns", None) 
print("\nDescripción numérica:\n", desc)

# -------------------------------------------------------------
# 3) Mediana, media y moda para variables clave
# -------------------------------------------------------------
# Elegimos dos variables para ejemplificar: Ventas_Mensuales y Precio_Promedio
var1 = "Ventas_Mensuales"
var2 = "Precio_Promedio"

# Media (promedio)
media_var1 = df[var1].mean(skipna=True)
media_var2 = df[var2].mean(skipna=True)

# Mediana
mediana_var1 = df[var1].median(skipna=True)
mediana_var2 = df[var2].median(skipna=True)

# Moda (si hay varias modas, tomamos la primera para mantenerlo simple)
moda_var1 = df[var1].mode(dropna=True)
moda_var2 = df[var2].mode(dropna=True)
moda_var1 = moda_var1.iloc[0] if not moda_var1.empty else None
moda_var2 = moda_var2.iloc[0] if not moda_var2.empty else None

print("\n=== Indicadores de tendencia central ===")
print(f"                  Media          Mediana          Moda                     ")
print(f"{var1}  {media_var1:.2f},      {mediana_var1:.2f},        {moda_var1}")
print(f"{var2}   {media_var2:.2f},         {mediana_var2:.2f},           {moda_var2}")


# -------------------------------------------------------------
# 4) Gráficos simples
# -------------------------------------------------------------
# 4.1) Boxplot (caja y bigotes) de Ventas_Mensuales
#     Útil para ver distribución, mediana y posibles outliers
plt.figure()
df[var1].plot(kind="box", title=f"Boxplot - {var1}")
plt.tight_layout()
plt.show()
plt.close()

# 4.2) Histograma de Ventas_Mensuales
#     Útil para ver la forma de la distribución y los rangos más frecuentes
plt.figure()
df[var1].plot(kind="hist", bins=10, title=f"Histograma - {var1}")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()
plt.close()

# 4.3) Boxplot adicional para otra variable (Precio_Promedio)
#     Esto cumple con el pedido de un boxplot extra para otra variable
plt.figure()
df[var2].plot(kind="box", title=f"Boxplot - {var2}")
plt.tight_layout()
plt.show()
plt.close()

# -------------------------------------------------------------
# 5) Matriz de correlación y heatmap
# -------------------------------------------------------------
# Calculamos la matriz de correlaciones solo con columnas numéricas
num = df.select_dtypes("number")
corr = num.corr(numeric_only=True)
pd.set_option("display.max_rows", None)      
pd.set_option("display.max_columns", None) 
print("\nMatriz de correlación:\n", corr)

# Heatmap para visualizar de forma gráfica las correlaciones
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", square=True)
plt.title("Heatmap de correlaciones (variables numéricas)")
plt.tight_layout()
plt.show()
plt.close()

# -------------------------------------------------------------
# 6) Correlaciones más fuertes con la variable objetivo
# -------------------------------------------------------------
if var1 in corr.columns:
    top_corr = corr[var1].dropna().abs().sort_values(ascending=False).head(6)
    print(f"\nCorrelaciones más fuertes (absolutas) con {var1}:\n", top_corr)