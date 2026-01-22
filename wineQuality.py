from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import numpy as np
import pandas as pd
import os

# ============================================================================
# 1. CARGA DE DATOS LOCAL
# ============================================================================
print("="*70)
print("ANÁLISIS DE CLASIFICACIÓN DE CALIDAD DE VINOS - VERSIÓN LOCAL")
print("="*70)

# Nombre del archivo local
archivo_local = "wine_quality_dataset.csv"

# Verificar si el archivo existe localmente
if os.path.exists(archivo_local):
    print(f"\n✓ Cargando dataset desde archivo local: {archivo_local}")
    df_vinos = pd.read_csv(archivo_local)
    print(f"✓ Dataset cargado exitosamente desde disco")
else:
    print(f"\n⚠️ Archivo local no encontrado. Descargando dataset...")
    print(f"   (Esto solo ocurrirá la primera vez)")
    
    try:
        from ucimlrepo import fetch_ucirepo
        
        # Descargar dataset
        wine_quality = fetch_ucirepo(id=186)
        
        # Combinar características y objetivo
        df_vinos = pd.concat([wine_quality.data.features, wine_quality.data.targets], axis=1)
        
        # Guardar localmente
        df_vinos.to_csv(archivo_local, index=False)
        print(f"✓ Dataset descargado y guardado como: {archivo_local}")
        print(f"✓ La próxima vez se cargará desde el archivo local")
        
    except ImportError:
        print("\n❌ ERROR: No se puede descargar el dataset.")
        print("   Necesitas instalar: pip install ucimlrepo")
        print("\n   O descarga manualmente los archivos CSV desde:")
        print("   https://archive.ics.uci.edu/dataset/186/wine+quality")
        exit(1)

print(f"\n✓ Total de muestras: {len(df_vinos)}")
print(f"\n📋 Columnas del dataset: {list(df_vinos.columns)}")
print(f"📊 Forma de los datos: {df_vinos.shape}")

# Mostrar distribución original de calidades
print(f"\n📊 Distribución ORIGINAL de calidades:")
print("-"*70)
distribucion_original = df_vinos['quality'].value_counts().sort_index()
for calidad, conteo in distribucion_original.items():
    print(f"   Calidad {calidad}: {conteo:4d} muestras ({conteo/len(df_vinos)*100:.2f}%)")

# Crear variable objetivo con TRES clases basadas en calidad
# Clase 0: Calidad Baja (quality <= 5)
# Clase 1: Calidad Media (quality 6-7)
# Clase 2: Calidad Alta (quality >= 8)
def clasificar_calidad(quality):
    if quality <= 5:
        return 0  # Baja
    elif quality <= 7:
        return 1  # Media
    else:
        return 2  # Alta

df_vinos['calidad_categorica'] = df_vinos['quality'].apply(clasificar_calidad)

print(f"\n📊 Distribución DESPUÉS de categorizar (Bajo-Medio-Alto):")
print("-"*70)
conteo_clases = df_vinos['calidad_categorica'].value_counts().sort_index()
for clase in range(3):
    if clase in conteo_clases.index:
        conteo = conteo_clases[clase]
        porcentaje = conteo/len(df_vinos)*100
        clase_nombre = ['Baja (≤5)', 'Media (6-7)', 'Alta (≥8)'][clase]
        print(f"   Clase {clase} ({clase_nombre}): {conteo:4d} muestras ({porcentaje:.2f}%)")

# Verificar desbalance
desbalance_ratio = conteo_clases.max() / conteo_clases.min()
print(f"\n⚖️ Ratio de desbalance: {desbalance_ratio:.2f}:1")
if desbalance_ratio > 3:
    print(f"   ⚠️ ADVERTENCIA: Dataset desbalanceado (ratio > 3:1)")
    print(f"   Se recomienda usar estratificación en train_test_split")
else:
    print(f"   ✓ Dataset relativamente balanceado")

print(f"\n📊 Forma de los datos: {df_vinos.shape}")
print(f"🍷 Clases: ['Calidad Baja (≤5)', 'Calidad Media (6-7)', 'Calidad Alta (≥8)']")

# ============================================================================
# 2. PREPARACIÓN DE DATOS
# ============================================================================
# Separar características y objetivo (excluyendo 'quality' y 'calidad_categorica')
columnas_excluir = ['quality', 'calidad_categorica']
X = df_vinos.drop(columns=columnas_excluir).values
y = df_vinos['calidad_categorica'].values

nombres_caracteristicas = df_vinos.drop(columns=columnas_excluir).columns.tolist()
nombres_clases = ['Calidad Baja (≤5)', 'Calidad Media (6-7)', 'Calidad Alta (≥8)']

print(f"\n✓ Características usadas para el modelo: {nombres_caracteristicas}")
print(f"✓ Número de características: {len(nombres_caracteristicas)}")

# División estratificada para mantener proporciones de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Datos preparados:")
print(f"   Conjunto de entrenamiento: {X_train_scaled.shape}")
print(f"   Conjunto de prueba: {X_test_scaled.shape}")
print(
    f"   Proporción train/test: {X_train_scaled.shape[0]/len(df_vinos)*100:.1f}% / {X_test_scaled.shape[0]/len(df_vinos)*100:.1f}%")

# ============================================================================
# 3. OPTIMIZACIÓN DE HIPERPARÁMETROS
# ============================================================================
print("\n" + "="*70)
print("OPTIMIZACIÓN DE HIPERPARÁMETROS")
print("="*70)

# Optimizar Random Forest
print("\n🌲 Optimizando Random Forest...")
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rf_grid.fit(X_train_scaled, y_train)

print(f"   Mejores parámetros encontrados: {rf_grid.best_params_}")
print(
    f"   Mejor puntuación en validación cruzada: {rf_grid.best_score_:.4f} ({rf_grid.best_score_*100:.2f}%)")

# Optimizar KNN
print("\n👥 Optimizando KNN...")
knn_params = {
    'n_neighbors': range(1, 21),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
knn_grid.fit(X_train_scaled, y_train)

print(f"   Mejores parámetros encontrados: {knn_grid.best_params_}")
print(
    f"   Mejor puntuación en validación cruzada: {knn_grid.best_score_:.4f} ({knn_grid.best_score_*100:.2f}%)")

# ============================================================================
# 4. ENTRENAMIENTO CON MEJORES MODELOS
# ============================================================================
print("\n" + "="*70)
print("EVALUACIÓN DE MODELOS")
print("="*70)

# Usar los mejores modelos encontrados
rf_model = rf_grid.best_estimator_
knn_model = knn_grid.best_estimator_

# Predicciones
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_knn = knn_model.predict(X_test_scaled)

# ============================================================================
# 5. EVALUACIÓN DETALLADA
# ============================================================================


def evaluar_modelo(nombre, y_true, y_pred, model):
    """Función para evaluar y mostrar métricas de un modelo"""
    print(f"\n{'='*70}")
    print(f"🎯 {nombre}")
    print(f"{'='*70}")

    # Exactitud (Accuracy)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n📊 Exactitud (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(
        f"   Esto significa que el modelo clasificó correctamente {accuracy*100:.2f}% de las muestras")

    # Validación cruzada
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\n📈 Validación Cruzada (5 folds):")
    print(
        f"   Puntuaciones por fold: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"   Media: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(
        f"   Esto indica que el modelo es {'muy estable' if cv_scores.std() < 0.05 else 'relativamente estable'} entre diferentes particiones")

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🔢 Matriz de Confusión:")
    print(f"   (Filas = Valor Real, Columnas = Predicción)")
    print(f"   {cm}")

    # Interpretación de la matriz de confusión
    print(f"\n   Interpretación:")
    for i, clase in enumerate(nombres_clases):
        correctos = cm[i, i]
        total = cm[i, :].sum()
        print(
            f"   - {clase}: {correctos}/{total} clasificados correctamente ({correctos/total*100:.1f}%)")



    return accuracy, cm


# Evaluar ambos modelos
acc_rf, cm_rf = evaluar_modelo(
    "Random Forest Classifier", y_test, y_pred_rf, rf_model)
acc_knn, cm_knn = evaluar_modelo(
    "K-Nearest Neighbors Classifier", y_test, y_pred_knn, knn_model)

# ============================================================================
# 6. ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
# ============================================================================
print("\n" + "="*70)
print("ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS (Random Forest)")
print("="*70)

feature_importance = rf_model.feature_importances_
indices = np.argsort(feature_importance)[::-1]

print("\n📊 Ranking de características más importantes:")
for i, idx in enumerate(indices, 1):
    print(
        f"   {i:2d}. {nombres_caracteristicas[idx]:30s}: {feature_importance[idx]:.4f} ({feature_importance[idx]*100:.2f}%)")

print("\n💡 Interpretación:")
print(f"   Las 3 características más importantes son:")
for i in range(3):
    idx = indices[i]
    print(
        f"   - {nombres_caracteristicas[idx]}: explica el {feature_importance[idx]*100:.2f}% de la clasificación")

# ============================================================================
# 7. CONCLUSIONES Y COMPARACIÓN
# ============================================================================
print("\n" + "="*70)
print("CONCLUSIONES Y COMPARACIÓN DE MODELOS")
print("="*70)

print(f"\n📊 Comparación de Exactitud:")
print(f"   Random Forest:  {acc_rf:.4f} ({acc_rf*100:.2f}%)")
print(f"   KNN:            {acc_knn:.4f} ({acc_knn*100:.2f}%)")
print(
    f"   Diferencia:     {abs(acc_rf - acc_knn):.4f} ({abs(acc_rf - acc_knn)*100:.2f} puntos porcentuales)")

mejor_modelo = "Random Forest" if acc_rf > acc_knn else "KNN"
mejor_accuracy = max(acc_rf, acc_knn)

print(
    f"\n🏆 Mejor modelo: {mejor_modelo} con {mejor_accuracy:.4f} ({mejor_accuracy*100:.2f}%) de exactitud")

print("\n" + "="*70)
print("✓ ANÁLISIS COMPLETADO")
print(f"✓ Dataset guardado localmente en: {archivo_local}")
print("="*70)