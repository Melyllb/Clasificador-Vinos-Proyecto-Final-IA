from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.datasets import load_wine
import numpy as np

# ============================================================================
# 1. CARGA Y EXPLORACIÓN DE DATOS
# ============================================================================
print("="*70)
print("ANÁLISIS DE CLASIFICACIÓN DE VINOS")
print("="*70)

wine_data = load_wine()

print(f"\n📊 Forma de los datos: {wine_data.data.shape}")
print(f"📋 Características: {wine_data.feature_names}")
print(f"🍷 Clases de vino: {wine_data.target_names}")
print(f"📈 Distribución de clases: {np.bincount(wine_data.target)}")
print(f"    Clase 0: {np.bincount(wine_data.target)[0]} muestras")
print(f"    Clase 1: {np.bincount(wine_data.target)[1]} muestras")
print(f"    Clase 2: {np.bincount(wine_data.target)[2]} muestras")

# ============================================================================
# 2. PREPARACIÓN DE DATOS
# ============================================================================
X = wine_data.data
y = wine_data.target

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
    f"   Proporción train/test: {X_train_scaled.shape[0]/wine_data.data.shape[0]*100:.1f}% / {X_test_scaled.shape[0]/wine_data.data.shape[0]*100:.1f}%")

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
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10)
    print(f"\n📈 Validación Cruzada (10 folds):")
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
    for i, clase in enumerate(wine_data.target_names):
        correctos = cm[i, i]
        total = cm[i, :].sum()
        print(
            f"   - {clase}: {correctos}/{total} clasificados correctamente ({correctos/total*100:.1f}%)")

    # Reporte de clasificación
    print(f"\n📋 Reporte de Clasificación Detallado:")
    print(classification_report(y_true, y_pred,
          target_names=wine_data.target_names))

    print(f"\n💡 Explicación de métricas:")
    print(f"   - Precision: De todas las predicciones de una clase, cuántas fueron correctas")
    print(f"   - Recall: De todas las muestras reales de una clase, cuántas se detectaron")
    print(f"   - F1-Score: Media armónica entre precision y recall (balance)")
    print(f"   - Support: Número de muestras reales de cada clase")

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
        f"   {i:2d}. {wine_data.feature_names[idx]:30s}: {feature_importance[idx]:.4f} ({feature_importance[idx]*100:.2f}%)")

print("\n💡 Interpretación:")
print(f"   Las 3 características más importantes son:")
for i in range(3):
    idx = indices[i]
    print(
        f"   - {wine_data.feature_names[idx]}: explica el {feature_importance[idx]*100:.2f}% de la clasificación")

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




