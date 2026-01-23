**Desarrollo y Validación de Modelos de Aprendizaje Automático para la Clasificación Objetiva de Calidad y Origen de Vinos a partir de Perfiles Químicos**

**Autora:** Melisa Llamazares Blanco

**Introducción**
La evaluación tradicional de la calidad del vino ha estado históricamente mediada por la percepción sensorial de catadores expertos, un proceso profundamente subjetivo, costoso y difícilmente escalable. La variabilidad intra e inter-evaluador, la fatiga sensorial y la imposibilidad de aplicar la cata clásica en líneas de producción de alta velocidad limitan críticamente la capacidad de la industria vitivinícola para mantener estándares de calidad consistentes y evaluaciones en tiempo real. Frente a este escenario, surge la necesidad de desarrollar sistemas objetivos, reproducibles y automatizados que traduzcan perfiles químicos medibles en categorías de calidad o de origen. El presente trabajo investiga la viabilidad de emplear aprendizaje automático para construir clasificadores que, a partir de análisis de laboratorio rutinarios (alcohol, ácidos, fenoles, etc.), asignen cada vino a una clase predeterminada sin intervención humana. Se estudian simultáneamente dos problemas: (i) la predicción de la calidad percibida (baja, media, alta) y (ii) la identificación del tipo de vino (clase 0, 1, 2) a partir de su firma química. El objetivo de la investigación consiste en desarrollar y validar un modelo predictivo de clasificación multiclase que, a partir de variables químicas objetivamente medibles, estime la categoría de calidad o el origen varietal de un vino con alta precisión y generalización.

---

**Marco Teórico y Análisis de los Conjuntos de Datos**

_Dataset de Scikit-learn: “Wine Dataset_"

El primer entorno experimental empleado es el denominado “Wine Dataset”, incluido de forma nativa en la librería scikit-learn. Se trata de un conjunto de 178 vinos producidos en la misma región italiana, cada uno representado por 13 variables químicas continuas y positivas: alcohol, ácido málico, ceniza, alcalinidad de cenizas, magnesio, fenoles totales, flavonoides, fenoles no flavonoides, proantocianinas, intensidad de color, de vinos diluidos y prolina. Las muestras fueron analizadas químicamente y luego asignadas por procedimientos externos a tres cultivares (clase 0, 1, 2), por lo que el problema es de clasificación multiclase, balanceado: 59, 71 y 48 ejemplares respectivamente. Gracias a su completa integridad de datos y a su dimensionalidad moderada, este conjunto permite evaluar el comportamiento de los algoritmos sin que la interferencia de valores nulos, ruido extremo o desequilibrios de clases distorsione los resultados experimentales.

_Dataset de la UCI: “Wine Quality (Vinho Verde)”_

El segundo escenario emplea el conjunto público “Wine Quality” alojado en el repositorio UCI. Contiene 6 497 muestras de vinos tintos y blancos de la región del Vinho Verde (Portugal), caracterizadas por 11 atributos físico-químicos: densidad relativa, alcohol, azúcar residual, cloruros, pH, sulfatos, dióxido de azufre total y libre, acidez fija, acidez volátil y acidez citrica. La variable objetivo es una puntuación entera entre 0 y 10 otorgada por evaluadores humanos. A diferencia del Wine Dataset, aquí la distribución es fuertemente sesgada hacia los valores centrales (5-7), siendo los extremos minoritarios. Esta asimetría convierte al conjunto en un caso práctico de aprendizaje con clases desbalanceadas y en un ejemplo realista de cómo la percepción humana se concentra en torno a la “calidad media”.

---

**Metodología de Preparación de Datos**

_Limpieza y Transformación del Objetivo_

En el Wine Dataset la variable respuesta ya se presenta como un identificador categórico ordinal (0, 1, 2) que denota, sin ambigüedad, la clase varietal de cada vino; por tanto, puede utilizarse directamente en el modelo sin transformaciones previas. En cambio, el Wine Quality codifica la percepción sensorial mediante una escala Likert de 11 puntos (0-10) que, además de estar fuertemente concentrada en los valores 5-7, introduce un desequilibrio severo en los extremos. Para evitar el deterioro del rendimiento causado por clases escasas y para alinear el problema con categorías comercialmente operativas, se agrupan los niveles originales en tres intervalos disjuntos: “Baja” (≤ 5), “Media” (6-7) y “Alta” (≥ 8). Esta recodificación reduce la dimensionalidad del espacio objetivo, atenúa el sesgo hacia la calidad intermedia y permite interpretar los resultados en términos de segmentos de mercado (estándar, premium y super-premium), conservando al mismo tiempo la información relevante para la toma de decisiones enológicas.

_Partición Estratificada_

Durante la partición de los datos se invoca `train_test_split` con el argumento `stratify=y`; esta opción activa un muestreo aleatorio estratificado que, a diferencia de un reparto puramente aleatorio, preserva la distribución real de clases en cada subconjunto. Internamente, la función separa los índices según la etiqueta y, dentro de cada estrato, extrae aleatoriamente el 20 % destinado al test. De este modo, si en el conjunto completo la clase “Alta” representa el 14 % de los vinos, exactamente ese mismo porcentaje se verá reflejado tanto en el grupo de entrenamiento como en el de prueba.

En el Wine Dataset esta precaución es meramente preventiva: al estar las clases casi equilibradas (≈ 33 % cada una), un sorteo simple difícilmente generaría desviaciones graves. No obstante, incluir `stratify=y` asegura que cualquier fluctuación muestral sea mínima y que los errores de validación se deban al modelo y no a una representación caprichosa de alguna clase.

En cambio, en el Wine Quality la estratificación deja de ser un detalle para convertirse en un requisito crítico. La clase “Media” abarca aproximadamente el 70 % de las muestras, mientras que “Alta” y “Baja” apenas alcanzan el 14 % y el 16 %, respectivamente. Sin estratificar, la aleatoriedad puede arrastrar, por ejemplo, solo el 8 % de vinos “Alta” al conjunto de prueba; la métrica de precisión de esa clase se vería artificialmente deteriorada y el investigador podría concluir erróneamente que el clasificador falla con los vinos premium cuando, en realidad, el problema radica en una evaluación incompleta. Al forzar la proporción constante, `stratify=y` garantiza que el modelo sea entrenado y juzgado sobre muestras que reflejan fielmente la estructura poblacional, permitiendo estimaciones fiables de su capacidad de generalización.

_Escalado de Características_

Previo al entrenamiento se aplica StandardScaler, técnica que convierte cada predictor a media 0 y varianza 1. La operación se ajusta únicamente sobre el conjunto de entrenamiento y luego se transforman tanto train como test, evitando así el _data-leakage_. Este paso es esencial para K-Nearest Neighbors, cuyas decisiones se basan en distancias euclidianas que, sin escalado, quedarían dominadas por variables de mayor magnitud. Por ejemplo, en el conjunto Wine Quality, los sulfatos totales pueden alcanzar valores de 200 mg/L mientras que el pH oscila entre 2.9 y 3.8. Si no se estandarizan, una diferencia de 40 mg/L en sulfatos aporta 1,600 unidades a la distancia, mientras que una variación de 0.2 unidades de pH solo contribuye con 0.04, haciendo que el algoritmo ignore completamente el pH. Tras aplicar StandardScaler, ambas variables quedan en la misma escala y contribuyen proporcionalmente a la similitud entre muestras. Random Forest, por su parte, no requiere este paso ya que sus particiones internas son invariantes a transformaciones monotónicas de escala; sin embargo, se incluye el escalado para mantener una metodología común que permita comparaciones justas entre algoritmos y facilitar la integración del modelo en flujos de trabajo estandarizados.

---

**4. Selección y Optimización de Algoritmos**

_K-Nearest Neighbors (KNN)_

K-Nearest Neighbors (KNN) es un algoritmo _lazy_ que no construye un modelo explícito durante el entrenamiento: simplemente almacena cada muestra junto con su etiqueta. Cuando llega una observación nueva, calcula la distancia entre ese punto y todas las instancias almacenadas, selecciona las _k_ más cercanas y asigna la clase mayoritaria (o el promedio ponderado si se emplea _weights='distance'_). La elección de _k_ y de la métrica de proximidad condiciona directamente la frontera de decisión: valores pequeños de _k_ generan límites muy flexibles y sensibles al ruido, mientras que _k_ grandes producen superficies de decisión más suaves y sesgadas hacia la clase predominante.

Para encontrar la combinación óptima de hiper-parámetros sin recurrir a un único split que pudiera ser atípico, se emplea **validación cruzada** _k-fold_. El método divide el conjunto de entrenamiento en _k_ particiones disjuntas del mismo tamaño; en cada iteración se reserva un pliegue como validación y los _k-1_ restantes sirven para ajustar el modelo. Tras _k_ iteraciones, cada muestra ha sido utilizada exactamente una vez para validar y _k-1_ para entrenar; la métrica de rendimiento se promedia sobre los _k_ resultados, proporcionando una estimación menos volátil y más fiable del error de generalización. En este trabajo se utiliza _k = 5_, lo que implica que el 80 % de los datos entrena y el 20 % valida en cada giro, reptitiéndose el proceso cinco veces.

`GridSearchCV` automatiza la exploración exhaustiva: construye una cuadrícula con todos los valores candidatos de los hiper-parámetros, entrena y valida cada combinación mediante la validación cruzada descrita, y selecciona la configuración que maximiza la métrica elegida (precisión, _accuracy_). En el caso del Wine Dataset se barajan 120 configuraciones para KNN, combinando _k_ ∈ [1, 21], pesos ∈ {uniform, distance} y métricas ∈ {euclidean, manhattan, minkowski}; tras los 5 pliegues, la puntuación media más alta (98,62 %) corresponde a _k = 8_, peso _uniform_ y métrica _manhattan_ en el **Wine Dataset**, mientras que en el **Wine Quality** la mejor configuración fue _k = 20_, peso _distance_ y métrica _manhattan_, con una precisión media del 78,83 %. 

_Random Forest (RF)_

Random Forest es un algoritmo de aprendizaje supervisado que pertenece a la familia de los métodos de conjunto (_ensemble_). Su objetivo principal es reducir la varianza de un clasificador individual (el árbol de decisión) sin aumentar el sesgo, consiguiendo así predicciones más estables y precisas. A diferencia de KNN, que memoriza instancias, Random Forest construye activamente un “bosque” de árboos durante el entrenamiento y combina sus salidas mediante votación mayoritaria.

El proceso comienza generando _B_ conjuntos de entrenamiento distintos mediante el denominado _bootstrap aggregating_ o _bagging_. Para cada árbol _b_ (con _b_ = 1 … _B_) se extrae aleatoriamente, con reemplazo, una muestra del mismo tamaño que el original; aproximadamente el 63 % de los ejemplares aparece al menos una vez en esa sub-muestra y el resto queda disponible para una estimación interna del error conocida como _out-of-bag_ (OOB). A continuación se crea un árbol de decisión, pero con una modificación crucial: en cada nodo, en lugar de evaluar todas las variables predictoras, solo se considera un subconjunto aleatorio de _m_ características (habitualmente √_p_ en clasificación o _p_/3 en regresión, donde _p_ es el número total de predictores). Esta doble aleatoriedad ,filas por el _bootstrap_ y columnas por la selección parcial de atributos, decorrelaciona los árboles y evita que un predictor dominante aparezca en todos ellos, reduciendo la varianza del conjunto y aumentando la robustez frente a _outliers_ y ruido.

Durante la predicción, la nueva observación se hace descender simultáneamente por los _B_ árboles; cada árbol emite un “voto” y la clase mayoritaria se adopta como resultado final. El proceso es paralelizable y, al depender de promedios, converge hacia una estimación más estable que la de cualquier árbol individual.

Para afinar el modelo se recurrió a _GridSearchCV_ con validación cruzada 5-fold, explorando 108 configuraciones que abarcaron: número de árboles _B_ ∈ {50, 100, 200}, profundidad máxima ∈ {None, 10, 20, 30}, mínimas muestras para dividir un nodo ∈ {2, 5, 10} y mínimas muestras en hoja ∈ {1, 2, 4}. En el **Wine Dataset** la combinación ganadora fue: 50 árboles, profundidad ilimitada (_None_), mínimas muestras por división 2 y una sola muestra por hoja, alcanzando 98,62 % de precisión media en CV. En el **Wine Quality**, la mejor configuración fue: 100 árboles, profundidad 20, mínimas muestras por división 2 y una sola muestra por hoja, con una precisión media del 79,80 %.

---

**Evaluación y Comparación de Resultados**

_Wine Dataset (clasificación de cultivares)_

Sobre el conjunto de prueba, tanto KNN como Random Forest alcanzan una exactitud del 100 %, evidenciando que los tres tipos de vino poseen regiones perfectamente separadas en el espacio químico normalizado. La matriz de confusión es diagonal para ambos modelos, sin falsos positivos ni negativos. La validación cruzada de 5 pliegues confirma la estabilidad de ambos modelos, con una media de precisión del 98,62 % y una desviación estándar < 0,06. El análisis de importancias de Random Forest revela que flavonoides (19,84 %), intensidad de color (17,22 %) y prolina (14,43 %) son los descriptores más relevantes, lo cual es coherente con la literatura enológica que asocia estos compuestos con la tipicidad varietal.

_Wine Quality (clasificación de calidad)_

En el escenario de calidad, Random Forest supera a KNN: 81,92 % vs. 80,54 % de exactitud en el conjunto de prueba. La matriz de confusión revela que el modelo clasifica correctamente el 89,7 % de los vinos de calidad media, pero solo el 25 % de los de calidad alta, debido al fuerte desbalance de clases. Las importancias globales indican que el alcohol (14,81 %), acidez volátil (11,43 %) y densidad (10,33 %) son los predictores más relevantes de la percepción de calidad, coincidiendo con criterios sensoriales tradicionales de equilibrio, intensidad aromática y estructura.

---

**Conclusiones**

Este estudio demuestra que la firma química del vino contiene, codificada en apenas unas pocas variables, suficiente información para discriminar cultivares con precisión absoluta y para estimar la calidad percibida con una fiabilidad superior al 80 %. La combinación de escalado riguroso, validación cruzada estratificada y optimización automática de hiper-parámetros ha permitido a KNN y Random Forest superar el umbral de utilidad industrial sin necesidad de aumentar la dimensionalidad instrumental. Más allá del récord de exactitud en el Wine Dataset, lo verdaderamente relevante es la robustez exhibida en el Wine Quality: a pesar del severo desbalance y del solapamiento natural entre clases adyacentes, ambos modelos mantienen una varianza inter-pliegue inferior al 4 % y señalan como responsables de la percepción de “alta gama” al alcohol, la acidez volátil y la densidad, tres magnitudes medibles en línea con equipos estándar de bodega. Por tanto, la transición de la cata artesanal a un control predictivo basado en datos no solo es viable: es escalable, auditable y compatible con los sistemas edge ya desplegados en las plantas de embotellado modernas.
