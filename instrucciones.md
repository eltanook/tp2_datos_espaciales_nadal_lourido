Actúa como un profesor experto en Geoestadística y Análisis de Datos Espaciales. Tu tarea es generar el contenido completo de un archivo Jupyter Notebook (.ipynb) destinado a mis estudiantes de la Licenciatura en Ciencia de Datos para la materia "Análisis de Datos Espaciales". Este será el notebook **ADE13** de la serie.

**Fuente Primaria Obligatoria:**
Este notebook debe estar basado fundamental e íntegramente en el paper académico que te he adjuntado en el contexto (Geographical Analysis - July 1992 - Getis - The Analysis of Spatial Association by Use of Distance Statistics.pdf). Debes consultar este paper como la **fuente primaria** para toda la teoría, fórmulas, definiciones, ejemplos y lógica de la explicación.

**Restricciones de Estilo Fundamentales (MUY IMPORTANTE):**
El notebook generado debe *emular* el estilo pedagógico, la estructura y el nivel técnico de los 12 notebooks de ejemplo del curso (serie `ADE01` a `ADE12`). Esto implica:
1.  **Formato de Inicio:** La primera celda de Markdown debe contener el título (`ADE13`), un bloque "En esta notebook..." que resuma los objetivos, y un "Índice" de las secciones.
2.  **Tono:** El lenguaje debe ser formal, académico, técnico y pedagógico, usando la voz en primera persona del plural (ej. "Vamos a explorar...", "Recordemos que...", "Notar que...").
3.  **Estructura:** Separación clara y rigurosa entre celdas de teoría (Markdown, con uso intensivo de LaTeX para fórmulas) y celdas de implementación (código Python).
4.  **Bibliotecas:** Asumir que los estudiantes ya dominan `geopandas`, `matplotlib`, `rasterio` y `scikit-learn`. Este notebook introducirá las bibliotecas `libpysal` (para pesos espaciales) y `esda` (Exploratory Spatial Data Analysis) como la continuación natural del curso.
5.  **Citación (Crítico):** Cualquier fórmula, definición, concepto teórico o referencia a los ejemplos empíricos (SIDS, San Diego) *debe* ser citado en línea usando el formato ``, donde `X` es el número de la fuente correspondiente del PDF adjunto.

**Formato de Salida:**
La respuesta debe ser un **único bloque de código JSON** que represente el archivo `.ipynb` completo, con todas sus celdas (tanto de `markdown` como de `code`). No incluyas ninguna explicación, saludo o texto fuera de este bloque JSON.

---
**ÍNDICE DE CONTENIDOS DEL NOTEBOOK (Guía Celda por Celda):**

**Celda 1: Título, Objetivos e Índice (Markdown)**
* **Título:** `Análisis de Datos Espaciales (ADE13): Estadísticos de Asociación Espacial Local (Getis-Ord Gi*)`
* **En esta notebook...:** Explicar que, habiendo visto estadísticos globales (que resumen todo el mapa en un número), ahora nos enfocaremos en estadísticos *locales*. Estos nos permiten identificar la *ubicación* de los clusters. [cite_start]Introduciremos la familia de estadísticos $G$ de Getis y Ord[cite: 3], la herramienta fundamental para lo que se conoce como "Hotspot Analysis".
* **Índice:**
    1.  El Problema: Estadísticos Globales vs. Locales
    2.  Teoría: La Familia de Estadísticos $G$
    3.  El Estadístico $G_i(d)$ (Excluyente)
    4.  El Estadístico $G_i^*(d)$ (Incluyente)
    5.  Inferencia e Interpretación: Puntuación Z
    6.  Comparativa Clave: $G_i^*$ vs. Moran's I Local
    7.  Taller Práctico: Análisis de Hotspots en Python
    8.  Paso 1: Carga y Preparación de Datos (Viviendas San Diego)
    9.  Paso 2: Creación de Pesos Espaciales por Distancia
    10. Paso 3: Cálculo de $G_i^*$ con `esda`
    11. Paso 4: Visualización e Interpretación (Hotspots y Coldspots)
    12. Conclusiones

**Celda 2: Importación de Bibliotecas (Markdown y Código)**
* **Markdown:** Breve explicación de que, además de nuestras bibliotecas habituales (`geopandas`, `matplotlib`), hoy sumaremos dos componentes clave de la familia PySAL: `libpysal` (para la creación de matrices de pesos espaciales) y `esda` (Exploratory Spatial Data Analysis), que contiene las implementaciones de los estadísticos.
* **Código:** Importar `geopandas`, `pandas`, `matplotlib.pyplot`, `numpy`, `libpysal.weights` y `esda`.

**Celda 3: 1. El Problema: Estadísticos Globales vs. Locales (Markdown)**
* Explicar la limitación de un estadístico global (como Moran's I Global). Un valor global puede indicar autocorrelación, pero no puede identificar *dónde* están los clusters.
* [cite_start]Destacar que el fracaso en tener en cuenta la autocorrelación puede llevar a serios errores de interpretación[cite: 11].
* [cite_start]El objetivo de los estadísticos $G$ es, precisamente, detectar "bolsones" (pockets) locales de dependencia que pueden no ser evidentes al usar un estadístico global[cite: 8].

**Celda 4: 2. Teoría: La Familia de Estadísticos $G$ (Markdown)**
* [cite_start]Introducir la familia de estadísticos $G$ como una medida de asociación espacial para una variable $X$[cite: 3].
* [cite_start]Mencionar un requisito clave: la variable debe tener un origen natural y ser positiva (ej. precios, tasas de enfermedad, conteos)[cite: 40, 97, 331].
* [cite_start]El estadístico $G$ se basa en una matriz de pesos espaciales $w_{ij}(d)$ que define qué ubicaciones $j$ están "cerca" de la ubicación $i$ (dentro de una distancia $d$)[cite: 46].

**Celda 5: 3. El Estadístico $G_i(d)$ (Excluyente) (Markdown)**
* [cite_start]Presentar la fórmula principal del $G_i(d)$ como se define en la Ecuación (1) del paper[cite: 43]:
    $$G_{i}(d)=\frac{\sum_{j=1}^{n}w_{ij}(d)x_{j}}{\sum_{j=1}^{n}x_{j}}, \quad j \neq i$$
* Desglosar los componentes:
    * [cite_start]$w_{ij}(d)$: Es una matriz de pesos espaciales binaria (simétrica), donde $w_{ij}=1$ si $j$ está dentro de una distancia $d$ de $i$, y $0$ si no[cite: 46].
    * [cite_start]$j \neq i$: Enfatizar que esta versión *excluye* el valor del propio punto $i$ ($x_i$) del numerador y del denominador[cite: 44, 48, 49].
* Explicar la Hipótesis Nula ($H_0$): Es la independencia espacial. [cite_start]Bajo $H_0$, cualquier permutación de los valores $x_j$ (excepto $x_i$) es igualmente probable[cite: 51].
* [cite_start]Presentar la fórmula de la Expectativa $E(G_i)$ y la Varianza $Var(G_i)$ (Ecuaciones 2 y 3)[cite: 64, 75].
    $$E(G_{i}) = \frac{W_i}{(n-1)}$$
    $$Var(G_{i})=\frac{W_{i}(n-1-W_{i})}{(n-1)^{2}(n-2)}\left(\frac{Y_{i2}}{Y_{i1}^{2}}\right)$$
    [cite_start](Definir $W_i$, $Y_{i1}$ y $Y_{i2}$ como en el paper [cite: 65, 75]).

**Celda 6: 4. El Estadístico $G_i^*(d)$ (Incluyente) (Markdown)**
* [cite_start]Introducir la variación $G_i^*(d)$, que es la más usada en la práctica (referenciar la Tabla 1 del paper)[cite: 89, 92].
* [cite_start]Explicar la diferencia clave: esta versión *incluye* el valor del punto $i$ en el análisis (es decir, $j$ puede ser igual a $i$)[cite: 92, 93].
* Presentar las fórmulas de la Tabla 1 para $G_i^*(d)$:
    $$G_{i}^{*}(d)=\frac{\sum_{j}w_{ij}(d)x_{j}}{\sum_{j}x_{j}}$$
    $$E(G_{i}^{*}) = \frac{W_i^*}{n}$$
    $$Var(G_{i}^{*})=\frac{W_{i}^{*}(n-W_{i}^{*})Y_{i2}^{*}}{n^{2}(n-1)(Y_{i1}^{*})^{2}}$$
* Explicar que esta es la versión que implementaremos y la que se usa comúnmente para "Hotspot Analysis" en software GIS y en `esda`.

**Celda 7: 5. Inferencia e Interpretación: Puntuación Z (Markdown)**
* [cite_start]Explicar que, asumiendo normalidad, no interpretamos el valor $G_i$ crudo, sino su Z-score (valor estandarizado) [cite: 107][cite_start], como se define en la Ecuación (4)[cite: 108]:
    $$Z_{i}=\frac{G_{i}(d)-E[G_{i}(d)]}{\sqrt{Var(G_{i}(d))}}$$
* **Interpretación (Crítico):**
    * [cite_start]**Hotspot (Cluster de valores ALTOS):** Un Z-score significativamente *positivo* (ej. > 1.96) implica una concentración espacial de valores *altos* (high values) de $X$ alrededor del punto $i$[cite: 111].
    * [cite_start]**Coldspot (Cluster de valores BAJOS):** Un Z-score significativamente *negativo* (ej. < -1.96) implica una concentración espacial de valores *bajos* (small values) de $X$ alrededor del punto $i$[cite: 112].
    * **No significativo:** Un Z-score cercano a 0 indica un patrón aleatorio.

**Celda 8: 6. Comparativa Clave: $G_i^*$ vs. Moran's I Local (LISA) (Markdown)**
* [cite_start]Esta es una distinción teórica fundamental, referenciando la Tabla 2 del paper[cite: 301].
* **Moran's I Local (LISA):** Mide *similitud*. Un I positivo alto significa que los vecinos son *similares*. Esto puede ser un cluster High-High (HH) *o* un cluster Low-Low (LL). [cite_start]Ambas situaciones producen un Z-score positivo, por lo que Moran's I *no distingue* entre clusters de altos y bajos valores[cite: 322].
* **Getis-Ord $G_i^*$:** Mide *concentración de magnitud*. Está diseñado *específicamente* para distinguir:
    * [cite_start]HH (High-High): Produce un Z-score positivo (Hotspot)[cite: 304, 308].
    * [cite_start]LL (Low-Low): Produce un Z-score negativo (Coldspot)[cite: 306, 321].
    * [cite_start]HL (High-Low): Produce un Z-score negativo (pero Moran's I sería fuertemente negativo)[cite: 318].
* [cite_start]Concluir que deben usarse en conjunto: $I$ para autocorrelación y $G$ para identificar las características (magnitud) de esos clusters[cite: 8, 324].

**Celda 9: 7. Taller Práctico: Análisis de Hotspots en Python (Markdown)**
* [cite_start]Introducir el caso de estudio: replicaremos conceptualmente el segundo ejemplo empírico del paper: el análisis de precios de vivienda en San Diego por código postal[cite: 339, 404].
* [cite_start]El objetivo es identificar los "bolsones" (pockets) de precios altos (hotspots) y precios bajos (coldspots) que el análisis de $G(d)$ global y $I(d)$ global no revelaron claramente [cite: 415-417].
* [cite_start]Usaremos los datos del Apéndice del paper[cite: 473].

**Celda 10: 8. Paso 1: Carga y Preparación de Datos (Viviendas San Diego) (Código)**
* [cite_start]**Markdown:** Explicar que, para que el notebook sea autocontenido, cargaremos una muestra de los datos presentados en el Apéndice (página 17)[cite: 473], que incluye las coordenadas (en millas) y el precio (en miles de USD).
* **Código:**
    ```python
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt

    # [cite_start]Datos extraídos del Apéndice (p. 17) [cite: 473]
    data = {
        'zip_code': ['92024', '92007', '92075', '92014', '92127', '92129', '92128', '92064',
                     '92037', '92122', '92117', '92109', '92110', '92111', '92123', '92108',
                     '92103', '92104', '92105', '92113', '92102', '92107', '92106', '92118'],
        'neighborhood': ['Encinitas', 'Cardiff', 'Solana Beach', 'Del Mar', 'Lake Hodges', 'R. Penasquitos', 'R. Bernardo', 'Poway',
                         'La Jolla', 'University City', 'Clairemont', 'Beaches', 'Bay Park', 'Kearny Mesa', 'Mission Village', 'Mission Valley',
                         'Hillcrest', 'North Park', 'East San Diego', 'Logan Heights', 'East San Diego', 'Ocean Beach', 'Point Loma', 'Coronado'],
        'x': [1, 2, 3, 5, 10, 12, 15, 17, 3, 6, 6, 4, 6, 8, 10, 9, 8, 11, 13, 11, 12, 3, 3, 7],
        'y': [39, 36, 34, 32, 34, 32, 35, 32, 22, 23, 20, 18, 15, 19, 19, 16, 14, 14, 14, 10, 12, 14, 12, 10],
        'price': [264, 260, 261, 309, 265, 194, 191, 236, 398, 201, 192, 249, 152, 138, 131, 89, 225, 152, 111, 84, 88, 229, 338, 374]
    }
    df = pd.DataFrame(data)

    # Convertir a GeoDataFrame usando las coordenadas en millas
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.x, df.y)
    )

    # [cite_start]Visualizar los datos base (similar a la Figura 3 [cite: 430])
    fig, ax = plt.subplots(figsize=(10, 12))
    gdf.plot(column='price', ax=ax, legend=True, cmap='viridis', s=50,
             legend_kwds={'label': "Precio de Vivienda (Miles USD)", 'orientation': "horizontal"})
    # Anotar algunos puntos para referencia
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.neighborhood):
        if label in ['La Jolla', 'Point Loma', 'Mission Valley', 'East San Diego']:
            ax.annotate(label, (x+0.2, y+0.2), textcoords="offset points", xytext=(0,1), ha='left', fontsize=9)
    [cite_start]ax.set_title('Precios de Vivienda en San Diego (Muestra) [cite: 473]')
    plt.xlabel('Coordenada X (millas)')
    plt.ylabel('Coordenada Y (millas)')
    plt.show()
    ```

**Celda 11: 9. Paso 2: Creación de Pesos Espaciales por Distancia (Markdown y Código)**
* [cite_start]**Markdown:** Explicar que los estadísticos $G_i$ requieren una matriz de pesos basada en distancia $d$[cite: 46]. [cite_start]En el paper, el análisis de $G_i^*$ para San Diego se realizó con $d=5$ millas[cite: 418, 460]. Replicaremos esto usando `libpysal.weights.DistanceBand`. Esta función crea una matriz binaria donde $w_{ij}=1$ si la distancia euclidiana entre $i$ y $j$ es menor o igual a $d$, y $0$ en caso contrario.
* **Código:**
    ```python
    import libpysal.weights as weights

    # [cite_start]Definir el umbral de distancia (d) en 5 millas, como en el paper [cite: 460]
    d_threshold = 5.0

    # Crear la matriz de pesos basada en banda de distancia
    # 'binary=True' es el default y replica la matriz {0,1} del paper
    W = weights.DistanceBand.from_dataframe(gdf, d_threshold)

    # Es importante que W no esté estandarizada por fila para este estadístico
    W.transform = 'b' # 'b' for binary (solo para asegurar)

    # Inspeccionemos los vecinos de 'Point Loma' (ZIP 92106)
    point_loma = gdf[gdf['neighborhood'] == 'Point Loma']
    point_loma_index = point_loma.index[0]

    print(f"Vecinos de '{point_loma.iloc[0].neighborhood}' (a {d_threshold} millas o menos):")
    neighbors_indices = W.neighbors[point_loma_index]
    print(gdf.loc[neighbors_indices]['neighborhood'])
    ```

**Celda 12: 10. Paso 3: Cálculo de $G_i^*$ con `esda` (Markdown y Código)**
* **Markdown:** Explicar que ahora aplicaremos el estadístico $G_i^*$ usando `esda.G_Local`. Pasaremos la variable de interés ($y$) y la matriz de pesos ($W$). [cite_start]Es fundamental especificar `star=True` para asegurar que estamos usando la versión $G_i^*$ (incluyente) como en la Tabla 1 y Figura 4 del paper[cite: 92, 460].
* **Código:**
    ```python
    import esda
    import numpy as np

    # Variable a analizar (debe ser un array de numpy)
    y = gdf['price'].values

    # Calcular el estadístico G_Local (G_i*)
    # star=True indica que usamos la versión G_i* (incluye el valor propio)
    g_local_star = esda.G_Local(y, W, star=True)

    # Añadir los Z-scores (Zs) y p-values (p_sim) al GeoDataFrame
    gdf['G_star_Zs'] = g_local_star.Zs
    gdf['G_star_p_sim'] = g_local_star.p_sim # p-values basados en simulación (permutaciones)

    # Ver los resultados, ordenados por Z-score
    print("Resultados del Análisis G_i* (d=5 millas)")
    print(gdf[['neighborhood', 'price', 'G_star_Zs', 'G_star_p_sim']].sort_values('G_star_Zs', ascending=False).to_markdown(index=False))
    ```

**Celda 13: 11. Paso 4: Visualización e Interpretación (Hotspots y Coldspots) (Markdown y Código)**
* [cite_start]**Markdown:** Ahora vamos a visualizar nuestros resultados, replicando la lógica de la Figura 4 del paper[cite: 460]. Crearemos dos mapas:
    1.  Un mapa de los Z-scores crudos para ver la distribución espacial de los valores altos (rojo) y bajos (azul).
    2.  Un mapa de significancia, que filtre solo los códigos postales donde el Z-score es extremo (ej. > 1.96 o < -1.96) y el p-value es estadísticamente significativo (ej. < 0.05).
* **Código (Mapa 1: Z-Scores):**
    ```python
    # Mapa 1: Visualización de los Z-scores
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Usamos un mapa de color divergente (Rojo-Azul)
    # 'coolwarm' o 'RdBu_r' son buenas opciones
    gdf.plot(
        column='G_star_Zs',
        cmap='coolwarm',
        legend=True,
        ax=ax,
        s=80,
        legend_kwds={'label': "Z-score $G_i^*$ (d=5 millas)"}
    )
    # Anotar los valores Z
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.G_star_Zs):
        ax.annotate(f"{label:.2f}", (x+0.2, y-0.1), textcoords="offset points", xytext=(0,1), ha='left', fontsize=8)
    
    ax.set_title('Análisis $G_i^*$ de Precios de Vivienda (Hotspots y Coldspots)')
    plt.xlabel('Coordenada X (millas)')
    plt.ylabel('Coordenada Y (millas)')
    plt.show()
    ```
* **Markdown (Interpretación Mapa 1):**
    * Explicar el mapa: Se observan Z-scores fuertemente positivos (rojo oscuro) en las zonas costeras como La Jolla y Point Loma, indicando hotspots de precios altos.
    * Se observan Z-scores fuertemente negativos (azul oscuro) en zonas centrales/del este como Mission Valley y East San Diego, indicando coldspots de precios bajos.
    * [cite_start]Esto coincide exactamente con los hallazgos del paper, que identificó los distritos costeros como positivamente asociados (hotspots) y los distritos centrales como negativamente asociados (coldspots)[cite: 434, 436, 447].
* **Código (Mapa 2: Mapa de Significancia):**
    ```python
    # Nivel de significancia
    alpha = 0.05

    # Crear una columna para la clasificación del cluster
    # Usamos los rangos del paper (1.96)
    gdf['cluster_type'] = 'No Significativo'
    gdf.loc[(gdf['G_star_Zs'] > 1.96) & (gdf['G_star_p_sim'] < alpha), 'cluster_type'] = 'Hotspot (HH) (p<0.05)'
    gdf.loc[(gdf['G_star_Zs'] < -1.96) & (gdf['G_star_p_sim'] < alpha), 'cluster_type'] = 'Coldspot (LL) (p<0.05)'

    # Mapear solo los clusters significativos
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Colores para el mapa categórico
    colors = {'Hotspot (HH) (p<0.05)': 'red', 'Coldspot (LL) (p<0.05)': 'blue', 'No Significativo': 'lightgrey'}
    
    gdf.plot(
        column='cluster_type',
        categorical=True,
        cmap='manual',
        values=colors.values(),
        legend=True,
        ax=ax,
        s=80,
        legend_kwds={'title': "Tipo de Cluster", 'loc': 'upper left'}
    )
    # Anotar los barrios significativos
    for x, y, label, cluster in zip(gdf.geometry.x, gdf.geometry.y, gdf.neighborhood, gdf.cluster_type):
        if cluster != 'No Significativo':
            ax.annotate(label, (x+0.2, y+0.2), textcoords="offset points", xytext=(0,1), ha='left', fontsize=9)

    ax.set_title('Hotspots y Coldspots Estadísticamente Significativos (p < 0.05)')
    plt.xlabel('Coordenada X (millas)')
    plt.ylabel('Coordenada Y (millas)')
    plt.show()
    ```
* **Markdown (Interpretación Final):**
    * El mapa de significancia confirma los "bolsones" (pockets). Vemos un claro hotspot de precios altos en la costa (Point Loma, La Jolla) y un coldspot de precios bajos en el interior (Mission Valley, East San Diego).
    * [cite_start]Este es el poder de los estadísticos $G_i$: revelan patrones locales que el $G(d)$ global (que fue negativo [cite: 416][cite_start]) y el $I(d)$ global (que fue positivo [cite: 415]) no pueden mostrar por sí solos. [cite_start]Los G-stats nos dicen *por qué* el G global fue negativo: porque el cluster de valores bajos (coldspot) era más influyente que el de valores altos[cite: 416, 436].

**Celda 14: 12. Conclusiones (Markdown)**
* Resumir la lección. [cite_start]Los estadísticos $G$ nos proveen una forma directa de evaluar la asociación espacial, tanto global como local[cite: 452].
* [cite_start]Hemos aprendido la diferencia fundamental entre el estadístico $G_i^*$ (que mide magnitud) y Moran's I (que mide similitud)[cite: 325, 455].
* La implementación práctica con `libpysal` y `esda` nos permitió replicar el análisis empírico del paper de Getis & Ord, identificando hotspots y coldspots de precios de vivienda.
* [cite_start]El uso de estadísticos locales ($G_i^*$) es fundamental, ya que, como vimos en el ejemplo, una medida global puede ser engañosa y ocultar los patrones locales subyacentes[cite: 401].