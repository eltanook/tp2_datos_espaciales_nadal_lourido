# Notebook Mejorada: Análisis de Datos Espaciales (ADE13) - Getis-Ord Gi*

## 📋 Descripción

Esta es una **versión mejorada y expandida** de la notebook original `tp_final_G.ipynb`, que integra:

1. **Toda la teoría fundamental** del paper de Getis & Ord (1992)
2. **Ejemplo clásico** de precios de vivienda en San Diego
3. **NUEVO**: Análisis de hotspots en datos raster (NDVI del Dique Roggero)
4. **NUEVO**: Integración con conceptos del curso (ROIs, máscaras, clasificación)

## 🎯 Objetivos de Aprendizaje

Al completar esta notebook, serás capaz de:

- Comprender la diferencia entre estadísticos globales y locales
- Aplicar el estadístico Getis-Ord Gi* para identificar hotspots y coldspots
- Distinguir entre Gi* (concentración de magnitud) y Moran's I Local (similitud)
- Extender el análisis a datos raster (imágenes satelitales)
- Interpretar Z-scores y mapas de significancia estadística

## 📂 Estructura de la Notebook

### Parte I: Fundamentos Teóricos
1. El Problema: Estadísticos Globales vs. Locales
2. Teoría: La Familia de Estadísticos G
3. El Estadístico Gi(d) (Excluyente)
4. El Estadístico Gi*(d) (Incluyente)  
5. Inferencia e Interpretación: Puntuación Z
6. Comparativa Clave: Gi* vs. Moran's I Local

### Parte II: Aplicaciones Vectoriales
7. Caso Clásico: Precios de Vivienda en San Diego
8. Carga y Preparación de Datos
9. Creación de Pesos Espaciales por Distancia
10. Cálculo de Gi* con esda
11. Visualización e Interpretación (Hotspots y Coldspots)
12. Conclusiones

### Parte III: Extensiones a Datos Raster (NUEVO)
13. Análisis de Hotspots en Datos Raster (NDVI del Dique Roggero)
    - Carga de imagen Sentinel-2
    - Creación de grilla de muestreo
    - Aplicación de Gi* a valores de NDVI
    - Identificación de hotspots de vegetación y coldspots degradados
14. Conclusiones Finales y Mejores Prácticas

## 🗂️ Datos Incluidos

Todos los datos necesarios están en la carpeta `data/`:

- `S2_dique_20181006.tif`: Imagen Sentinel-2 del Dique Roggero (oct 2018) con 5 bandas (B, G, R, NIR, NDVI)
- `S2_dique_20191120.tif`: Imagen Sentinel-2 del Dique Roggero (nov 2019) 
- `radios_BA.*`: Shapefile de radios censales de Buenos Aires (para extensiones futuras)
- `funciones.py`: Funciones auxiliares del curso (nequalize, plot_rgb, etc.)

## 🚀 Cómo Usar

1. **Asegúrate de tener instaladas las dependencias:**
   ```bash
   pip install geopandas pandas matplotlib numpy rasterio libpysal esda scipy seaborn
   ```

2. **Ejecuta las celdas en orden:**
   - Las primeras 22 celdas replican el análisis clásico de San Diego
   - Las celdas 23-30 son las NUEVAS extensiones con datos raster

3. **Explora y modifica:**
   - Cambia el parámetro `step` en la grilla de muestreo (celda 25) para más/menos puntos
   - Prueba diferentes valores de `k_neighbors` (celda 27) para cambiar la escala de vecindad
   - Aplica el mismo análisis a la imagen de 2019 y compara resultados

## 📊 Principales Visualizaciones

La notebook genera múltiples visualizaciones:

1. **Mapas de precios de San Diego** con Z-scores Gi*
2. **Mapa NDVI del Dique Roggero** con histograma
3. **Grilla de muestreo** sobre el raster
4. **Panel 2x2** con:
   - NDVI original
   - Z-scores espaciales
   - Hotspots/coldspots significativos
   - Distribución de Z-scores

## 🔑 Conceptos Clave

### Interpretación de Z-scores

| Z-score | p-value | Interpretación |
|---------|---------|----------------|
| Z > 1.96 | p < 0.05 | **Hotspot** (concentración de valores altos) |
| Z < -1.96 | p < 0.05 | **Coldspot** (concentración de valores bajos) |
| -1.96 < Z < 1.96 | p > 0.05 | No significativo (patrón aleatorio) |

### Gi* vs. Moran's I Local

| Característica | Getis-Ord Gi* | Moran's I Local (LISA) |
|----------------|---------------|------------------------|
| **Mide** | Concentración de magnitud | Similitud (autocorrelación) |
| **Distingue HH de LL** | ✅ Sí (por el signo del Z) | ❌ No (ambos dan I positivo) |
| **Uso principal** | Hotspot analysis | Detección de autocorrelación y outliers |

## 🔬 Aplicaciones

Este framework puede aplicarse a:

- **Epidemiología espacial**: Focos de enfermedades
- **Crimen y seguridad**: Zonas de alta criminalidad  
- **Economía urbana**: Precios inmobiliarios, desempleo
- **Ecología**: Deforestación, biodiversidad
- **Cambio climático**: Anomalías de temperatura
- **Teledetección**: NDVI, índices espectrales, temperatura superficial

## 📚 Referencias

**Paper original:**
Getis, A., & Ord, J. K. (1992). *The Analysis of Spatial Association by Use of Distance Statistics.* Geographical Analysis, 24(3), 189-206.

**Librerías utilizadas:**
- PySAL (libpysal + esda): https://pysal.org
- GeoPandas: https://geopandas.org
- Rasterio: https://rasterio.readthedocs.io

## 📝 Notas

- Esta notebook es parte del Trabajo Final de la materia "Análisis de Datos Espaciales"
- Los datos del Dique Roggero fueron procesados en la notebook ADE08 del curso
- Las referencias numéricas [N] en el texto corresponden a páginas del paper original

## ✅ Checklist de Completitud

- [x] Teoría completa de Getis-Ord (Secciones 1-6)
- [x] Ejemplo vectorial (San Diego, Secciones 7-12)
- [x] Ejemplo raster (NDVI Dique, Sección 13)
- [x] Visualizaciones comparativas
- [x] Interpretaciones detalladas
- [x] Conexiones con conceptos del curso
- [x] Mejores prácticas y conclusiones

---

**Autor:** Trabajo Final ADE13  
**Fecha:** Octubre 2025  
**Versión:** 2.0 (Mejorada)
