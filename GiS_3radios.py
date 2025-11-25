#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 10:19:40 2025

@author: rgrimson
"""

import numpy as np
import rasterio
from scipy.ndimage import convolve
from scipy.signal import gaussian
import matplotlib.pyplot as plt

def create_square_kernel(radius=1):
    """Crea kernel cuadrado (vecindad tipo reina)"""
    size = 2 * radius + 1
    kernel = np.ones((size, size))
    center = radius
    
    # Excluir el pixel central
    kernel[center, center] = 0
    
    # Normalizar
    kernel = kernel / kernel.sum()
    return kernel

def create_gaussian_kernel(radius=1, sigma=1.0):
    """
    Crea kernel gaussiano 2D
    radius: radio del kernel (tamaño = 2*radius + 1)
    sigma: desviación estándar de la gaussiana en multiplos del radio
    """
    size = 2 * radius + 1
    kernel_1d = gaussian(size, std=sigma*radius)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    
    # Excluir el píxel central (¡IMPORTANTE para Gi*!)
    center = radius
    kernel_2d[center, center] = 0
    
    # Normalizar para que sume 1
    if np.sum(kernel_2d) > 0:
        kernel_2d = kernel_2d / np.sum(kernel_2d)
    
    return kernel_2d

def create_circular_kernel(radius=1):
    """Kernel circular con centro excluido"""
    size = 2 * radius + 1
    kernel = np.zeros((size, size))
    center = radius
    
    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - center)**2 + (j - center)**2)
            if distance <= radius:
                kernel[i, j] = 1
    
    # EXCLUIR PÍXEL CENTRAL (requisito de Gi*)
    kernel[center, center] = 0
    
    # Normalizar solo si hay vecinos
    if np.sum(kernel) > 0:
        kernel = kernel / np.sum(kernel)
    
    return kernel

#%%
def compute_gistar_optimized(raster_data, kernel, global_mean=None, global_std=None):
    """Calcula Gi* manteniendo la esencia del método original"""
    
    if global_mean is None:
        global_mean = np.nanmean(raster_data)
    if global_std is None:
        global_std = np.nanstd(raster_data)
    
    valid_mask = ~np.isnan(raster_data)
    n_valid = np.sum(valid_mask)
    
    # 1. Suma ponderada de VECINOS (excluyendo centro)
    weighted_sum = convolve(raster_data, kernel, mode='constant', cval=np.nan)
    
    # 2. Número efectivo de vecinos
    kernel_ones = np.ones_like(kernel)
    kernel_ones[kernel.shape[0]//2, kernel.shape[1]//2] = 0  # Excluir centro
    n_neighbors = convolve(valid_mask.astype(float), kernel_ones, 
                          mode='constant', cval=0)
    
    # 3. Cálculo de Gi* (fórmula simplificada pero conceptualmente correcta)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Componente clave: solo vecinos, no el punto central
        gi_star = (weighted_sum - global_mean * np.sum(kernel)) / (
            global_std * np.sqrt((n_valid * np.sum(kernel**2) - (np.sum(kernel))**2) / (n_valid - 1)))
    
    return gi_star
#%%
# PROGRAMA PRINCIPAL CON OPCIONES DE KERNEL
def main(raster_path, kernel_type='gaussian', radius=2, sigma=1.0):
    """Ejecuta análisis con diferentes kernels"""
    
    # Cargar raster
    with rasterio.open(raster_path) as src:
        ndvi_data = src.read(1)
    
    # Crear kernel según tipo
    if kernel_type == 'gaussian':
        kernel = create_gaussian_kernel(radius, sigma)
        kernel_name = f'Gaussiano (σ={sigma})'
    elif kernel_type == 'circular':
        kernel = create_circular_kernel(radius)
        kernel_name = 'Circular'
    elif kernel_type == 'square':
        kernel = create_square_kernel(radius)
        kernel_name = 'Cuadrado'
    else:
        raise ValueError("Kernel debe ser: 'gaussian', 'circular', o 'square'")
    
    print(f"Kernel: {kernel_name}, Radio: {radius}")
    print(f"Vecinos efectivos: {np.sum(kernel > 0)}")
    print(f"Kernel:\n{kernel}")
    
    # Calcular Gi*
    gi_star_result = compute_gistar_optimized(ndvi_data, kernel)
    #%%
    # Visualización comparativa
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Kernel
    im0 = axes[0].imshow(kernel, cmap='viridis')
    axes[0].set_title(f'Kernel {kernel_name}\nRadio={radius}')
    plt.colorbar(im0, ax=axes[0])
    
    # Gi* scores
    im1 = axes[1].imshow(gi_star_result, cmap='RdBu_r', vmin=-50, vmax=150)
    axes[1].set_title('Gi* Z-scores')
    plt.colorbar(im1, ax=axes[1])
    
    # Significancia
    significant = np.where(np.abs(gi_star_result) > 86 | (gi_star_result < 40), gi_star_result, np.nan)
    im2 = axes[2].imshow(significant, cmap='RdBu_r', vmin=-50, vmax=150)
    axes[2].set_title('Hot/Cold Spots (p < 0.05)')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(f'gi_star_{kernel_type}_r{radius}.png', dpi=300)
    plt.show()
    #%%
    return gi_star_result

#%%
# EJECUCIÓN CON DIFERENTES KERNELS
raster_path = "/home/rgrimson/Downloads/GiS/ndvi_caba.tif"  # Cambia por tu ruta
raster_out_path = raster_path.split('.')[0]

# Comparar kernels
results = {}
for kernel_type in ['gaussian', 'circular', 'square']:
    print(f"\n=== {kernel_type.upper()} ===")
    results[kernel_type] = main(raster_path, 
                              kernel_type=kernel_type,
                              radius=10,         # Radio en píxeles
                              sigma=1.5)        # Solo para gaussian
    fn = raster_out_path  + '_' + kernel_type +'.tif'
    save_gistar_raster(fn, gi_star_result, raster_path)
    


#%%
#NUNCA LO PROBE ESTO es con fft
#%%
#%%
def save_gistar_raster(output_path, gi_star_data, reference_raster_path, output_dtype=rasterio.float32):
    """
    Guarda los resultados de Gi* como raster GeoTIFF con referencia espacial
    
    Parameters:
    output_path: ruta donde guardar el raster de salida
    gi_star_data: matriz numpy con los valores de Gi*
    reference_raster_path: ruta al raster original para extraer metadatos
    output_dtype: tipo de datos de salida (por defecto float32)
    """
    try:
        # Abrir raster de referencia para obtener metadatos
        with rasterio.open(reference_raster_path) as src:
            # Copiar el perfil (metadata) del raster original
            profile = src.profile.copy()
            
            # Actualizar el perfil para el raster de salida
            profile.update({
                'dtype': output_dtype,
                'count': 1,  # Una banda
                'nodata': -9999.0,  # Valor para datos faltantes
                'driver': 'GTiff',  # Formato GeoTIFF
                'compress': 'lzw'  # Compresión para reducir tamaño
            })
        
        # Asegurar que los datos tengan las mismas dimensiones
        if gi_star_data.shape != (src.height, src.width):
            print(f"Advertencia: Dimensiones diferentes. Original: {src.height}x{src.width}, Gi*: {gi_star_data.shape}")
            # Recortar o ajustar si es necesario
            gi_star_data = gi_star_data[:src.height, :src.width]
        
        # Reemplazar NaN por el valor nodata
        gi_star_data_clean = np.where(np.isnan(gi_star_data), -9999.0, gi_star_data.astype(output_dtype))
        
        # Guardar el raster
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(gi_star_data_clean, 1)
            
        print(f"✅ Raster guardado exitosamente: {output_path}")
        print(f"   Dimensiones: {gi_star_data.shape}")
        print(f"   Rango de valores: [{np.nanmin(gi_star_data):.3f}, {np.nanmax(gi_star_data):.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error guardando raster: {str(e)}")
        return False
#%%
#%%
import numpy as np
import rasterio
from scipy.ndimage import convolve, zoom
import time

def compute_gistar_large_scale(raster_path, target_scale_meters=100, original_resolution=10):
    """
    Calcula Gi* para grandes escalas optimizado
    target_scale_meters: escala de análisis en metros (ej: 100, 200, 500)
    original_resolution: resolución del raster en metros (10m)
    """
    
    # 1. CALCULAR FACTOR DE REDUCCIÓN
    scale_factor = target_scale_meters / original_resolution
    reduction_factor = int(np.round(scale_factor / 2))  # Reducción óptima
    
    print(f"Escala objetivo: {target_scale_meters}m")
    print(f"Factor de reducción: {reduction_factor}x")
    print(f"Nueva resolución: {original_resolution * reduction_factor}m")
    
    # 2. CARGAR Y REDUCIR RESOLUCIÓN
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        transform = src.transform
        profile = src.profile.copy()
    
    # Reducir resolución para análisis
    if reduction_factor > 1:
        reduced_data = zoom(data, 1/reduction_factor, order=1)  # Interpolación bilineal
        print(f"Tamaño original: {data.shape} -> Reducido: {reduced_data.shape}")
    else:
        reduced_data = data
    
    # 3. KERNEL OPTIMIZADO PARA GRAN ESCALA
    # Radio en píxeles de la nueva resolución
    kernel_radius_pixels = max(2, int(3 * scale_factor / reduction_factor))
    kernel_size = 2 * kernel_radius_pixels + 1
    
    print(f"Radio del kernel: {kernel_radius_pixels} píxeles")
    print(f"Tamaño del kernel: {kernel_size}x{kernel_size}")
    
    # Crear kernel gaussiano para gran escala
    kernel = create_gaussian_kernel_large(kernel_radius_pixels, sigma=kernel_radius_pixels/2)
    
    # 4. CÁLCULO EN BAJA RESOLUCIÓN (¡RÁPIDO!)
    start_time = time.time()
    
    global_mean = np.nanmean(reduced_data)
    global_std = np.nanstd(reduced_data)
    
    gi_star_reduced = compute_gistar_optimized(reduced_data, kernel, global_mean, global_std)
    
    # 5. RECONSTRUIR A ALTA RESOLUCIÓN
    if reduction_factor > 1:
        gi_star_highres = zoom(gi_star_reduced, reduction_factor, order=1)
        # Ajustar a tamaño original si es necesario
        if gi_star_highres.shape != data.shape:
            gi_star_highres = gi_star_highres[:data.shape[0], :data.shape[1]]
    else:
        gi_star_highres = gi_star_reduced
    
    computation_time = time.time() - start_time
    print(f"Tiempo de cálculo: {computation_time:.2f} segundos")
    
    return gi_star_highres, reduced_data, kernel

def create_gaussian_kernel_large(radius, sigma=None):
    """Kernel gaussiano optimizado para grandes radios"""
    if sigma is None:
        sigma = radius / 2  # Sigma proporcional al radio
    
    size = 2 * radius + 1
    x = np.arange(size) - radius
    y = np.arange(size) - radius
    xx, yy = np.meshgrid(x, y)
    
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel[radius, radius] = 0  # Excluir centro
    kernel = kernel / np.sum(kernel)  # Normalizar
    
    return kernel

# FUNCIÓN SUPER OPTIMIZADA CON FFT
def compute_gistar_fft_optimized(data, kernel):
    """Calcula Gi* usando FFT para máxima velocidad con kernels grandes"""
    from scipy.signal import fftconvolve
    
    # Calcular estadísticas globales
    global_mean = np.nanmean(data)
    global_std = np.nanstd(data)
    
    # Reemplazar NaNs por media para la convolución
    data_filled = np.where(np.isnan(data), global_mean, data)
    
    # Calcular suma ponderada con FFT (¡SUPER RÁPIDO!)
    weighted_sum = fftconvolve(data_filled, kernel, mode='same')
    
    # Calcular número de vecinos válidos
    valid_mask = (~np.isnan(data)).astype(float)
    n_neighbors = fftconvolve(valid_mask, np.ones_like(kernel), mode='same')
    
    # Cálculo de Gi*
    with np.errstate(divide='ignore', invalid='ignore'):
        gi_star = (weighted_sum - global_mean * np.sum(kernel)) / (
            global_std * np.sqrt((np.sum(valid_mask) * np.sum(kernel**2) - 
                               (np.sum(kernel))**2) / (np.sum(valid_mask) - 1)))
    
    return gi_star

# EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    raster_path = "/home/rgrimson/Downloads/ndvi_caba.tif"
    
    # PARA DIFERENTES ESCALAS
    scales_to_analyze = [100, 200, 500]  # Escalas en metros
    
    results = {}
    for scale in scales_to_analyze:
        print(f"\n{'='*50}")
        print(f"ANALIZANDO ESCALA: {scale}m")
        print(f"{'='*50}")
        
        gi_result, reduced_data, kernel = compute_gistar_large_scale(
            raster_path, 
            target_scale_meters=scale,
            original_resolution=10
        )
        
        results[scale] = gi_result
        
        # Visualización rápida
        plt.figure(figsize=(10, 4))
        plt.imshow(gi_result, cmap='RdBu_r', vmin=-3, vmax=3)
        plt.colorbar()
        plt.title(f'Gi* - Escala {scale}m')
        plt.savefig(f'gistar_scale_{scale}m.png', dpi=150, bbox_inches='tight')
        plt.show()

#%%
# MORAN
#%%
import numpy as np
import rasterio
from scipy.sparse import coo_matrix
import libpysal
from esda.moran import Moran
import time

def compute_moran_global_raster(raster_path):
    """
    Calcula el I de Moran global para un raster NDVI
    """
    print("Cargando raster...")
    with rasterio.open(raster_path) as src:
        ndvi_data = src.read(1)
        transform = src.transform
        crs = src.crs
    
    # Vectorizar datos (excluyendo NaN)
    flat_data = ndvi_data.flatten()
    valid_mask = ~np.isnan(flat_data)
    y = flat_data[valid_mask]
    
    print(f"Píxeles válidos: {len(y)}/{len(flat_data)}")
    
    # Crear matriz de pesos espaciales para raster
    print("Creando matriz de pesos...")
    height, width = ndvi_data.shape
    w = create_raster_weight_matrix(height, width)
    
    # Calcular I de Moran
    print("Calculando I de Moran global...")
    start_time = time.time()
    
    moran = Moran(y, w, transformation='r', permutations=999)
    
    computation_time = time.time() - start_time
    print(f"Tiempo de cálculo: {computation_time:.2f} segundos")
    
    return moran, ndvi_data

def create_raster_weight_matrix(height, width):
    """
    Crea matriz de pesos para una rejilla regular (raster)
    usando contigüidad tipo reina (8 vecinos)
    """
    # Calcular número total de píxeles válidos
    n = height * width
    
    # Listas para matriz sparse
    row_indices = []
    col_indices = []
    values = []
    
    # Llenar la matriz de pesos
    for i in range(height):
        for j in range(width):
            current_idx = i * width + j
            
            # Vecinos tipo reina (8-connectivity)
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue  # Saltar el propio píxel
                    
                    ni, nj = i + di, j + dj
                    
                    # Verificar límites
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_idx = ni * width + nj
                        row_indices.append(current_idx)
                        col_indices.append(neighbor_idx)
                        values.append(1.0)  # Peso 1 para vecinos
    
    # Crear matriz sparse
    from scipy.sparse import coo_matrix
    w_sparse = coo_matrix((values, (row_indices, col_indices)), 
                         shape=(n, n))
    
    return w_sparse

# FUNCIÓN ALTERNATIVA MÁS RÁPIDA PARA RASTERS GRANDES
def compute_moran_fast(raster_path, sample_fraction=0.1):
    """
    Versión optimizada para rasters grandes usando muestreo
    """
    with rasterio.open(raster_path) as src:
        ndvi_data = src.read(1)
    
    # Muestrear píxeles para hacerlo manejable
    if sample_fraction < 1.0:
        flat_data = ndvi_data.flatten()
        valid_indices = np.where(~np.isnan(flat_data))[0]
        sample_size = int(len(valid_indices) * sample_fraction)
        sampled_indices = np.random.choice(valid_indices, sample_size, replace=False)
        y = flat_data[sampled_indices]
    else:
        flat_data = ndvi_data.flatten()
        valid_mask = ~np.isnan(flat_data)
        y = flat_data[valid_mask]
    
    print(f"Muestra: {len(y)} píxeles")
    
    # Para muestras grandes, usar aproximación
    if len(y) > 10000:
        print("Usando método aproximado para muestra grande...")
        from sklearn.metrics.pairwise import euclidean_distances
        from scipy.spatial.distance import squareform
        from scipy.sparse import lil_matrix
        
        # Crear matriz de distancias inversas
        coords = np.array([(i, j) for i in range(ndvi_data.shape[0]) 
                          for j in range(ndvi_data.shape[1]) 
                          if not np.isnan(ndvi_data[i, j])])[sampled_indices]
        
        # Muestrear coordenadas también
        if len(coords) > 10000:
            sample_idx = np.random.choice(len(coords), 10000, replace=False)
            coords = coords[sample_idx]
            y = y[sample_idx]
        
        dist_matrix = euclidean_distances(coords)
        inv_dist_matrix = 1 / (1 + dist_matrix)  # Suavizar para evitar división por 0
        
        # Convertir a matriz sparse
        w = lil_matrix(inv_dist_matrix)
        w.setdiag(0)  # Diagonal cero
    else:
        # Método exacto para muestras pequeñas
        w = create_raster_weight_matrix(ndvi_data.shape[0], ndvi_data.shape[1])
    
    # Calcular Moran
    moran = Moran(y, w,     ='r', permutations=99)  # Menas permutaciones para velocidad
    
    return moran, ndvi_data

# PROGRAMA PRINCIPAL
def main_moran_analysis(raster_path, method='fast'):
    """
    Análisis completo de I de Moran
    """
    print("=== ANÁLISIS I DE MORAN GLOBAL ===")
    
    if method == 'fast':
        moran, ndvi_data = compute_moran_fast(raster_path, sample_fraction=0.2)
    else:
        moran, ndvi_data = compute_moran_global_raster(raster_path)
    
    # RESULTADOS
    print("\n=== RESULTADOS I DE MORAN ===")
    print(f"I de Moran: {moran.I:.6f}")
    print(f"Valor-p: {moran.p_norm:.6f}")
    print(f"Significativo (p < 0.05): {moran.p_norm < 0.05}")
    print(f"Expectativa (E[I]): {moran.EI:.6f}")
    print(f"Varianza: {moran.VI:.6f}")
    print(f"Z-score: {moran.z_norm:.6f}")
    
    # INTERPRETACIÓN
    print("\n=== INTERPRETACIÓN ===")
    if moran.I > moran.EI:
        print("✅ Autocorrelación ESPACIAL POSITIVA")
        print("   Los valores similares tienden a agruparse")
        if moran.I > 0.7:
            print("   ➤ Fuertes clusters espaciales")
        elif moran.I > 0.3:
            print("   ➤ Clusters espaciales moderados")
        else:
            print("   ➤ Débil agrupamiento espacial")
    else:
        print("✅ Autocorrelación ESPACIAL NEGATIVA")
        print("   Los valores disimilares tienden a agruparse")
    
    if moran.p_norm < 0.05:
        print("   📊 Resultado ESTADÍSTICAMENTE SIGNIFICATIVO")
    else:
        print("   ⚠️  Resultado NO significativo (patrón aleatorio)")
    
    return moran, ndvi_data

# VISUALIZACIÓN DE RESULTADOS
def plot_moran_results(moran, ndvi_data):
    """Visualiza resultados del análisis Moran"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mapa NDVI
    im0 = axes[0].imshow(ndvi_data, cmap='viridis', vmin=-1, vmax=1)
    axes[0].set_title('NDVI Original')
    plt.colorbar(im0, ax=axes[0])
    
    # Diagrama de dispersión de Moran (simulado)
    axes[1].scatter([0], [moran.I], color='red', s=100)
    axes[1].axhline(y=moran.EI, color='blue', linestyle='--', label=f'E[I] = {moran.EI:.3f}')
    axes[1].axhline(y=0, color='black', linestyle='-')
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(-1, 1)
    axes[1].set_title(f'I de Moran = {moran.I:.3f} (p = {moran.p_norm:.3f})')
    axes[1].set_ylabel('I de Moran')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('moran_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# EJECUCIÓN
if __name__ == "__main__":
    raster_path = "/home/rgrimson/Downloads/GiS/ndvi_caba.tif"
    
    # Análisis rápido (recomendado para rasters grandes)
    moran_result, ndvi_data = main_moran_analysis(raster_path, method='fast')
    
    # Visualizar resultados
    plot_moran_results(moran_result, ndvi_data)
    
    # Guardar resultados en archivo
    with open('moran_results.txt', 'w') as f:
        f.write("RESULTADOS ANÁLISIS I DE MORAN\n")
        f.write("===============================\n")
        f.write(f"I de Moran: {moran_result.I:.6f}\n")
        f.write(f"Valor-p: {moran_result.p_norm:.6f}\n")
        f.write(f"Significativo: {moran_result.p_norm < 0.05}\n")
        f.write(f"Z-score: {moran_result.z_norm:.6f}\n")
        f.write(f"Expectativa E[I]: {moran_result.EI:.6f}\n")
        f.write(f"Varianza: {moran_result.VI:.6f}\n")
        
    #%%
def moran_final_solution(raster_path):
    """
    Solución definitiva para I de Moran en raster
    """
    import rasterio
    from libpysal.weights import lat2W
    from esda.moran import Moran
    
    # Cargar raster
    with rasterio.open(raster_path) as src:
        data = src.read(1)
    
    # Si el raster es muy grande, muestrear o reducir
    if data.size > 10000:
        # Muestrear 10000 píxeles aleatorios
        flat_data = data.flatten()
        valid_mask = ~np.isnan(flat_data)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 10000:
            sample_indices = np.random.choice(valid_indices, 10000, replace=False)
            values = flat_data[sample_indices]
            
            # Crear pesos de distancia inversa para puntos muestreados
            from libpysal.weights import DistanceBand
            coords = np.array([(i, j) for idx in sample_indices 
                             for i, j in [(idx//data.shape[1], idx%data.shape[1])]])
            w = DistanceBand(coords, threshold=10, binary=True)
            
            moran = Moran(values, w, permutations=99)
            
        else:
            # Usar todos los píxeles válidos
            values = flat_data[valid_mask]
            w = lat2W(data.shape[0], data.shape[1]).subset(valid_indices)
            moran = Moran(values, w, permutations=99)
    else:
        # Raster pequeño, procesar completo
        flat_data = data.flatten()
        valid_mask = ~np.isnan(flat_data)
        values = flat_data[valid_mask]
        w = lat2W(data.shape[0], data.shape[1]).subset(valid_indices)
        moran = Moran(values, w, permutations=99)
    
    return moran, data

#%%
# EJECUTAR
raster_path = '/home/rgrimson/Downloads/GiS/ndvi_chacalermo.tif'
raster_path = '/home/rgrimson/Downloads/GiS/ndvi_caba.tif'
moran, ndvi_data = moran_final_solution(raster_path)    

# Visualizar resultados
plot_moran_results(moran_result, ndvi_data)

# RESULTADOS
print("\n=== RESULTADOS I DE MORAN ===")
print(f"I de Moran: {moran.I:.6f}")
print(f"Valor-p: {moran.p_norm:.6f}")
print(f"Significativo (p < 0.05): {moran.p_norm < 0.05}")
print(f"Expectativa (E[I]): {moran.EI:.6f}")
print(f"Varianza: {moran.VI_norm:.6f}")
print(f"Z-score: {moran.z_norm:.6f}")

# INTERPRETACIÓN
print("\n=== INTERPRETACIÓN ===")
if moran.I > moran.EI:
    print("✅ Autocorrelación ESPACIAL POSITIVA")
    print("   Los valores similares tienden a agruparse")
    if moran.I > 0.7:
        print("   ➤ Fuertes clusters espaciales")
    elif moran.I > 0.3:
        print("   ➤ Clusters espaciales moderados")
    else:
        print("   ➤ Débil agrupamiento espacial")
else:
    print("✅ Autocorrelación ESPACIAL NEGATIVA")
    print("   Los valores disimilares tienden a agruparse")

if moran.p_norm < 0.05:
    print("   📊 Resultado ESTADÍSTICAMENTE SIGNIFICATIVO")
else:
    print("   ⚠️  Resultado NO significativo (patrón aleatorio)")
