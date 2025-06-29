"""
Earth mover's distance (EMD)

Cuauhtli Porras Diaz 2086192}

Este programa lo que hace es calcular el EMD  de dos distribuciones de puntos de igual tamaño, este código calcula una 
matriz de distancias de todos los puntos (entre puntos de las dos distribuciones) y lo que hace esencialemnte es encontrar una suma de distancias 
que sea mínima. Visto de otro forma este problema se reduce a un problema de transporte, que al considerar que nuestros puntos
tienen todos oferta y demanda igual a uno esto vendría a ser un problema de asignación de tareas pudiendo así implementar el algorítmo Húngaro. 
"""
import numpy as np
from itertools import product
import pandas as pd
import os

# Este bloque de códigos estará destinada a implementar el algoritmo Húngaro para obtener la asignación óptima

def matrix_reduction(m):
    """
    Esta función lo que hace es devolver la matriz reducida por filas y columnas, 
    lo que constituye el paso 1 y 2 del algoritmo Hungaro. 

    Parameters
    ----------
    m : matriz nxn
    """
    n = len(m[:,1])
    minimos_filas = m.min(axis=1)
    for i in range(n):
        m[i,:] = m[i,:] - minimos_filas[i]
    minimos_columns = m.min(axis=0)
    for i in range(n):
        m[:,i] = m[:,i] - minimos_columns[i]
    return m

def cover_with_lines(m):
    """
    Este función lo que hace es cubrir ceros con el mínimo número de lineas verticales
    y horizontales, si el numéro de lineas es igual a n entonces el problema de optimización 
    termina y devuelve True. En caso de que el número de lineas sea menor a n entonces
    devuelve dos listas, la primera con las filas con lineas y la segunda con las columnas 
    con lineas.

    Parameters
    ----------
    m : matriz nxn
    """
    n = len(m[:,1])
    matriz_asignaciones = np.full((n,n), 1)
    columnas_ocupadas = []
    filas_marcadas = []
    columnas_marcadas = []
    # El siguiente loop for llena la matriz_asignaciones con ceros en lo ceros emparejados y -1 en los otros ceros
    for i in range(n):
        for j in range(n):
            if (m[i][j] == 0) and (j not in columnas_ocupadas):
                matriz_asignaciones[i][j] = 0
                columnas_ocupadas.append(j)
            elif (m[i][j] == 0) and (j in columnas_ocupadas):
                matriz_asignaciones[i][j] = -1
    # Paso 1; marcar filas que no tienen ceros emparejados 
    for i in range(n):
        if (-1 in matriz_asignaciones[i,:]) and (0 not in matriz_asignaciones[i,:]):
            filas_marcadas.append(i)
    # Paso 2; marcar columnas en los ceros de las filas marcadas (ya sea ceros emparejados (0) o ceros normales (-1))
    for i in filas_marcadas:
        for j in range(n):
            if (matriz_asignaciones[i][j] == 0) or (matriz_asignaciones[i][j] == -1):
                if j not in columnas_marcadas:
                    columnas_marcadas.append(j)
    # Paso 3; marcar filas que tinen ceros emparejados en las columnas 
    for j in columnas_marcadas:
        for i in range(n):
            if matriz_asignaciones[i][j] == 0:
                if i not in filas_marcadas:
                    filas_marcadas.append(i)
    # Paso 4; repetir paso 2 y 3 hasta ya no tener nuevas filas o columnas marcadas
    while True:
        filas_marcadas1 = filas_marcadas.copy()
        columnas_marcadas1 = columnas_marcadas.copy()
        for i in filas_marcadas:
            for j in range(n):
                if (matriz_asignaciones[i][j] == 0) or (matriz_asignaciones[i][j] == -1):
                    if j not in columnas_marcadas:
                        columnas_marcadas.append(j)
        for j in columnas_marcadas:
            for i in range(n):
                if matriz_asignaciones[i][j] == 0:
                    if i not in filas_marcadas:
                        filas_marcadas.append(i)
        if (filas_marcadas == filas_marcadas1) and (columnas_marcadas == columnas_marcadas1):
            break
    columnas_linea = columnas_marcadas.copy()
    columnas_linea.sort()
    filas_linea = []
    for i in range(n):
        if i not in filas_marcadas:
            filas_linea.append(i)
    if (len(filas_linea) + len(columnas_linea)) == n:
        return True
    else:
        return filas_linea, columnas_linea

def modify_matrix(m, f, c):
    """
    Una vez que se han calculado las lineas en la matriz y su número fue menor a n, luego esta se debe
    modificar a otra equivalente donde sea más fáci obtener la configuración óptima

    Parameters
    ----------
    m : matriz (ya reducida)
    f : arreglo de los índices de filas cubiertas por lineas
    c : arreglo de los índices de las columnas cubiertas por lineas
    """
    n = m.shape[0]
    valores_no_cubiertos = []
    #El siguiente loop for llena una lista con los valores o celdas que no fueron cubiertas por las lineas 
    for i in range(n):
        for j in range(n):
            if (i not in f) and (j not in c):
                valores_no_cubiertos.append(m[i][j])
    minimo = min(valores_no_cubiertos)
    # Ahora restamos este mínimo valor a las filas no cubiertas 
    for i in range(n):
        if i not in f:
            m[i,:] = m[i,:] - minimo
    # Luego sumamos ese mismo valor a las columnas
    for j in range(n):
        if j in c:
            m[:,j] = m[:,j] + minimo
    return m

def encontrar_conf(m):
    """
    Esta función lo que hace es encontrar la configuración óptima entre las permutaciones posibles de ceros disponibles

    Parameters
    ----------
    m : matriz que ya fue cubierta por un número mínimo de lineas igual a n 
    """
    # El siguiente loop for retorna una lista con listas de los indices de columnas en cada fila donde hay ceros
    opciones_por_fila = []
    for fila in m:
        indices_ceros = [ i for i, valor in enumerate(fila) if valor == 0]
        opciones_por_fila.append(indices_ceros)
    # El siguiente loop for calcula todas las combinaciones de ceros de cada fila y retorna aquella donde no se repite ninguna columna
    for combinacion in product(*opciones_por_fila):
        if len(set(combinacion)) == m.shape[0]:  #El set elimina elementos repetidos
            return combinacion
    return None

def hungaro(m):
    """
    Esta función comprende todo el funcionamiento del algoritmo hungaro y retornará la suma de la combinación óptima

    Parameters
    ----------
    m : matriz de la que se quiere obtener la asignación optima
    """
    matriz_simplificada = m.copy()
    matriz_simplificada = matrix_reduction(matriz_simplificada)
    while True:
        if cover_with_lines(matriz_simplificada) == True:
            configuracion = encontrar_conf(matriz_simplificada)
            break
        else:
            f, c = cover_with_lines(matriz_simplificada)
            matriz_simplificada = modify_matrix(matriz_simplificada,f,c)
    suma = 0
    for i in range(len(configuracion)):
        suma += m[i][configuracion[i]]
    return suma

# El siguiente bloque de código ya ataca el problema del calculo del EMD 

def matriz_distancias(df):
    """
    Esta función recibe un DataFrame donde la primeras dos columnas son de los valores x1 y y1 y la tercera y cuarta de x2 y y2 respectivamente
    deben ser la misma cantidad para ambas distribuciones, tambien admite para tres dimensiones o caso unidimensional

    Parameters
    ----------
    df : dataframe de los datos de las coordenadas de los puntos.
    """
    if df.shape[1] == 2:
        matriz = np.zeros((df.shape[0],df.shape[0]))
        n = df.shape[0]
        for i in range(n):
            for j in range(n):
                d = np.abs(df.iloc[i,0] - df.iloc[j,1])
                matriz[i][j] = d
        return matriz
    elif df.shape[1] == 4:
        matriz = np.zeros((df.shape[0],df.shape[0]))
        n = df.shape[0]
        for i in range(n):
            for j in range(n):
                d = np.sqrt((df.iloc[i,0]-df.iloc[j,2])**2 + (df.iloc[i,1]-df.iloc[j,3])**2)
                matriz[i][j] = d
        return matriz
    elif df.shape[1] == 6:
        matriz = np.zeros((df.shape[0],df.shape[0]))
        n = df.shape[0]
        for i in range(n):
            for j in range(n):
                d = np.sqrt((df.iloc[i,0]-df.iloc[j,3])**2 + (df.iloc[i,1]-df.iloc[j,4])**2 + (df.iloc[i,2]-df.iloc[j,5])**2)
                matriz[i][j] = d
        return matriz

def emd(df):
    """
    Esta función calcula el EMD de dos distribuciones de puntos donde ambas tienen la misma cantidad de puntos

    Parameters
    ----------
    df : DataFrame que contiene las coordenadas de los puntos
    """
    matriz = matriz_distancias(df)
    n = matriz.shape[0]
    suma_distancias = hungaro(matriz)
    return suma_distancias/n
        



df = pd.read_csv("coordenadas.csv")
emd(df)



        