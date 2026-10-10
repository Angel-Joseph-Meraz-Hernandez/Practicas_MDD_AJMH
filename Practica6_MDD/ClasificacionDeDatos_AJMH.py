#Practica 6. Clasificación de Datos
#Dataset: OnlineRetailLimpiado.csv para su procesamiento
#Nombre: Angel Joseph Meraz Hernandez
#Matricula: 2067151
"""
Crear y evaluar un modelo de clasificación usando el algoritmo de K Vecinos Más
Cercanos (KNN)
"""

#Primero que nada, aqui estan los imports necesarios para el KNN, los tomo de una
#clase distinta donde use este mismo algoritmo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay)

#Funcion para preparar y convertir los datos del dataset
def preparacion(df):
    print("Dataset cargado correctamente.\n")
    print("Numero de filas:", df.shape[0])
    print("Numero de columnas:", df.shape[1])
    df["IDFactura"] = df["IDFactura"].astype(object)
    df["IDArticulo"] = df["IDArticulo"].astype(object)
    df["Descripcion"] = df["Descripcion"].astype(object)
    #por alguna razon lo leia como flotante, asi que lo converti a int y luego a objeto
    df["IDCliente"] = df["IDCliente"].astype(int)
    df["IDCliente"] = df["IDCliente"].astype(object)
    df["Pais"] = df["Pais"].astype(object)
    #Al recuperarlos se leia la fecha en str y al volverla fecha se perdian los datos, con el mixed sea arregla
    df["FechaCompra"] = pd.to_datetime(df["FechaCompra"], format = "mixed", errors = "coerce")
    #print(df.dtypes)
    return df

def crear_datos(df):
    #Ordenar las fechas disponibles
    fechas = np.sort(df["FechaCompra"].dropna().unique())

    if len(fechas) < 2:
        raise ValueError("No hay suficientes fechas para el analisis.")

    #Separar el historial del periodo futuro
    punto_corte = int(len(fechas) * 0.80)
    punto_corte = max(1, min(punto_corte, len(fechas) - 1))

    fecha_corte = fechas[punto_corte]
    historico = df[df["FechaCompra"] < fecha_corte].copy()
    futuro = df[df["FechaCompra"] >= fecha_corte].copy()

    print("\nFecha de corte:", pd.Timestamp(fecha_corte))
    print("Registros historicos:", len(historico))
    print("Registros futuros:", len(futuro))

    #Agrupar las ventas historicas por producto y pais
    grupo = historico.groupby(["IDArticulo", "Pais"], observed=True)

    datos = grupo.agg(CantidadHistorica=("Cantidad", "sum"),
        PrecioPromedio=("PrecioUnitario", "mean"),
        FacturasHistoricas=("IDFactura", "nunique"),
        ClientesHistoricos=("IDCliente", "nunique"),
        MesesActivos=("FechaCompra", lambda fechas: fechas.dt.to_period("M").nunique()))

    #Calcular la cantidad vendida en el ultimo mes del historial
    ultimo_mes = (historico.assign(Mes=historico["FechaCompra"].dt.to_period("M"))
        .groupby(["IDArticulo", "Pais", "Mes"], observed=True)["Cantidad"]
        .sum()
        .reset_index())

    #Identificar el ultimo mes disponible en el historial
    mes_corte = pd.Timestamp(fecha_corte).to_period("M")
    ultimo_mes_historico = mes_corte - 1

    ventas_recientes = ultimo_mes[ultimo_mes["Mes"] == ultimo_mes_historico].set_index(["IDArticulo", "Pais"])["Cantidad"]

    datos["CantidadUltimoMes"] = (ventas_recientes.reindex(datos.index).fillna(0))

    #Incluir el mes del calendario como variable temporal
    datos["MesUltimoHistorico"] = ultimo_mes_historico.month

    #Calcular las ventas reales durante el periodo futuro
    ventas_futuras = futuro.groupby(["IDArticulo", "Pais"], observed=True)["Cantidad"].sum()

    datos["CantidadFutura"] = (ventas_futuras.reindex(datos.index).fillna(0))

    #Crear categorias de demanda usando las ventas futuras positivas
    positivas = datos.loc[datos["CantidadFutura"] > 0, "CantidadFutura"]

    if len(positivas) < 3:
        raise ValueError("No hay suficientes productos con ventas futuras.")

    #Clasificar segun la posicion de las ventas positivas
    posiciones = positivas.rank(method="average", pct=True)

    datos["CategoriaDemanda"] = "Baja"
    datos.loc[posiciones[posiciones > 0.50].index, "CategoriaDemanda"] = "Media"
    datos.loc[posiciones[posiciones > 0.80].index, "CategoriaDemanda"] = "Alta"
    
    print("\nDistribucion de la demanda futura:")
    print(datos["CategoriaDemanda"].value_counts())

    # Convertir IDArticulo y Pais del indice a columnas
    datos = datos.reset_index()

    print("\nCantidad de combinaciones producto-pais:", len(datos))
    return datos



#Funcion para entrenar y evaluar el modelo KNN
def modelo_knn(datos):
    #Variables numericas que utilizara el modelo
    variables_numericas = [
        "CantidadHistorica",
        "PrecioPromedio",
        "FacturasHistoricas",
        "ClientesHistoricos",
        "MesesActivos",
        "CantidadUltimoMes",
        "MesUltimoHistorico"
    ]

    #El pais se procesara como una variable categorica
    variables = variables_numericas + ["Pais"]

    X = datos[variables].copy()
    y = datos["CategoriaDemanda"].copy()

    #Comprobar que existan las tres categorias
    if y.nunique() < 3:
        raise ValueError("No se generaron las tres categorias de demanda.")

    if y.value_counts().min() < 5:
        raise ValueError("Hay muy pocos ejemplos en alguna categoria para realizar la validacion cruzada.")

    print("\nPreparacion del modelo KNN\nVariables predictoras:", variables)
    print("Productos-pais para entrenamiento y prueba:", len(X))

    #Separar los datos en entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
    
    #Escalar las variables numericas y codificar el pais
    preparador = ColumnTransformer([("numericas", RobustScaler(), variables_numericas), ("pais", OneHotEncoder(handle_unknown="ignore"), ["Pais"])])

    #Crear el pipeline de KNN
    modelo = Pipeline([("preparacion", preparador), ("knn", KNeighborsClassifier())])

    #Probar distintos valores de K
    valores_k = [5, 9, 15]
    parametros = {
        "knn__n_neighbors": valores_k,
        "knn__weights": ["uniform", "distance"]
    }

    #Buscar los mejores parametros mediante validacion cruzada
    busqueda = GridSearchCV(
        modelo,
        parametros,
        cv=3,
        scoring="f1_macro",
        n_jobs=1
    )
    busqueda.fit(X_entrenamiento, y_entrenamiento)
    mejor_modelo = busqueda.best_estimator_

    print("\nMejores parametros:", busqueda.best_params_)
    print("Esto quiere decir que el mejor modelo de los 3 que puse el que dio mejores resultados fue el de 15 vecinos mas cercanos con pesos uniformes")
    print("F1 macro en validacion cruzada:", round(busqueda.best_score_, 4))
    print("El F1 macro surge de promediar los F1-score de cada clasificacion, indica que esta bastante bien el modelo para clasificar si todas los tipos de demanda tienen la misma importancia")

    #Predecir la categoria de demanda
    y_predic = mejor_modelo.predict(X_prueba)

    #Evaluar el resultado
    exactitud = accuracy_score(y_prueba, y_predic)

    print("\nResultados del modelo KNN")
    print("\nAccuracy:", round(exactitud, 4))
    print("\nReporte de clasificacion:")
    print(classification_report(y_prueba, y_predic, labels=["Baja", "Media", "Alta"], zero_division=0))

    #Crear la matriz de confusion
    matriz = confusion_matrix(y_prueba, y_predic, labels=["Baja", "Media", "Alta"])
    matriz_texto = pd.DataFrame(matriz,
        index=["Real Baja", "Real Media", "Real Alta"],
        columns=["Predicha Baja", "Predicha Media", "Predicha Alta"])

    print("\nMatriz de confusion")
    print("Las filas representan la categoria real, mientras que las columnas representan la categoria predicha.")
    print(matriz_texto.to_string())


    #Graficar las regiones de decision del modelo KNN
    categorias = ["Baja", "Media", "Alta"]

    #Colores para cada categoria
    colores = ["#78B7E5", "#F4C05E", "#77C98B"]
    mapa_colores = ListedColormap(colores)

    #Definir los limites de la grafica para evitar que valores extremos dificulten la visualizacion
    x_min = X_entrenamiento["CantidadHistorica"].quantile(0.01)
    x_max = X_entrenamiento["CantidadHistorica"].quantile(0.99)
    y_min = X_entrenamiento["PrecioPromedio"].quantile(0.01)
    y_max = X_entrenamiento["PrecioPromedio"].quantile(0.99)

    #Crear una malla de puntos para mostrar las regiones
    eje_x = np.linspace(x_min, x_max, 70)
    eje_y = np.linspace(y_min, y_max, 70)
    XX, YY = np.meshgrid(eje_x, eje_y)

    # Mantener las demas variables constantes:
    # variables numericas en su mediana y pais en su moda
    valores_base = X_entrenamiento[variables_numericas].median()
    pais_base = X_entrenamiento["Pais"].mode().iloc[0]

    datos_malla = pd.DataFrame({
        columna: np.full(XX.size, valores_base[columna])
        for columna in variables_numericas
    })

    datos_malla["Pais"] = pais_base

    #Variar unicamente las dos variables de la grafica
    datos_malla["CantidadHistorica"] = XX.ravel()
    datos_malla["PrecioPromedio"] = YY.ravel()

    #Predecir la categoria en cada punto de la malla
    predicciones_malla = mejor_modelo.predict(datos_malla)

    #Convertir las categorias a numeros para dibujar las regiones
    equivalencias = {"Baja": 0, "Media": 1, "Alta": 2}
    Z = np.array([equivalencias[c] for c in predicciones_malla])
    Z = Z.reshape(XX.shape)

    #Dibujar las regiones de decision
    plt.figure(figsize=(10, 6))
    plt.contourf(
        XX, YY, Z,
        levels=[-0.5, 0.5, 1.5, 2.5],
        cmap=mapa_colores,
        alpha=0.35
    )

    #Mostrar los productos del conjunto de prueba
    for i, categoria in enumerate(categorias):
        seleccion = y_prueba == categoria

        plt.scatter(
            X_prueba.loc[seleccion, "CantidadHistorica"],
            X_prueba.loc[seleccion, "PrecioPromedio"],
            color=colores[i],
            edgecolors="black",
            linewidths=0.4,
            s=22,
            alpha=0.8,
            label="Real: " + categoria
        )

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Cantidad historica vendida")
    plt.ylabel("Precio promedio")
    plt.title("Regiones de decision del modelo KNN")
    plt.legend()
    plt.tight_layout()
    plt.show()

    



#Programa principal
df = pd.read_csv("OnlineRetaiLimpio.csv")
if not df.empty:
    df = preparacion(df)
    """
    La pregunta de lo que hara el modelo es simple: ¿Podemos estimar si un
    producto tendrá una demanda baja, media o alta en el periodo futuro a partir
    de su historial de ventas, precio, clientes, facturas y país?
    """
    datos = crear_datos(df)
    #La divicion de datos fue de un 80% para entrenar y un 20% para pruebas
    modelo_knn(datos)

    """
    La exactitud es de casi el 89% el modelo clasifico casi 9 de cada 10
    casos correctamente en su respectiva categoria de bajo, medio y alto. De la
    demanda baja se clasifico bien el 98% de los casos (serun el recall), mientras
    que el siguiente mejor clasificado fueron los del alta demanda con un 79%, el
    peor clasificado fue el de demanda media con un 35% ya que su mayoria fue clasificado
    como baja demanda (segun la matriz de confusión.

    Esta misma clasificacion se muestra en la grafica donde casi todos los puntos
    azules (baja demanda) estan en la zona de su mismo color, y de los verdes (demanda
    alta) gran parte se concentran en la franja verde de enmedio, pero la mayoria de
    los puntos amarillos (clasificacion media) estan en la parte azul.
    """
else:
    print("El archivo no contiene datos.")

    

"""
Dataset cargado correctamente.

Numero de filas: 390885
Numero de columnas: 8

Fecha de corte: 2011-10-16 16:09:00
Registros historicos: 303129
Registros futuros: 87756

Distribucion de la demanda futura:
CategoriaDemanda
Baja     13253
Media     1983
Alta      1322
Name: count, dtype: int64

Cantidad de combinaciones producto-pais: 16558

Preparacion del modelo KNN
Variables predictoras: ['CantidadHistorica', 'PrecioPromedio', 'FacturasHistoricas', 'ClientesHistoricos', 'MesesActivos', 'CantidadUltimoMes', 'MesUltimoHistorico', 'Pais']
Productos-pais para entrenamiento y prueba: 16558

Mejores parametros: {'knn__n_neighbors': 15, 'knn__weights': 'uniform'}
Esto quiere decir que el mejor modelo de los 3 que puse el que dio mejores resultados fue el de 15 vecinos mas cercanos con pesos uniformes
F1 macro en validacion cruzada: 0.7144
El F1 macro surge de promediar los F1-score de cada clasificacion, indica que esta bastante bien el modelo para clasificar si todas los tipos de demanda tienen la misma importancia

Resultados del modelo KNN

Accuracy: 0.8919

Reporte de clasificacion:
              precision    recall  f1-score   support

        Baja       0.92      0.98      0.95      2691
       Media       0.65      0.35      0.46       392
        Alta       0.78      0.79      0.78       229

    accuracy                           0.89      3312
   macro avg       0.78      0.71      0.73      3312
weighted avg       0.88      0.89      0.88      3312


Matriz de confusion
Las filas representan la categoria real, mientras que las columnas representan la categoria predicha.
            Predicha Baja  Predicha Media  Predicha Alta
Real Baja            2635              45             11
Real Media            212             139             41
Real Alta              19              30            180
"""
