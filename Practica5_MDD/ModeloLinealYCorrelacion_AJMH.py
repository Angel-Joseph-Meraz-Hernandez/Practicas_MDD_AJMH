#Practica 5. Modelos Lineales y Correlación
#Dataset: OnlineRetailLimpiado.csv para su procesamiento
#Nombre: Angel Joseph Meraz Hernandez
#Matricula: 2067151
"""
Modelos Lineales y Correlación (Entrega: Semana 9)
Generar un modelo lineal, crear las gráficas correspondientes y obtener la
métrica $R^2$.
"""

#Importamos las librerías necesarias para el procesamiento de datos, creación del
#modelo lineal, cálculo de R^2 y generación de gráficas
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

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
    
    return df

#Funcion para identificar y eliminar valores atipicos en el precio de cada
#producto utilizando el método del rango intercuartílico (IQR).
def limpiar_atipicos(df):
    print("\nAnalizando precios atipicos de cada productos.")
    precios_limpios = []

    for articulo, grupo in df.groupby("IDArticulo"):
        if len(grupo) < 4:
            precios_limpios.append(grupo)
            continue

        Q1 = grupo["PrecioUnitario"].quantile(0.25)
        Q3 = grupo["PrecioUnitario"].quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        grupo_limpio = grupo[
            (grupo["PrecioUnitario"] >= limite_inferior) &
            (grupo["PrecioUnitario"] <= limite_superior)
        ]

        precios_limpios.append(grupo_limpio)

    df_limpio = pd.concat(precios_limpios)

    print("Registros originales:", len(df))
    print("Registros despues de quitar atipicos:", len(df_limpio))
    print("Registros considerados atipicos:", len(df) - len(df_limpio))
    return df_limpio

#Funcion para preparar las variables que seran utilizadas en el modelo de
#regresión lineal
def crear_datos_modelo(df):
    #Seleccionamos las variables que utilizaremos para el modelo lineal.
    datos = df[["PrecioUnitario", "Cantidad"]].copy()

    #Eliminamos posibles valores faltantes.
    datos = datos.dropna()

    return datos

#Funcion encargada de crear, entrenar y evaluar el modelo lineal
def modelo_lineal(datos):
    #Variable independiente
    X = datos[["PrecioUnitario"]]

    #Variable dependiente
    y = datos["Cantidad"]

    #Dividimos los datos en un conjunto de entrenamiento (80%) y otro de prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    #Aqui creo el modelo de regresion lineal
    modelo = LinearRegression()

    #Entreno el modelo
    modelo.fit(X_train, y_train)

    #Genero las predicciones en base del modelo
    y_pred = modelo.predict(X_test)

    #Calculamos el coeficiente R^2 para evaluar que tan bien funciona el modelo
    r2 = r2_score(y_test, y_pred)

    #Calculamos la correlacion
    correlacion = datos["PrecioUnitario"].corr(datos["Cantidad"])

    print("\nModelo lineal:")
    print("Intercepto:", modelo.intercept_)
    print("Coeficientes:", modelo.coef_)
    print("\nCorrelacion:", round(correlacion, 4))
    print("R^2:", r2)
    #Aqui evaluo el valor de R^2 para ver si el modelo sirve
    if r2 >= 0.70:
        print("Evaluacion: El modelo presenta un buen ajuste.")
    elif r2 >= 0.50:
        print("Evaluacion: El modelo presenta un ajuste moderado.")
    else:
        print("Evaluacion: El modelo presenta un ajuste bajo.")

    return modelo, X_test, y_test, y_pred


def grafica_regresion(modelo, X_test,y_test, y_pred):
    plt.figure(figsize=(10, 6))

    #Mostramos los valores reales 
    plt.scatter(
        X_test["PrecioUnitario"],
        y_test,
        alpha=0.5,
        label="Datos reales"
    )

    #Ordenamos los datos para dibujar correctamente la línea de regresión.
    datos_grafica = X_test.copy()

    datos_grafica["Prediccion"] = y_pred

    datos_grafica = datos_grafica.sort_values("PrecioUnitario")

    # Dibujamos la línea del modelo.
    plt.plot(
        datos_grafica["PrecioUnitario"],
        datos_grafica["Prediccion"],
        label="Regresión lineal"
    )

    plt.xlabel("Precio unitario")
    plt.ylabel("Cantidad comprada")
    plt.title("Regresión lineal: Precio unitario vs Cantidad")

    plt.legend()
    plt.tight_layout()
    plt.show()

#Programa principal
df = pd.read_csv("OnlineRetaiLimpio.csv")
if not df.empty:
    df = preparacion(df)
    
    df_limpio = limpiar_atipicos(df)

    #Crear datos para el modelo
    datos = crear_datos_modelo(df_limpio)

    #Crear y evaluar modelo
    modelo, X_test, y_test, y_pred = modelo_lineal(datos)

    #Mostrar grafica
    grafica_regresion(modelo, X_test,y_test, y_pred)


    #Con el valor de R^2 se observa que el precio unitario explica una proporcion
    #muy pequeña de la variacion en la cantidad comprada.
    #Por lo tanto, en este modelo no se encuentra una relacion lineal significativa
    #entre el precio unitario y la cantidad comprada.
    #Es decir, que un producto tenga un precio menor no implica necesariamente que
    #se compre una mayor cantidad.

"""
IMPRIME ALGO COMO LO SIGUIENTE:
Dataset cargado correctamente.

Numero de filas: 390885
Numero de columnas: 8

Analizando precios atipicos de cada productos.
Registros originales: 390885
Registros despues de quitar atipicos: 361438
Registros considerados atipicos: 29447

Modelo lineal:
Intercepto: 11.923855481205283
Coeficientes: [-0.79937552]

Correlacion: -0.0215
R^2: 0.01136092006384326
Evaluacion: El modelo presenta un ajuste bajo.
"""
#Ademas de la grafica con una linea de la funcion desviandose demostrando que el modelo no funciona para predecir la cantidad comprada
