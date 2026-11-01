#Practica 8. Pronóstico (Forecasting)
#Dataset: OnlineRetailLimpio.csv para su procesamiento
#Nombre: Angel Joseph Meraz Hernandez
#Matricula: 2067151
"""
Crear un modelo de regresión/serie de tiempo para predecir nuevos datos futuros.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Funcion para preparar y convertir los datos del dataset
def preparacion(df):
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
    #Asegurar que la cantidad sea numerica
    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce")
    #Eliminar registros sin fecha o cantidad valida
    df = df.dropna(subset=["FechaCompra", "Cantidad"])

    print("\nRegistros validos para el pronostico:", len(df))
    print("Fecha inicial:", df["FechaCompra"].min())
    print("Fecha final:", df["FechaCompra"].max())
    #print(df.dtypes)
    return df
    
#Funcion que preparo los datos, entreno el modelo y realice el pronostico
def pronostico(df):
    #Agrupe las cantidades vendidas durante cada dia
    datos = df.groupby(df["FechaCompra"].dt.floor("D"))["Cantidad"].sum()
    #Ordene las fechas y complete los dias sin ventas
    datos = datos.sort_index()
    datos = datos.asfreq("D", fill_value=0)
    #Converti los datos en un DataFrame
    datos = datos.to_frame(name="CantidadVendida")

    #Cree variables relacionadas con la fecha
    datos["DiaSemana"] = datos.index.dayofweek
    datos["Mes"] = datos.index.month
    datos["DiaMes"] = datos.index.day

    #Obtuve las ventas de dias anteriores
    datos["VentasDiaAnterior"] = datos["CantidadVendida"].shift(1)
    datos["VentasSemanaAnterior"] = datos["CantidadVendida"].shift(7)

    #Calcule el promedio de ventas de los siete dias anteriores
    datos["Promedio7Dias"] = datos["CantidadVendida"].shift(1).rolling(7).mean()

    #Elimine los registros que no tenian suficiente historial
    datos = datos.dropna()

    #Seleccione las variables independientes y dependiente
    X = datos.drop(columns=["CantidadVendida"])
    y = datos["CantidadVendida"]

    #Dividi los datos en entrenamiento (80%) y prueba (20%)
    #Mantuve el orden cronologico para evitar mezclar fechas
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    #Cree y entrene el modelo de regresion lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    #Realice las predicciones con los datos de prueba
    y_pred = modelo.predict(X_test)

    #Calcule las metricas para evaluar el modelo
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nResultados del modelo")
    print("Error Absoluto Medio (MAE):", mae) #Indica cuantas unidades se equivoco el modelo, mientras mas cercano a 0 mejor.
    print("Raíz del Error Cuadrático Medio (RMSE):", rmse) #Mientras mas cercano a 0 mejor.
    print("Coeficiente de Determinación (R^2):", r2)
    #>0 el modelo explicó perfectamente los datos
    #0 no mejoró respecto a predecir el promedio de las ventas
    #<0 el modelo tuvo un rendimiento peor que esa predicción basada en el promedio

    #Entrene nuevamente el modelo utilizando todos los datos disponibles
    modelo.fit(X, y)
    #Mostre la ecuacion de regresion lineal
    print("\nEcuacion del modelo:")
    ecuacion = f"Y = {modelo.intercept_:.4f}"
    for variable, coeficiente in zip(X.columns, modelo.coef_):
        signo = "+" if coeficiente >= 0 else "-"
        ecuacion += f" {signo} {abs(coeficiente):.4f}({variable})"
    print(ecuacion)

    #Copie el historial de ventas para realizar el pronostico futuro
    historial = datos["CantidadVendida"].copy()
    ultima_fecha = historial.index[-1]

    print("\nPronostico para los siguientes 7 dias")

    for i in range(1, 8):
        fecha = ultima_fecha + pd.Timedelta(days=i)

        #Prepare las variables correspondientes a la nueva fecha
        variables = pd.DataFrame([{
            "DiaSemana": fecha.dayofweek,
            "Mes": fecha.month,
            "DiaMes": fecha.day,
            "VentasDiaAnterior": historial.iloc[-1],
            "VentasSemanaAnterior": historial.iloc[-7],
            "Promedio7Dias": historial.tail(7).mean()
        }])

        #Calcule la cantidad de ventas estimada
        prediccion = modelo.predict(variables)[0]
        prediccion = max(0, prediccion)
        print(fecha.date(), ":", round(prediccion, 2))

        #Agregue la prediccion al historial para calcular el siguiente dia
        historial.loc[fecha] = prediccion


#Programa principal
df = pd.read_csv("OnlineRetaiLimpio.csv")
if not df.empty:
    print("Dataset cargado correctamente.\n")
    df = preparacion(df)
    """
    El objetivo de este programa fue desarrollar un modelo de pronostico
    mediante regresion lineal, utilizando el historial de ventas y las
    fechas de compra para estimar la cantidad de productos que se venderan
    durante los siguientes siete dias. Ademas, se evaluo el rendimiento
    del modelo mediante metricas de error
    """
    #Ejecute el modelo de pronostico
    pronostico(df)

    """
    Con esta práctica desarrollé un modelo de pronóstico mediante regresión
    lineal, utilizando las fechas de compra y el historial de ventas para estimar
    la cantidad de productos que se venderían durante los siguientes siete días.
    Al evaluar el modelo, obtuve un MAE de 8884.54 y un RMSE de 12242.81, lo que
    indicó que las predicciones presentaron errores considerables. Además, el
    coeficiente de determinación (R²) fue de 0.036, por lo que el modelo solamente
    explicó aproximadamente el 3.6% de la variación de las ventas. Esto demostró
    que la regresión lineal tuvo una capacidad limitada para representar el
    comportamiento de los datos. A pesar de estas limitaciones, el modelo permitió
    generar un pronóstico para los siguientes siete días. 
    """

else:
    print("El archivo no contiene datos.")


    
#Ejemplo de la impresion
"""
Dataset cargado correctamente.

Numero de filas: 390885
Numero de columnas: 8

Registros validos para el pronostico: 390885
Fecha inicial: 2010-01-12 08:26:00
Fecha final: 2011-12-10 17:19:00

Resultados del modelo
Error Absoluto Medio (MAE): 8884.543114393999
Raíz del Error Cuadrático Medio (RMSE): 12242.812881948912
Coeficiente de Determinación (R^2): 0.03599291822613215

Ecuacion del modelo:
Y = 4028.5824 - 634.6044(DiaSemana) + 4.1471(Mes) - 65.8395(DiaMes) + 0.1628(VentasDiaAnterior) + 0.1277(VentasSemanaAnterior) + 0.5683(Promedio7Dias)

Pronostico para los siguientes 7 dias
2011-12-11 : 13453.71
2011-12-12 : 19545.26
2011-12-13 : 14887.32
2011-12-14 : 14135.43
2011-12-15 : 13461.81
2011-12-16 : 13523.03
2011-12-17 : 12577.28
"""
