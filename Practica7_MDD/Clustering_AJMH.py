#Practica 7. Agrupamiento de Datos (Clustering)
#Dataset: OnlineRetailLimpiado.csv para su procesamiento
#Nombre: Angel Joseph Meraz Hernandez
#Matricula: 2067151
"""
Crear y evaluar un modelo de agrupamiento mediante K-Means.
"""

#Primero que nada, aqui estan los imports necesarios para el K Mean:
#Proceso en el cual es el proceso por el cual se unican los registros, conforme
#a sus caracteristicas, en un espacio, y pone centroides de forma aleatoria y
#calculando su distancia con cada centroide, se elige la menor distacia con cada uno
#para asociar esos registros a ese centroide, luego se cambia de lugar al centroide
#promediando las ubicaciones de los registros y se le asigna ese promedio al centroide
#y el proceso se reinicia hasta que los centroides se dejen de mover.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

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
    """
    Genera seis caracteristicas:
        Recencia: dias desde la ultima compra registrada.
        Frecuencia: numero de facturas distintas.
        GastoTotal: suma del gasto registrado.
        CantidadTotal: suma de las unidades compradas.
        PrecioPromedio: precio unitario promedio de sus registros.
        ProductosDistintos: numero de articulos diferentes.
    """
    #Calcular el gasto de cada linea de compra
    df["Gasto"] = df["Cantidad"] * df["PrecioUnitario"]

    #Tomar como referencia el dia posterior a la ultima compra registrada
    fecha_referencia = df["FechaCompra"].max() + pd.Timedelta(days=1)

    #Agrupar los registros para obtener una fila por cliente
    datos = df.groupby("IDCliente").agg(
        Recencia=("FechaCompra", lambda fechas: (fecha_referencia - fechas.max()).days),
        Frecuencia=("IDFactura", "nunique"),
        GastoTotal=("Gasto", "sum"),
        CantidadTotal=("Cantidad", "sum"),
        PrecioPromedio=("PrecioUnitario", "mean"),
        ProductosDistintos=("IDArticulo", "nunique"))

    datos = datos.reset_index()
    """
    Se verifica que existan al menos tres clientes, ya que el modelo final esta
    configurado inicialmente para utilizar tres clusters.
    Si no existen suficientes clientes, se detiene el programa para evitar
    entrenar un modelo con datos insuficientes.
    """
    if len(datos) < 3:
        raise ValueError("No hay suficientes clientes para realizar el clustering.")

    print("\nCaracteristicas creadas por cliente:")
    print("1. Recencia: dias desde la ultima compra.")
    print("2. Frecuencia: cantidad de facturas distintas.")
    print("3. GastoTotal: gasto total registrado.")
    print("4. CantidadTotal: unidades compradas.")
    print("5. PrecioPromedio: precio unitario promedio.")
    print("6. ProductosDistintos: articulos distintos comprados.")

    print("\nCantidad de clientes para agrupar:", len(datos))
    print("\nPrimeros clientes:")
    print(datos.head().to_string(index=False))
    return datos
    


#Funcion para encontrar un numero razonable de clusters con la tecnica del codo
def tecnica_codo(X_escalado):
    """
    La tecnica del codo la vi en otra clase y es para explorar diferentes cantidades
    de clusters.
    La inercia mide la suma de las distancias cuadradas entre los registros y los
    centroides a los que fueron asignados.
    La tecnica ayuda a proponer un valor de k, pero no garantiza por si sola que
    sea la mejor cantidad de clusters.
    """
    #Probar desde 1 hasta 10 clusters, sin superar la cantidad de clientes
    max_k = min(10, len(X_escalado))
    valores_k = list(range(1, max_k + 1))
    inercias = []

    for k in valores_k:
        modelo = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        modelo.fit(X_escalado)
        inercias.append(modelo.inertia_)

    print("\nInercia por numero de clusteres")
    for k, inercia in zip(valores_k, inercias):
        print("k =", k, "| Inercia =", round(inercia, 2))

    #Graficar la tecnica del codo
    plt.figure(figsize=(8, 5))
    plt.plot(valores_k, inercias, marker="o")
    plt.xlabel("Numero de clusters (k)")
    plt.ylabel("Inercia")
    plt.title("Tecnica del Codo para K-Means")
    plt.xticks(valores_k)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    """
    Hice todos estos calculos porque dejar una muy baja cantidad de clusters hace
    que practicamente no se dividen los datos lo suficiente, por eso al final decidi
    dejarlo en 3 clusters del 0 al 2.
    """

#Funcion para entrenar el modelo final de K-Means
def modelo_kmeans(datos):
    #Caracteristicas que se utilizaran para formar los grupos
    variables = [
        "Recencia",
        "Frecuencia",
        "GastoTotal",
        "CantidadTotal",
        "PrecioPromedio",
        "ProductosDistintos"
    ]

    X = datos[variables].copy()
    escalador = RobustScaler()
    X_escalado = escalador.fit_transform(X)

    print("\nPreparacion del modelo K-Means")
    print("Caracteristicas utilizadas:", variables)
    print("Clientes analizados:", len(X))

    #Calcular y mostrar la tecnica del codo
    tecnica_codo(X_escalado)

    #Valor inicial propuesto: revisar las graficas y cambiarlo si corresponde
    k_final = 3
    if k_final >= len(datos):
        raise ValueError("El numero de clusters debe ser menor que los clientes.")
    modelo = KMeans(n_clusters=k_final, random_state=42, n_init=10)
    datos["Cluster"] = modelo.fit_predict(X_escalado)

    print("\nNumero de clusters seleccionado para el modelo final:", k_final)
    print("Revisa la tecnica del codo y el Silhouette Score para justificar este valor.")

    #Mostrar la cantidad de clientes en cada cluster
    tamanos = datos["Cluster"].value_counts().sort_index()
    print("\nCantidad de clientes por cluster")
    print(tamanos.to_string())

    #Mostrar el promedio de las caracteristicas de cada grupo
    resumen = datos.groupby("Cluster")[variables].mean().round(2)
    resumen.insert(0, "NumeroClientes", tamanos)

    print("\nPromedio de las caracteristicas por cluster")
    print(resumen.to_string())

    #Graficar dos caracteristicas para visualizar los grupos
    plt.figure(figsize=(10, 6))
    for cluster in sorted(datos["Cluster"].unique()):
        grupo = datos[datos["Cluster"] == cluster]
        plt.scatter(
            grupo["Recencia"],
            grupo["GastoTotal"],
            label="Cluster " + str(cluster),
            alpha=0.7,
            s=25
        )

    plt.xlabel("Recencia (dias desde la ultima compra)")
    plt.ylabel("Gasto total")
    plt.title("Clusters de clientes: recencia y gasto total")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #Mostrar solamente 5 clientes de cada cluster
    print("\nEjemplos de clientes por cluster")
    for cluster in sorted(datos["Cluster"].unique()):
        print("\nCluster", cluster)
        ejemplos = datos[datos["Cluster"] == cluster].head(10)
        print(ejemplos.to_string(index=False))
    
    print("\nAnalisis de clustering terminado.")
    print("Los resultados solo se muestran en pantalla; no se guardan archivos.")


#Programa principal
df = pd.read_csv("OnlineRetaiLimpio.csv")
if not df.empty:
    df = preparacion(df)
    
    datos = crear_datos(df)
    #La divicion de datos fue de un 80% para entrenar y un 20% para pruebas
    modelo_kmeans(datos)

else:
    print("El archivo no contiene datos.")

    

"""
Dataset cargado correctamente.

Numero de filas: 390885
Numero de columnas: 8

Caracteristicas creadas por cliente:
1. Recencia: dias desde la ultima compra.
2. Frecuencia: cantidad de facturas distintas.
3. GastoTotal: gasto total registrado.
4. CantidadTotal: unidades compradas.
5. PrecioPromedio: precio unitario promedio.
6. ProductosDistintos: articulos distintos comprados.

Cantidad de clientes para agrupar: 4330

Primeros clientes:
 IDCliente  Recencia  Frecuencia  GastoTotal  CantidadTotal  PrecioPromedio  ProductosDistintos
     12346       327           1    77183.60          74215        1.040000                   1
     12347        41           7     4310.00           2458        2.644011                 103
     12348        77           4     1437.24           2332        0.692963                  21
     12349        20           1     1457.55            630        4.237500                  72
     12350       312           1      294.40            196        1.581250                  16

Preparacion del modelo K-Means
Caracteristicas utilizadas: ['Recencia', 'Frecuencia', 'GastoTotal', 'CantidadTotal', 'PrecioPromedio', 'ProductosDistintos']
Clientes analizados: 4330

Inercia por numero de clusteres
k = 1 | Inercia = 524769.14
k = 2 | Inercia = 296152.96
k = 3 | Inercia = 185514.95
k = 4 | Inercia = 128406.04
k = 5 | Inercia = 103016.45
k = 6 | Inercia = 82299.21
k = 7 | Inercia = 64548.93
k = 8 | Inercia = 52359.48
k = 9 | Inercia = 46982.43
k = 10 | Inercia = 42425.12

Numero de clusters seleccionado para el modelo final: 3
Revisa la tecnica del codo y el Silhouette Score para justificar este valor.

Cantidad de clientes por cluster
Cluster
0    4315
1      13
2       2

Promedio de las caracteristicas por cluster
         NumeroClientes  Recencia  Frecuencia  GastoTotal  CantidadTotal  PrecioPromedio  ProductosDistintos
Cluster                                                                                                     
0                  4315    106.86        4.11     1619.88         967.36            3.29               60.27
1                    13     45.92       51.46   131009.20       75317.15            2.49              444.62
2                     2    163.50        2.50    20253.25          61.50          364.83                1.00

Ejemplos de clientes por cluster

Cluster 0
 IDCliente  Recencia  Frecuencia  GastoTotal  CantidadTotal  PrecioPromedio  ProductosDistintos  Cluster
     12347        41           7     4310.00           2458        2.644011                 103        0
     12348        77           4     1437.24           2332        0.692963                  21        0
     12349        20           1     1457.55            630        4.237500                  72        0
     12350       312           1      294.40            196        1.581250                  16        0
     12352        74           7     1385.74            526        4.075455                  57        0
     12353       205           1       89.00             20        6.075000                   4        0
     12354       234           1     1079.40            530        4.503793                  58        0
     12355        97           1      459.40            240        4.203846                  13        0
     12356        24           3     2487.43           1573        2.946034                  52        0
     12357       183           1     6207.67           2708        3.348626                 131        0

Cluster 1
 IDCliente  Recencia  Frecuencia  GastoTotal  CantidadTotal  PrecioPromedio  ProductosDistintos  Cluster
     12346       327           1    77183.60          74215        1.040000                   1        1
     12415        26          20   124564.53          77669        2.443469                 442        1
     13694        27          50    65039.62          63312        1.568996                 365        1
     14156         1          54   116560.08          57755        3.399949                 713        1
     14298        54          44    51527.30          58343        1.497978                 883        1
     14646         3          73   279138.02         197420        2.390063                 700        1
     14911         1         198   136161.83          80404        3.329934                1784        1
     16029        31          62    72708.09          40107        2.407917                  43        1
     16446        90           2   168472.50          80997        1.660000                   3        1
     16684        13          28    66653.56          50255        2.451625                 119        1

Cluster 2
 IDCliente  Recencia  Frecuencia  GastoTotal  CantidadTotal  PrecioPromedio  ProductosDistintos  Cluster
     13452       261           2       590.0              2          295.00                   1        2
     15098        66           3     39916.5            121          434.65                   1        2

Analisis de clustering terminado.
Los resultados solo se muestran en pantalla; no se guardan archivos.
"""
