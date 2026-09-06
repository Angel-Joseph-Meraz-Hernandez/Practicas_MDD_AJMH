#Practica 2. Estadistica Descriptiva
#Dataset: Online Retail Limpiado para su procesamiento
#Nombre: Angel Joseph Meraz Hernandez
#Matricula: 2067151

import pandas as pd
import matplotlib.pyplot as plt

def estadistica_descrip(df):
    print("\nDescripcion generica de los datos:")
    print(df.describe())
    #Asimetria: Representa hacia que lado esta la asimetria: >0 a la derecha, <0 a la izquierda
    #Kurtosis: Representa que tan alejado esta un valor extremo de su distribucion normal
    
    #Estadisticas adicionales de Cantidad
    print("\nEstadisticas extra de Cantidad:")
    print("Moda:", df["Cantidad"].mode().tolist())
    print("Sumatoria:", df["Cantidad"].sum())
    print("Varianza:", df["Cantidad"].var())
    print("Asimetria:", df["Cantidad"].skew())
    print("Kurtosis:", df["Cantidad"].kurt())

    #Estadisticas adicionales de PrecioUnitario
    print("\nEstadisticas extra de PrecioUnitario:")
    print("Moda:", df["PrecioUnitario"].mode().tolist())
    print("Sumatoria:", df["PrecioUnitario"].sum())
    print("Varianza:", df["PrecioUnitario"].var())
    print("Asimetria:", df["PrecioUnitario"].skew())
    print("Kurtosis:", df["PrecioUnitario"].kurt())


def diagr_metric_dagrupados(df):
    print("\nMetricas agrupadas por articulo:")
    """
    Las columnas representanta:
        IDArticulo. Un poco obvio
        Descripcion. Otro obvio
        CantidadTotal. Total de unidades vendidas de ese articulo
        CantidadApariciones. Cantidad de registros en los que aparece
        CantidadPromedio. Promedio de unidades por registro de ese producto
        CantidadMinima. Menor cantidad de unidades compradas en un registro
        CantidadMaxima. Mayor cantidad de unidades compradas en un registro
    """
    metricas_articulo = df.groupby(["IDArticulo", "Descripcion"]).agg(
        CantidadTotal = ("Cantidad", "sum"),
        CantidadApariciones = ("Cantidad", "count"),
        CantidadPromedio = ("Cantidad", "mean"),
        CantidadMinima = ("Cantidad", "min"),
        CantidadMaxima = ("Cantidad", "max"),
        CantidadVarianza = ("Cantidad", "var"),
        CantidadDesviacion = ("Cantidad", "std"),
        PrecioPromedio = ("PrecioUnitario", "mean"),
        PrecioMinimo = ("PrecioUnitario", "min"),
        PrecioMaximo = ("PrecioUnitario", "max")
    )

    """
    Columnas adicionales a las metricas de articulo (no se me ocurrio como ponerlos de forma facil):
        CantidadModa. Cantidad de unidades compradas en un registro mas repetida
        CantidadAsimetria. Indica hacia que lado estan los valores mas grandes, cerca a cero cerca del promedio, mayor a 0 muy a la derecha, menor a cero muy a la izquierda
        CantidadKurtosis. Indica que tan normal es que las cantidades compradas se vayan a los extremos
        Si es NA, porbablemente se porque como solo hay un registro que los compro ese es su referente y por lo tanto esta en el centro
    """
    moda_articulo = df.groupby(["IDArticulo", "Descripcion"])["Cantidad"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )
    asimetria_articulo = df.groupby(["IDArticulo", "Descripcion"])["Cantidad"].skew()
    kurtosis_articulo = df.groupby(["IDArticulo", "Descripcion"])["Cantidad"].apply(pd.Series.kurt)

    metricas_articulo["CantidadModa"] = moda_articulo
    metricas_articulo["CantidadAsimetria"] = asimetria_articulo
    metricas_articulo["CantidadKurtosis"] = kurtosis_articulo

    print("\nMetricas por articulo:")
    print(metricas_articulo.head(10))
    

    print("\nMetricas agrupadas por cliente:")
    """
    Las columnas representan:
        IDCliente. No necesito explicar
        Pais. El primer pais asociado al cliente
        CantidadArticComprados. Cantidad de articulos de la tienda (sin que importe el ID) que el cliente compro
        CantidadApariciones. Cantidad de registros en los que aparece el cliente.
        CantidadPromedio. Promedio de unidades por registro
        PrecioPromedio. Precio unitario promedio por registro de ese cliente
    """
    metricas_cliente = df.groupby(["IDCliente", "Pais"]).agg(
        CantidadArticComprados = ("Cantidad", "sum"),
        CantidadApariciones = ("Cantidad", "count"),
        CantidadPromedio = ("Cantidad", "mean"),
        PrecioPromedio = ("PrecioUnitario", "mean")
    )

    print(metricas_cliente.head(10))

    
    print("\nDiagrama de relaciones")

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.axis("off")

    #Entidades
    ax.text(
        0.15, 0.70,
        "CLIENTE\n\nIDCliente\nPais",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=1", facecolor="lightblue")
    )

    ax.text(
        0.50, 0.70,
        "FACTURA\n\nIDFactura\nFechaCompra",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=1", facecolor="lightgreen")
    )

    ax.text(
        0.85, 0.70,
        "ARTICULO\n\nIDArticulo\nDescripcion\nPrecioUnitario",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow")
    )


    #Relaciones
    ax.annotate(
        "realiza",
        xy=(0.40, 0.70),
        xytext=(0.25, 0.70),
        arrowprops=dict(arrowstyle="->", lw=2),
        ha="center",
        va="bottom"
    )

    ax.annotate(
        "contiene",
        xy=(0.75, 0.70),
        xytext=(0.60, 0.70),
        arrowprops=dict(arrowstyle="->", lw=2),
        ha="center",
        va="bottom"
    )

    ax.set_title(
        "Diagrama de Entidades y Relaciones - Online Retail",
        fontsize=16
    )

    plt.savefig("diagrama_entidades_relaciones.png")
    plt.close()

    print("Diagrama guardado como: diagrama_entidades_relaciones.png")


#Programa principal
df = pd.read_csv("online_retail_limpio.csv")

if not df.empty:

    print("Dataset cargado correctamente.\n")

    print("Numero de filas:", df.shape[0])
    print("Numero de columnas:", df.shape[1])

    print("\nColumnas:")
    print(df.columns.tolist())

    print("\nTipos de datos antes de modificar:")
    print(df.dtypes)

    df["IDFactura"] = df["IDFactura"].astype(object)
    df["IDArticulo"] = df["IDArticulo"].astype(object)
    df["Descripcion"] = df["Descripcion"].astype(object)
    #por alguna razon lo leia como flotante, asi que lo converti a int y luego a objeto
    df["IDCliente"] = df["IDCliente"].astype(int)
    df["IDCliente"] = df["IDCliente"].astype(object)
    df["Pais"] = df["Pais"].astype(object)
    #Al recuperarlos se leia la fecha en str y al volverla fecha se perdian los datos, con el mixed sea arregla
    df["FechaCompra"] = pd.to_datetime(df["FechaCompra"], format = "mixed", errors = "coerce")

    print("\nTipos de datos despues de modificar:")
    print(df.dtypes)


    #La practica en si, lo anterior fue solo recuperar los datos de la limpieza
    estadistica_descrip(df)

    #Identificarlas fue muy sencillo y se hizo a mano
    print("\nEntidades identificadas:")

    print("\n1. Cliente")
    print("\tIDCliente")
    print("\tPais")

    print("\n2. Factura")
    print("\tIDFactura")
    print("\tFechaCompra")

    print("\n3. Articulo")
    print("\tIDArticulo")
    print("\tDescripcion")
    print("\tPrecioUnitario")

    print("\nRelaciones identificadas:")
    
    print("\nCliente -> Factura")
    print("Un cliente puede realizar varias facturas.")
    print("\nFactura -> Articulo")
    print("Una factura puede contener varios articulos.")
    print("\nArticulo -> Factura")
    print("Un articulo puede aparecer en varias facturas.")

    print("\nLa relacion entre Factura y Articulo es de muchos a muchos.\nLa relacion entre Factura y Cliente es de uno a muchos.")

    
    diagr_metric_dagrupados(df)


else:

    print("Error: No se encontro el dataset limpiado llamado: online_retail_limpio.csv")

"""
Ejemplo de lo que me salio al ejecutarlo, asun asi se espera que funcione perfectamente para el profe.

Dataset cargado correctamente.

Numero de filas: 390885
Numero de columnas: 8

Columnas:
['IDFactura', 'IDArticulo', 'Descripcion', 'Cantidad', 'FechaCompra', 'PrecioUnitario', 'IDCliente', 'Pais']

Tipos de datos antes de modificar:
IDFactura           int64
IDArticulo            str
Descripcion           str
Cantidad            int64
FechaCompra           str
PrecioUnitario    float64
IDCliente         float64
Pais                  str
dtype: object

Tipos de datos despues de modificar:
IDFactura                 object
IDArticulo                object
Descripcion               object
Cantidad                   int64
FechaCompra       datetime64[us]
PrecioUnitario           float64
IDCliente                 object
Pais                      object
dtype: object

Descripcion generica de los datos:
            Cantidad                 FechaCompra  PrecioUnitario
count  390885.000000                      390885   390885.000000
mean       13.183893  2011-07-10 19:45:45.907671        2.873717
min         1.000000         2010-12-01 08:26:00        0.000000
25%         2.000000         2011-04-07 10:43:00        1.250000
50%         6.000000         2011-07-31 12:52:00        1.950000
75%        12.000000         2011-10-20 13:06:00        3.750000
max     80995.000000         2011-12-09 12:50:00      649.500000
std       181.976514                         NaN        4.285199

Estadisticas extra de Cantidad:
Moda: [1]
Sumatoria: 5153386
Varianza: 33115.45177236746
Asimetria: 400.1312271519315
Kurtosis: 171115.94416986825

Estadisticas extra de PrecioUnitario:
Moda: [1.25]
Sumatoria: 1123293.0
Varianza: 18.362926274363254
Asimetria: 35.70119461103926
Kurtosis: 3464.109373651346

Entidades identificadas:

1. Cliente
	IDCliente
	Pais

2. Factura
	IDFactura
	FechaCompra

3. Articulo
	IDArticulo
	Descripcion
	PrecioUnitario

Relaciones identificadas:

Cliente -> Factura
Un cliente puede realizar varias facturas.

Factura -> Articulo
Una factura puede contener varios articulos.

Articulo -> Factura
Un articulo puede aparecer en varias facturas.

La relacion entre Factura y Articulo es de muchos a muchos.
La relacion entre Factura y Cliente es de uno a muchos.

Metricas agrupadas por articulo:

Metricas por articulo:
                                         CantidadTotal  ...  CantidadKurtosis
IDArticulo Descripcion                                  ...                  
10002      INFLATABLE POLITICAL GLOBE              823  ...          9.953991
10080      GROOVY CACTUS INFLATABLE                291  ...          2.732062
10120      DOGGY RUBBER                            192  ...          4.442586
10123C     HEARTS WRAPPING TAPE                      5  ...               NaN
10124A     SPOTS ON RED BOOKCOVER TAPE              16  ...          2.664360
10124G     ARMY CAMO BOOKCOVER TAPE                 17  ...          4.000000
10125      MINI FUNKY DESIGN TAPES                1225  ...         11.496276
10133      COLOURING PENCILS BROWN TUBE           2373  ...         13.679328
10135      COLOURING PENCILS BROWN TUBE           1936  ...         25.489754
11001      ASSTD DESIGN RACING CAR PEN            1252  ...         55.946125

[10 rows x 13 columns]

Metricas agrupadas por cliente:
                          CantidadArticComprados  ...  PrecioPromedio
IDCliente Pais                                    ...                
12346     United Kingdom                   74215  ...        1.040000
12347     Iceland                           2458  ...        2.644011
12348     Finland                           2332  ...        0.692963
12349     Italy                              630  ...        4.237500
12350     Norway                             196  ...        1.581250
12352     Norway                             526  ...        4.075455
12353     Bahrain                             20  ...        6.075000
12354     Spain                              530  ...        4.503793
12355     Bahrain                            240  ...        4.203846
12356     Portugal                          1573  ...        2.946034

[10 rows x 4 columns]

Diagrama de relaciones
Diagrama guardado como: diagrama_entidades_relaciones.png
"""
