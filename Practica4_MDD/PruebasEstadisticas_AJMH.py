"""
Pruebas Estadisticas (Entrega: Semana 7)
Comprobar diferencias en datos etiquetados mediante ANOVA + Prueba t o
la prueba de Kruskal-Wallis.
"""
#Practica 4. Pruebas Estadisticas
#Dataset: OnlineRetailLimpiado.csv para su procesamiento
#Nombre: Angel Joseph Meraz Hernandez
#Matricula: 2067151

import pandas as pd
from scipy import stats
    
    
def preparacion(df):
    print("Dataset cargado correctamente.\n")

    print("Numero de filas:", df.shape[0])
    print("Numero de columnas:", df.shape[1])
    """print("\nColumnas:")
    print(df.columns.tolist())
    
    print("\nTipos de datos antes de modificar:")
    print(df.dtypes)
    """
    df["IDFactura"] = df["IDFactura"].astype(object)
    df["IDArticulo"] = df["IDArticulo"].astype(object)
    df["Descripcion"] = df["Descripcion"].astype(object)
    #por alguna razon lo leia como flotante, asi que lo converti a int y luego a objeto
    df["IDCliente"] = df["IDCliente"].astype(int)
    df["IDCliente"] = df["IDCliente"].astype(object)
    df["Pais"] = df["Pais"].astype(object)
    #Al recuperarlos se leia la fecha en str y al volverla fecha se perdian los datos, con el mixed sea arregla
    df["FechaCompra"] = pd.to_datetime(df["FechaCompra"], format = "mixed", errors = "coerce")
    """
    #Esto servia para ver si no fallo la conversion a objeto y fecha
    print("\nIDFactura no validas con el nuevo formato:", df["IDFactura"].isna().sum())
    print("IDArticulo no validas con el nuevo formato:", df["IDArticulo"].isna().sum())
    print("Descripción no validas con el nuevo formato:", df["Descripcion"].isna().sum())
    print("IDCliente no validas con el nuevo formato:", df["IDCliente"].isna().sum())
    print("Pais no validas con el nuevo formato:", df["Pais"].isna().sum())
    print("Fechas no validas con el nuevo formato:", df["FechaCompra"].isna().sum())
    """
    return df

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


#Considerando que haria varias pruebas t para cada grupo de precios bajos, medios y altos
#Preferi hacer una funcion general para solo llamarla una y otra ves
def prueba_t(grupo1, grupo2):
    resultado = stats.ttest_ind(grupo1, grupo2, equal_var=False)
    return resultado.statistic, resultado.pvalue


def anova(productos):
    #Esto es para indicar cual es la hipotesis inicial, si hay diferencias o no entre la cantidad que se compra de cada una de las 3 clasificaciones
    print("\nPrueba ANOVA:")
    print("Hipotesis nula (H0): La cantidad promedio vendida es igual entre los tres grupos de precio.")
    print("Hipotesis alternativa (H1): Al menos uno de los grupos de precio tiene una cantidad promedio vendida diferente.")

    alpha = 0.05
    print("\nNivel de significacncia (alpha):", alpha)
    print("Nivel de confianza:", (1-alpha)*100, "%")
    
    #Inicia el proceso creando los grupos dependiendo de si tienen un precio promedio alto o bajo
    productos["GrupoPrecio"] = pd.qcut(productos["PrecioPromedio"],q=3,labels=["Bajo", "Medio", "Alto"])

    #Aqui muestra los datos de los grupos creados como la cantidad de productos en cada grupo,
    #el rango de precios que son tomados como bajo, alto, y medio
    print("\nCantidad de productos por grupo:", productos["GrupoPrecio"].value_counts())
    print("\nRangos de precios:")
    for grupo in ["Bajo", "Medio", "Alto"]:
        datos = productos[productos["GrupoPrecio"] == grupo]
        print(grupo, ": $" + str(round(datos["PrecioPromedio"].min(), 3)), "hasta $" + str(round(datos["PrecioPromedio"].max(), 3)))
    print("\nCantidad promedio vendida por grupo:", productos.groupby("GrupoPrecio", observed=True)["CantidadTotal"].mean().round().astype(int))

    #Separamos la cantidad total vendida de cada grupo
    grupo_bajo = productos[productos["GrupoPrecio"] == "Bajo"]["CantidadTotal"]
    grupo_medio = productos[productos["GrupoPrecio"] == "Medio"]["CantidadTotal"]
    grupo_alto = productos[productos["GrupoPrecio"] == "Alto"]["CantidadTotal"]

    # Realizamos la prueba ANOVA
    resultado = stats.f_oneway(grupo_bajo, grupo_medio, grupo_alto)

    print("\nDespues del calculo ANOVA")
    print("Estadistico F:", round(resultado.statistic, 3))
    #Aqui hice algo diferente ya que mostraba 0.0 si lo redondeaba,
    #asi que mejor lo deje en notacion cientifica
    print("Valor p:", format(resultado.pvalue, ".3e"))

    #Interpretamos el resultado, si el p valor es menor a alpha se rechaza la hipotesis nula, es decir que si hay alguna diferencia significativa en cuanto se compra cada grupo de precios
    #de lo contrario se conserva la hipotesis nula y no hace falta la prueba t, pues toodos son casi iguales
    if resultado.pvalue < 0.05:
        print("Se rechaza H0.")
        print("Existen diferencias significativas entre los grupos de precio.")

        print("\nPruebas t:")
        t_bajo_medio, p_bajo_medio = prueba_t(grupo_bajo, grupo_medio)
        print("\nComparacion de cantidad promedio vendida entre productos de precio bajo vs productos de precio medio")
        print("Estadistico t:", round(t_bajo_medio, 3))
        print("Valor p:", format(p_bajo_medio, ".3e"))
        if p_bajo_medio < alpha:
            print("Existe una diferencia significativa en la cantidad promedio vendida entre los productos de precio bajo y medio.")
        else:
            print("No existe una diferencia significativa en la cantidad promedio vendida entre los productos de precio bajo y medio.")

        t_bajo_alto, p_bajo_alto = prueba_t(grupo_bajo, grupo_alto)
        print("\nComparacion de cantidad promedio vendida entre productos de precio bajo vs productos de precio alto")
        print("Estadistico t:", round(t_bajo_alto, 3))
        print("Valor p:", format(p_bajo_alto, ".3e"))
        if p_bajo_alto < alpha:
            print("Existe una diferencia significativa en la cantidad promedio vendida entre los productos de precio bajo y alto.")
        else:
            print("No existe una diferencia significativa en la cantidad promedio vendida entre los productos de precio bajo y alto.")

        t_medio_alto, p_medio_alto = prueba_t(grupo_medio, grupo_alto)
        print("\nComparacion de cantidad promedio vendida entre productos de precio medio vs productos de precio alto")
        print("Estadistico t:", round(t_medio_alto, 3))
        print("Valor p:", format(p_medio_alto, ".3e"))
        if p_medio_alto < alpha:
            print("Existe una diferencia significativa en la cantidad promedio vendida entre los productos de precio medio y alto.")
        else:
            print("No existe una diferencia significativa en la cantidad promedio vendida entre los productos de precio medio y alto.")

        
        #Se concluye que si hay una diferencia notable entre la cantidad promedio que se compra de productos de precio alto,
        #en comparacion a precio bajo o medio.
        #Pero con la prueba t no se puede saber si se compra mas o menos.
        print("\nSe concluye que si hay una diferencia notable entre la cantidad promedio que se compra de productos de precio alto, en comparacion a precio bajo o medio.")
        
        #Esto se concluyo al mirar los resultados al ejecutarlo,
        #al igual que se ve que en promedio de compraron menos los productos de alto precio a compraracion de los otros
        print("Por los promedios de las cantidades se puede saber que en promedio de compraron menos los productos de alto precio a compraracion de los otros.")

    else:
        print("No se rechaza H0.")
        print("No hay evidencia suficiente de diferencias significativas entre los grupos, no hace falta la prueba t.")

    
#Programa principal
df = pd.read_csv("OnlineRetaiLimpio.csv")
if not df.empty:
    #Para esta practica elegi usar la prueba ANOVA + las pruebas t.
    df = preparacion(df)
    
    """
    print("\nTipos de datos despues de modificar:")
    print(df.dtypes)
    """
    #La idea es analizar con la prueba ANOVA y la prueba t si:
    #¿Hay alguna diferencia entre la cantidad total vendida de productos con precio bajo, medio y alto?

    #Limpiamos los precios atipicos, ya que podrian afectar los promedios,
    #pero se eligieron no eliminar desde la limpieza aunque si se planteo,
    #porque podria ser util para calculos en alguna actividad futura.
    df_limpio = limpiar_atipicos(df)

    #Cantidad total vendida por producto
    productos = df.groupby("IDArticulo").agg(CantidadTotal=("Cantidad", "sum")).reset_index()

    #Precio promedio por producto, sin precios atipicos
    precios = df_limpio.groupby("IDArticulo").agg(PrecioPromedio=("PrecioUnitario", "mean")).reset_index()

    #Unimos ambas tablas
    productos = productos.merge(precios, on="IDArticulo")

    """#esto fue mas que nada para ver el promedio que muestran y corroborar con calculos personales
    print("\nDatos agrupados por producto:")
    print(productos[productos["IDArticulo"] == "22502"])
    print(productos.head())"""
    
    anova(productos)
    #Todo se hizo con una funcion para que fuera mas limpio, algo asi imprimio:



"""
Dataset cargado correctamente.

Numero de filas: 390885
Numero de columnas: 8

Analizando precios atipicos de cada productos.
Registros originales: 390885
Registros despues de quitar atipicos: 361438
Registros considerados atipicos: 29447

Prueba ANOVA:
Hipotesis nula (H0): La cantidad promedio vendida es igual entre los tres grupos de precio.
Hipotesis alternativa (H1): Al menos uno de los grupos de precio tiene una cantidad promedio vendida diferente.

Nivel de significacncia (alpha): 0.05
Nivel de confianza: 95.0 %

Cantidad de productos por grupo: GrupoPrecio
Bajo     1404
Alto     1134
Medio    1103
Name: count, dtype: int64

Rangos de precios:
Bajo : $0.04 hasta $1.25
Medio : $1.258 hasta $2.95
Alto : $2.95 hasta $158.077

Cantidad promedio vendida por grupo: GrupoPrecio
Bajo     1888
Medio    1659
Alto      594
Name: CantidadTotal, dtype: int64

Despues del calculo ANOVA
Estadistico F: 47.331
Valor p: 5.099e-21
Se rechaza H0.
Existen diferencias significativas entre los grupos de precio.

Pruebas t:

Comparacion de cantidad promedio vendida entre productos de precio bajo vs productos de precio medio
Estadistico t: 1.405
Valor p: 1.600e-01
No existe una diferencia significativa en la cantidad promedio vendida entre los productos de precio bajo y medio.

Comparacion de cantidad promedio vendida entre productos de precio bajo vs productos de precio alto
Estadistico t: 10.972
Valor p: 3.240e-27
Existe una diferencia significativa en la cantidad promedio vendida entre los productos de precio bajo y alto.

Comparacion de cantidad promedio vendida entre productos de precio medio vs productos de precio alto
Estadistico t: 8.096
Valor p: 1.195e-15
Existe una diferencia significativa en la cantidad promedio vendida entre los productos de precio medio y alto.

Se concluye que si hay una diferencia notable entre la cantidad promedio que se compra de productos de precio alto, en comparacion a precio bajo o medio.
Por los promedios de las cantidades se puede saber que en promedio de compraron menos los productos de alto precio a compraracion de los otros.
"""
