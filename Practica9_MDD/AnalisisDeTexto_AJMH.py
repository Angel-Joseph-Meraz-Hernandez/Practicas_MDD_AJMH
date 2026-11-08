#Practica 9. Análisis de Texto
#Dataset: OnlineRetailLimpio.csv para su procesamiento
#Nombre: Angel Joseph Meraz Hernandez
#Matricula: 2067151
"""
Generar una nube de palabras y realizar un análisis exploratorio a partir de
datos textuales.
"""

import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
#Primero tuve que descargar e instalar py -m pip install pandas matplotlib wordcloud
#lo note cuando me salto un error.

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
    #print(df.dtypes)

    #a esta parte del codigo casi no la toque mas que elo siguiente, para determinar cuantos datos me dirven o si hay errores
    descripciones = df["Descripcion"].dropna()
    print("\nCantidad de descripciones validas:", len(descripciones))
    print("Cantidad de descripciones diferentes:", descripciones.nunique())
    
    return descripciones
    



#Funcion para realizar el analisis de texto
def analisis_texto(descripciones):

    #Conte las apariciones de cada descripcion completa
    frecuencias_productos = Counter(descripciones)

    print("\nResultados del analisis de productos")
    print("Cantidad de productos diferentes:", len(frecuencias_productos))

    #Mostre los 20 productos que aparecieron mas veces
    print("\nLos 20 productos con mas registros:")

    for producto, cantidad in frecuencias_productos.most_common(20):
        print(producto + ":", cantidad)

    #Uni todas las descripciones en un solo texto
    texto = " ".join(descripciones)

    #Genere la nube de palabras a partir de las descripciones
    nube = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        min_font_size=5
    ).generate(texto)

    #Mostre y guarde la nube de palabras
    plt.figure(figsize=(10, 6))
    plt.imshow(nube, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig("NubePalabras_Practica9_AJMH.png")
    plt.show()
    plt.close()




#Programa principal
df = pd.read_csv("OnlineRetaiLimpio.csv")
if not df.empty:
    print("Dataset cargado correctamente.\n")
    descrip = preparacion(df)
    
    #Ejecute la funcion que toma las descripciones diferentes encontradas en el dataset
    #Considerendo su repititividad en los datos.
    if not descrip.empty:
        analisis_texto(descrip)
        print("\nAl realizar el análisis textual de la columna Descripcion, "
              "encontré 390,885 descripciones válidas, de las cuales 3,641 "
              "eran diferentes.\nTambien gracias a la lista de los 20 productos "
              "con mayor frecuencia, observé coincidencias entre los términos de "
              "la nube y los nombres de los productos más frecuentes. Por ejemplo,"
              "identifique las palabras, Cake cases, lunch bag, red retrospot, light "
              "holder, jumbo bag, T light, hot water, metal sigh; por lo que puedo "
              "deducir de esto es que los articulos con mayor frecuencia de apariciones"
              " son bolsas, artículos decorativos y productos relacionados con la cocina y la repostería.")
    else:
        print("No se encontraron descripciones para analizar.")
        
    

else:
    print("El archivo no contiene datos.")


    
#Ejemplo de la impresion
"""
Dataset cargado correctamente.

Numero de filas: 390885
Numero de columnas: 8

Cantidad de descripciones validas: 390885
Cantidad de descripciones diferentes: 3641

Resultados del analisis de productos
Cantidad de productos diferentes: 3641

Los 20 productos con mas registros:
WHITE HANGING HEART T-LIGHT HOLDER: 2023
REGENCY CAKESTAND 3 TIER: 1712
JUMBO BAG RED RETROSPOT: 1615
ASSORTED COLOUR BIRD ORNAMENT: 1392
PARTY BUNTING: 1390
LUNCH BAG RED RETROSPOT: 1304
SET OF 3 CAKE TINS PANTRY DESIGN : 1151
JUMBO BAG DOILEY PATTERNS: 1091
LUNCH BAG  BLACK SKULL.: 1078
LUNCH BAG SUKI  DESIGN : 1063
SMALL POPCORN HOLDER: 1060
PACK OF 72 RETROSPOT CAKE CASES: 1049
SPOTTY BUNTING: 1021
LUNCH BAG DOILEY PATTERN : 1020
PAPER CHAIN KIT 50'S CHRISTMAS : 1003
LUNCH BAG SPACEBOY DESIGN : 993
HEART OF WICKER SMALL: 979
LUNCH BAG CARS BLUE: 977
NATURAL SLATE HEART CHALKBOARD : 971
LUNCH BAG PINK POLKADOT: 941

Al realizar el análisis textual de la columna Descripcion, encontré 390,885 descripciones válidas, de las cuales 3,641 eran diferentes.
Tambien gracias a la lista de los 20 productos con mayor frecuencia, observé coincidencias entre los términos de la nube y los nombres de los productos más frecuentes. Por ejemplo,identifique las palabras, Cake cases, lunch bag, red retrospot, light holder, jumbo bag, T light, hot water, metal sigh; por lo que puedo deducir de esto es que los articulos con mayor frecuencia de apariciones son bolsas, artículos decorativos y productos relacionados con la cocina y la repostería.
"""
