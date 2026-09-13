"""
Visualización de Datos (Entrega: Semana 6)
Generar al menos 5 tipos de gráficas distintas (diagramas de pastel, histogramas,
diagramas de caja, dispersión, etc.) usando ciclos o automatización en código.
"""
#Practica 3. Visualizacion de Datos
#Dataset: Online Retail Limpiado para su procesamiento
#Nombre: Angel Joseph Meraz Hernandez
#Matricula: 2067151
#Nota al realizarlo me di cuenta que habia que terminar un poco mas de la limpieza

import pandas as pd
import matplotlib.pyplot as plt

#Una funcion para sacar el procentaje de palabras que se igualan en ambas Descriciones de un mismo IDArticulo
def porcentaje_palabras(descripcion1, descripcion2):
    palabras1 = set(descripcion1.upper().replace(",", "").split())

    palabras2 = set(descripcion2.upper().replace(",", "").split())

    #Evitar división entre cero si ambas descripciones estan vacias
    if not palabras1 or not palabras2:
        return 0

    comunes = palabras1.intersection(palabras2)
    #Se toma como referencia la descripcion que tenga menos palabras
    porcentaje = len(comunes) / min(len(palabras1),len(palabras2))

    return porcentaje

#Funcion de lo ultimo que falto limpiar ya que en su momento lo haba visto bien
def ultima_limpieza(df):
    df_prueba = df.copy()
    
    #Habian textos que tenian "DESCRIPCION , ARTICULO", esto es para arreglar la coma
    df_prueba["Descripcion"] = (df_prueba["Descripcion"].str.replace(" ,", ",", regex=False))


    varios = (df_prueba.groupby("IDArticulo")["Descripcion"].unique())

    #Fui en prueba y error bajando el umbral hasta 0.2, es decir que minimo el 0.2 de caracteres deben igualars en ambas descripciones para tomar la primera
    umbral = 0.2
    corregidos = 0

    for articulo, descripciones in varios.items():
        if len(descripciones) > 1:
            primera = descripciones[0]

            similares = True

            for descripcion in descripciones[1:]:
                porcentaje = porcentaje_palabras(
                    primera,
                    descripcion
                )

                if porcentaje < umbral:
                    similares = False
                    break

            if similares:
                df_prueba.loc[df_prueba["IDArticulo"] == articulo,"Descripcion"] = primera

                corregidos += 1

    #Esto era para confirmar que este listo
    restantes = (df_prueba.groupby("IDArticulo")["Descripcion"].nunique())

    restantes = restantes[restantes > 1]
    """
    print("Casos corregidos:", corregidos)
    print("Casos que todavía tienen varias descripciones:",
          len(restantes))"""

    #IDArticulo que todavia tienen mas de una descripcion
    restantes = (df_prueba.groupby("IDArticulo")["Descripcion"].unique())

    restantes = restantes[restantes.apply(len) > 1]

    
    
    #Ahora lo contrario, ver cuantas descripciones tiene viarios IDArticulos
    descripciones_varios = (df_prueba.groupby("Descripcion")["IDArticulo"].nunique())

    descripciones_varios = descripciones_varios[descripciones_varios > 1]
    
    
    #Ahora empieza el verdadero cambio de las IDArticulo
    descripciones_varios = (df_prueba.groupby("Descripcion")["IDArticulo"].nunique())

    descripciones_varios = descripciones_varios[descripciones_varios > 1]

    cambios_id = 0

    for descripcion in descripciones_varios.index:
        datos = df_prueba[df_prueba["Descripcion"] == descripcion]

        conteo = datos["IDArticulo"].value_counts()

        #Tomar el IDArticulo con mayor cantidad de registros
        #Si hay empate, value_counts() conserva el primero
        id_principal = conteo.index[0]

        # Cambiar los demás IDArticulo por el principal
        indices_cambio = ((df_prueba["Descripcion"] == descripcion) &
            (df_prueba["IDArticulo"] != id_principal))

        cambios_id += indices_cambio.sum()

        df_prueba.loc[indices_cambio,"IDArticulo"] = id_principal

    

    #Se habia copiado la impresion de las descripciones para ver si funciono
    #Toca arreglar el pais por IDCliente
    pais_cliente = (df_prueba.groupby("IDCliente")["Pais"].agg(lambda x: x.value_counts().index[0]))

    pais_correcto = df_prueba["IDCliente"].map(pais_cliente)

    cambios = (df_prueba["Pais"] != pais_correcto).sum()

    

    #Corregir solamente los países que no coinciden
    df_prueba.loc[df_prueba["Pais"] != pais_correcto,"Pais"] = pais_correcto
    
    return df_prueba
    
    
def preparacion(df):
    print("Dataset cargado correctamente.\n")

    print("Numero de filas:", df.shape[0])
    print("Numero de columnas:", df.shape[1])
    print("\nColumnas:")
    print(df.columns.tolist())
    """
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

def grafica1(df):
    #Primero se obtuvieron las sumas de la cantidad de articulos comprados agrupadas por pais
    ventas_articulo = df.groupby("IDArticulo")["Cantidad"].sum()
    top_10 = ventas_articulo.sort_values(ascending=False).head(10)
    
    #Ahora se colocan en la grafica de barras horizontales que seran 10
    plt.barh(top_10.index, top_10.values)

    plt.xlabel("Cantidad total vendida")
    plt.ylabel("Articulo")
    plt.title("Top 10 articulos mas vendidos")
    
    #Debido a que la mayoria del top 10 no llega ni al millon, esto es para que se vean bien sus cantidades
    for i, valor in enumerate(top_10.values):
        plt.text(valor, i, f" {valor:,.0f}", va="center")
    #Para que quede bien acomodado, ya que el titulo del eje Y estaba escondido
    plt.tight_layout()

    #Aqui la anotacion de lo que se observo
    #En este caso se observo que:
    #   Del top 10 El articulo mas vendido es el 23843 con 80995  ventas
    #   Del top 10 el articulo menos vendido es el 22492 con 26076 ventsa
    plt.text(45000, 9, "Comentarios:")
    plt.text(45000, 8, "Del top 10 el articulo mas vendido")
    plt.text(45000, 7.5, "es el 23843 con 80995 ventas")
    plt.text(45000, 7, "Del top 10 el articulo menos vendido")
    plt.text(45000, 6.5, "es el 22492 con 26076 ventsa")
    #Esto es para cambiar el nombre de la ventana porque todas decian Figura1
    plt.get_current_fig_manager().window.title("Top 10 articulos mas vendidos")
    plt.show()

def grafica2(df):
    #Primero se obtuvieron las sumas de la cantidad de articulos comprados agrupadas por al dia
    ventas_mes = df.groupby(df["FechaCompra"].dt.to_period("M"))["Cantidad"].sum()
    ventas_mes.index = ventas_mes.index.astype(str)
    

    #Aqui se hace la grafica de lineas, dibujando los puntos y las linaeas de punto a punto
    plt.plot(ventas_mes.index, ventas_mes.values)
    plt.xlabel("Mes")
    plt.ylabel("Cantidad vendida por mes")
    plt.title("Cantidad de productos vendidos por mes")
    #Esto es para que los valores del eje x, las fechas por mes, se giren 45 grados para que no se empalmen
    plt.xticks(rotation=45)
    plt.tight_layout()

    #Se me ocurrio añadir esto para que muestre cada vez que subia para llevar constancia de cuanto fue maximo anterior
    #Fuera del for puse la cantidad vendida del ultimo mes
    maximo = -999
    for i, valor in enumerate(ventas_mes.values):
        if (valor > maximo):
            plt.text(i, valor, f" {valor:,.0f}",  ha="center", va="bottom")
            maximo = valor
    plt.text(i, valor, f" {valor:,.0f}",  ha="center", va="bottom")
    #Aqui la anotacion de lo que se observo
    #En este caso se observo que:
    #   En el ultimo mes, diciembre de 2011, las ventas se desplomaron;
    #   pero siguen siendo altas con respecto al mes anterior.
    plt.text(0, 600000, "Comentarios:")
    plt.text(0, 575000, "En el ultimo mes, diciembre de 2011, las ventas")
    plt.text(0, 550000, "se desplomaron; pero siguen siendo altas con")
    plt.text(0, 525000, "respecto al inicio anterior.")
    plt.get_current_fig_manager().window.title("Cantidad de productos vendidos por mes")
    plt.show()

def grafica3(df):
    #Primero agrupamos la cantidad total de ese articulo comprada a lo lardo de los 2 años, y el precio promedio de cada articulo
    articulos = (df.groupby("IDArticulo").agg(CantidadTotal=("Cantidad", "sum"), PrecioPromedio=("PrecioUnitario", "mean")).reset_index())

    # Gráfica de dispersión
    plt.scatter(articulos["PrecioPromedio"],articulos["CantidadTotal"])

    plt.xlabel("Precio promedio del artículo (En libras)")
    plt.ylabel("Cantidad total vendida")
    plt.title("Cantidad total vendida vs precio promedio por artículo")
    plt.get_current_fig_manager().window.title("Relacion entre el precio promedio de cada articulo y cuanto se vendio.")
    plt.tight_layout()
    
    #Aqui la anotacion de lo que se observo
    #En este caso se observo que:
    #   La mayoria de los productos se concentran enlos precios promedio mas
    #   bajos, por lo que sí, los productos mas comprados son los mas baratos
    plt.text(20, 80000, "Comentarios:")
    plt.text(20, 75000, "La mayoria de los productos se concentran en los precios ")
    plt.text(20, 70000, "promedio mas bajos, por lo que sí, los productos mas")
    plt.text(20, 65000, "comprados son los mas baratos")
    
    plt.show()

def grafica4(df):
    #Primero agrupamos el promedio del precio por articulo
    precios_articulos = (df.groupby("IDArticulo")["PrecioUnitario"].mean())
    #Ahora si realizamos el histograma 
    plt.hist(precios_articulos, bins=80)
    
    plt.xlabel("Precio promedio del artículo (En libras)")
    plt.ylabel("Cantidad de artículos")
    plt.title("Distribución del precio promedio de los artículos")
    plt.get_current_fig_manager().window.title("Precio promedio de los articulos.")
    plt.tight_layout()
    #Esto es para crear una marca para ver algo que me servira en la interpretacion
    plt.text(10, 100, "! marca de 10 libras")
    
    #Aqui la anotacion de lo que se observo
    #En este caso se observo que:
    #   La mayoria de los precios de los articulos estan entre 0 y 10 libras(moneda)
    plt.text(20, 400, "Comentarios:")
    plt.text(20, 300, "La mayoria de los precios de los articulos estan entre")
    plt.text(20, 225, "0 y 10 libras(moneda)")
    
    plt.show()

def grafica5(df):
    #Primero se calcula en ingraso producido en cada compra de la base de datos, multiplicando la cantidad por el precio unitario
    df["Ingreso"] = df["Cantidad"] * df["PrecioUnitario"]
    #Ahora se agrupan esos ingresos calculados por pais
    ingresos_pais = (df.groupby("Pais")["Ingreso"].sum().sort_values(ascending=False))

    top_9_pais = ingresos_pais.head(9)
    otros = ingresos_pais.iloc[9:].sum()

    datos_pastel = pd.concat([top_9_pais,pd.Series({"Otros": otros})])

    plt.pie(datos_pastel,labels=datos_pastel.index,autopct="%1.1f%%",startangle=90)
    plt.get_current_fig_manager().window.title("Distribución del ingreso total por país")
    plt.title("Distribución del ingreso total por país")
    #Aqui la anotacion de lo que se observo
    #En este caso se observo que:
    #   United Kingdom es el que mas ingreso genera
    plt.text(-1.2, 1, "Comentarios:")
    plt.text(-1.2, 0.9, "United Kingdom")
    plt.text(-1.2, 0.83, "es el que mas")
    plt.text(-1.2, 0.72, "ingreso genera")
    plt.show()

#Programa Principal
df = pd.read_csv("online_retail_limpio.csv")

if not df.empty:
    #Esta parte es porque note que me falto limpiar para que los datos tengan los mismo IDArticulo la misma Descripcion
    #Ademas de que habia descripciones que se repetian en diferentes IDArticulos
    #Y habia varioas IDCliente con diferentes Pais, que podria tener logica pero preferi que un cliente estuviera en un mismo pais
    df = preparacion(df)

    df_limpio = ultima_limpieza(df)

    df_limpio.to_csv("OnlineRetaiLimpio.csv",index=False)



df = pd.read_csv("OnlineRetaiLimpio.csv")

if not df.empty:
    df = preparacion(df)

    """
    print("\nTipos de datos despues de modificar:")
    print(df.dtypes)
    """
    #Se ejecuto 2 veces cada grafica para decir en comentarios que se observa
    #ya que con la misma tabla limpiada debe dar lo mismo sin importar quien la ejecute.

    
    #Grafica de barras horizontales para ver los 10 paises que relizaron mas compras.
    grafica1(df)

    
    #Grafica de linea para ver mes por mes, cuanto se compro para ver si fue en aumento en decenso
    grafica2(df)

    #Grafica de dispercion para saber si la cantidad total de compra de un articulo tiene relacion con su precio
    grafica3(df)
    
    #Histograma para saber en que precio ronda la mayoria de los productos, teniendo en cuenta el precio promedio por articulo
    grafica4(df)

    #Grafica de pastel para saber que paises es de los que se obtiene una mayor ganancia
    grafica5(df)
