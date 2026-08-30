#Practica 1. Limpieza de datos
#Dataset: Online Retail Convertido en CSV para su procesamiento
#Nombre: Angel Joseph Meraz Hernandez
#Matricula: 2067151

import pandas as pd
import numpy as np

df = pd.read_excel("Online Retail.xlsx")
print("Dataset cargado correctamente.\n")

cant_regis_inicio = df.shape[0]
print("Numero de filas:", df.shape[0])
print("Numero de columnas:", df.shape[1])

print("\nColumnas:")
print(df.columns.tolist())

print("\nPrimeras 5 filas (para ver que se cargo perfectamente):")
print(df.head())

print("\nTipos de datos:")
print(df.dtypes)

#Le cambie los nombres a las columnas
df = df.rename(columns={
    'InvoiceNo': 'IDFactura',
    'StockCode': 'IDArticulo',
    'Description': 'Descripcion',
    'Quantity': 'Cantidad',
    'InvoiceDate': 'FechaCompra',
    'UnitPrice': 'PrecioUnitario',
    'CustomerID': 'IDCliente',
    'Country': 'Pais'
})
print("\nFechas no validas con el nuevo formato:", df["FechaCompra"].isna().sum())

print("\nNuevas Columnas:")
print(df.columns.tolist())

#Se eliminaran los registros que su columna Cantidad <= 0, porque a mi solo me interesan las compras
print("\nLa cantidad de registro que se eliminaran porque su Cantidad <= 0 es:", (df["Cantidad"] <= 0).sum())
df = df[df["Cantidad"] > 0]

#Se eliminaran los registros que su columna PrecioUnitario < 0, porque puede que un producto cueste 0 por un descuento por ejemplo
print("La cantidad de registro que se eliminaran porque su PrecioUnitario < 0 es:", (df["PrecioUnitario"] < 0).sum())
df = df[df["PrecioUnitario"] >= 0]

#Se eliminaran los registros cuyo IDCliente esta vacio, dado que a mi solo me interesan las compras de articulos y estas compras tuvieron que hacerse por alguien
print("La cantidad de registros eliminados porque IDClientes esta vacio es:", df["IDCliente"].isna().sum())
df = df.dropna(subset=["IDCliente"])

#Se eliminaron registros correspondientes a cargos y servicios administrativos (Carriage, Dotcom Postage, Manual, PADS, Postage y Bank Charges), ya que no representan productos.
codigos_no_producto = [
    "C2",
    "DOT",
    "M",
    "PADS",
    "POST",
    "BANK CHARGES"
]
print("La cantidad de registros eliminados porque no representan productos es:", df["IDArticulo"].isin(codigos_no_producto).sum())
df = df[~df["IDArticulo"].isin(codigos_no_producto)]

#Me asegure que los valores que se pueden quedar como objetos lo sean.
#Me asegure que los ID de articulo, factura y cliente, sean cATEGORIA
#Me asegure que la fecha sea una fecha
#Deje igual los valores de cantidad (int) y precio unitario (float)
df["IDFactura"] = df["IDFactura"].astype(object)
df["IDArticulo"] = df["IDArticulo"].astype(object)
df["Descripcion"] = df["Descripcion"].astype(object)
df["FechaCompra"] = pd.to_datetime(df["FechaCompra"], errors="coerce")
df["IDCliente"] = df["IDCliente"].astype(object)
df["Pais"] = df["Pais"].astype(object)


print("\nNuevos tipos de datos:")
print(df.dtypes)

#Antes de eliminar paises, podriamos cambiarles el nombre a algunos
cambios = {
    'EIRE': 'Ireland',
    'RSA': 'South Africa'
    }
print("\nSe estandarizaron los nombres EIRE a Ireland y RSA a South Africa.")
df['Pais'] = df['Pais'].replace(cambios)

#Ahora si a eliminar elementos como los que solo hayan especificado que de que pais son
eliminar = ['European Community', 'Unspecified']
print("\nSe eliminaron los registros con Pais 'European Community' o 'Unspecified':", df["Pais"].isin(eliminar).sum())
df = df[~df["Pais"].isin(eliminar)]

#Ahora si a eliminar los repetidos, que todo el registro este repetido
print("\nCantidad de registros duplicados que se eliminaran:", df.duplicated().sum())
df = df.drop_duplicates()

cant_regis_fin = df.shape[0]
print("\nNuevo numero de filas:", df.shape[0])
print("Nuevo numero de columnas:", df.shape[1])
print("Se eliminaron esta cantidad de registros en la limpieza:", str(cant_regis_inicio - cant_regis_fin))

#Ahora lo guardamos como un CSV para proyectos futuros
df.to_csv("online_retail_limpio.csv", index=False)
print("\nDataset limpio guardado como: Online_Retail_Limpio.csv")
