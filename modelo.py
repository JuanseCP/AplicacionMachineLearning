import pyodbc
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

# Conexión a la base de datos SQL Server
def connect_to_sql_server():
    connection_string = (
        "DRIVER={SQL Server};"
        "SERVER=YOGURTDEMON\\SQLEXPRESS;"  # Cambia por tu servidor
        "DATABASE=vigendb;"  # Cambia por tu base de datos
    )
    conn = pyodbc.connect(connection_string)
    return conn

# Obtener datos desde SQL Server
def fetch_data_from_db(conn):
    query = "SELECT * FROM poll;"  # Cambia por tu tabla
    df = pd.read_sql(query, conn)
    return df

# Preprocesar los datos
def preprocess_data(df):
    # Eliminar columnas no numéricas si es necesario
    df = df.select_dtypes(include=[int, float])
    df = df.dropna()  # Opcional: eliminar filas con valores nulos
    return df

# Aplicar DBSCAN
def apply_dbscan(df, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df)
    return labels

# Aplicar Isolation Forest
def apply_isolation_forest(df, contamination=0.1):
    isolation_forest = IsolationForest(contamination=contamination)
    anomalies = isolation_forest.fit_predict(df)
    return anomalies

# Proceso principal
def main():
    # Conectar a la base de datos
    conn = connect_to_sql_server()

    # Obtener y preprocesar los datos
    df = fetch_data_from_db(conn)
    df_preprocessed = preprocess_data(df)

    # Aplicar DBSCAN para detectar clusters de riesgo
    dbscan_labels = apply_dbscan(df_preprocessed)

    # Aplicar Isolation Forest para detectar anomalías
    isolation_forest_anomalies = apply_isolation_forest(df_preprocessed)

    # Mostrar los resultados
    df['Cluster_Label'] = dbscan_labels
    df['Anomalies'] = isolation_forest_anomalies
    print(df)

    # Cerrar conexión
    conn.close()

if __name__ == "__main__":
    main()
