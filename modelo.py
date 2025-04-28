import pyodbc
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import mlflow
import mlflow.sklearn
import os

# --- Configuración (sin cambios) ---
VIOLENCE_TYPE_COLUMN_NAME = 'tipo_caso_reglas_v2'
MLFLOW_EXPERIMENT_NAME = "Priorizacion_Violencia_Reglas_Detalladas_V2_SQL" # V2 en nombre exp
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
P7_MAP = {1: 'infantil', 2: 'intrafamiliar', 3: 'adulto mayor', 4: 'género', 0: 'ninguno_p7'}
FINAL_VIOLENCE_TYPES = ['infantil', 'adulto mayor', 'género', 'intrafamiliar', 'Ninguno_Detectado']


# --- Funciones connect_to_sql_server, fetch_data_from_db (sin cambios) ---
def connect_to_sql_server():
    connection_string = ("DRIVER={SQL Server};SERVER=YOGURTDEMON\\SQLEXPRESS;DATABASE=vigendb;")
    try: conn = pyodbc.connect(connection_string); print("Conexión SQL Server OK."); return conn
    except pyodbc.Error as ex: print(f"Error SQL Server: {ex.args[0]}\n{ex}"); return None

def fetch_data_from_db(conn):
     if conn is None: return None
     try:
         query = f"SELECT * FROM poll;"
         df = pd.read_sql(query, conn)
         print(f"Datos leídos: {df.shape[0]} filas, {df.shape[1]} columnas.")
         required_cols = ['edad'] + [f'p{i}' for i in range(1, 8)]
         missing_cols = [col for col in required_cols if col not in df.columns]
         if missing_cols: print(f"¡ERROR CRÍTICO! Faltan columnas para reglas: {missing_cols}"); return None
         return df
     except Exception as e: print(f"Error lectura DB: {e}"); return None


# --- Función assign_violence_type_rule_based_v2 (sin cambios) ---
def assign_violence_type_rule_based_v2(df, column_name):
    if df is None or df.empty: return df
    rule_cols = ['edad'] + [f'p{i}' for i in range(1, 8)]
    print("Verificando y convirtiendo columnas para reglas...")
    for col in rule_cols:
        if col in df.columns:
            original_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any(): print(f"Advertencia: NaNs en '{col}' (original: {original_dtype}).")
        else: print(f"Error: Columna '{col}' no encontrada."); return None
    print(f"Asignando tipo de violencia en '{column_name}'...")
    cond_infantil_edad = (df['edad'] < 18); cond_adulto_mayor_edad = (df['edad'] >= 60)
    p7_type = df['p7'].map(P7_MAP)
    cond_p7_genero = (p7_type == 'género') & (~cond_infantil_edad) & (~cond_adulto_mayor_edad)
    cond_p7_intrafamiliar = (p7_type == 'intrafamiliar') & (~cond_infantil_edad) & (~cond_adulto_mayor_edad)
    cond_p3_genero = (df['p3'] == 1) & (~cond_infantil_edad) & (~cond_adulto_mayor_edad) & (~cond_p7_genero) & (~cond_p7_intrafamiliar)
    any_other_aggression = (df['p1'] == 1) | (df['p2'] == 1) | (df['p4'] == 1) | (df['p5'] == 1) | (df['p6'] == 1)
    cond_other_intrafamiliar = any_other_aggression & (~cond_infantil_edad) & (~cond_adulto_mayor_edad) & (~cond_p7_genero) & (~cond_p7_intrafamiliar) & (~cond_p3_genero)
    conditions = [cond_infantil_edad, cond_adulto_mayor_edad, cond_p7_genero, cond_p7_intrafamiliar, cond_p3_genero, cond_other_intrafamiliar]
    choices = ['infantil', 'adulto mayor', 'género', 'intrafamiliar', 'género', 'intrafamiliar']
    default_choice = 'Ninguno_Detectado'
    df[column_name] = np.select(conditions, choices, default=default_choice)
    print(f"Columna '{column_name}' asignada por reglas v2."); print("Distribución de tipos (reglas v2):"); print(df[column_name].value_counts(normalize=True, dropna=False) * 100)
    return df


# --- Función preprocess_and_scale_data (sin cambios) ---
def preprocess_and_scale_data(df, cols_to_exclude_from_scaling=['edad', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']):
    if df is None or df.empty: return None, None, None
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.empty: print("No hay columnas numéricas."); return None, None, None
    cols_present = [col for col in cols_to_exclude_from_scaling if col in df_numeric.columns]
    if cols_present: print(f"Excluyendo del escalado: {cols_present}"); df_to_scale = df_numeric.drop(columns=cols_present)
    else: df_to_scale = df_numeric.copy()
    if df_to_scale.empty:
        print("No quedan columnas para escalar."); valid_indices_from_numeric = df_numeric.dropna().index
        return pd.DataFrame(index=valid_indices_from_numeric), None, valid_indices_from_numeric
    print(f"Columnas para escalar: {list(df_to_scale.columns)}")
    initial_rows = len(df_to_scale)
    df_processed = df_to_scale.dropna(); rows_after_dropna = len(df_processed); valid_indices = df_processed.index
    if initial_rows > rows_after_dropna: print(f"Se eliminaron {initial_rows - rows_after_dropna} filas con NaN en columnas a escalar.")
    if df_processed.empty: print("DataFrame vacío post-dropna."); return pd.DataFrame(index=valid_indices), None, valid_indices
    scaler = StandardScaler(); df_scaled_values = scaler.fit_transform(df_processed)
    df_scaled = pd.DataFrame(df_scaled_values, index=valid_indices, columns=df_processed.columns)
    print("Datos escalados OK."); return df_scaled, scaler, valid_indices

# --- Funciones apply_pca, apply_dbscan, apply_isolation_forest (sin cambios) ---
def apply_pca(df_scaled, n_components=2):
     if df_scaled is None or df_scaled.empty or df_scaled.shape[1] == 0: print("PCA no aplicable."); return None, None
     n_components = min(n_components, df_scaled.shape[1])
     if n_components <= 0: print("PCA n_components inválido."); return None, None
     pca = PCA(n_components=n_components); print(f"Aplicando PCA con n_components={n_components}")
     try: pcs = pca.fit_transform(df_scaled); df_pca = pd.DataFrame(pcs, index=df_scaled.index, columns=[f'PC{i+1}' for i in range(n_components)]); print(f"PCA Varianza: {pca.explained_variance_ratio_}"); return df_pca, pca
     except Exception as e: print(f"Error PCA: {e}"); return None, None

def apply_dbscan(df_scaled, eps=0.5, min_samples=5):
     if df_scaled is None or df_scaled.empty: print("DBSCAN no aplicable."); return None, None
     dbscan = DBSCAN(eps=eps, min_samples=min_samples); print(f"Aplicando DBSCAN eps={eps}, min_samples={min_samples}")
     try: labels = dbscan.fit_predict(df_scaled); print("DBSCAN completado."); return labels, dbscan
     except Exception as e: print(f"Error DBSCAN: {e}"); return None, None

def apply_isolation_forest(df_scaled, contamination='auto'):
     if df_scaled is None or df_scaled.empty: print("IF no aplicable."); return None, None
     iso = IsolationForest(contamination=contamination, random_state=42); print(f"Aplicando IF contamination={contamination}")
     try: anomalies = iso.fit_predict(df_scaled); print("IF completado."); return anomalies, iso
     except Exception as e: print(f"Error IF: {e}"); return None, None


# --- FUNCIÓN REVISADA: classify_priority (con nueva lógica) ---
def classify_priority(df_results, type_column_name, pca_threshold_std_dev=1.5):
    """
    Clasifica la prioridad ('Alta', 'Media', 'Baja') combinando el tipo de violencia
    con los resultados de IF, DBSCAN y PCA.
    """
    if df_results is None or df_results.empty: return None
    print(f"Clasificando prioridad v2 (umbral PCA std dev={pca_threshold_std_dev})...")

    # --- Verificar columnas necesarias ---
    required_cols = ['Anomaly_IF', 'Cluster_Label', type_column_name]
    pca_cols = [col for col in df_results.columns if col.startswith('PC')]
    missing_req = [col for col in required_cols if col not in df_results.columns]
    if missing_req:
        print(f"Error: Faltan columnas requeridas para prioridad: {missing_req}")
        return None
    use_pca = bool(pca_cols)
    if not use_pca: print("Advertencia: No se encontraron columnas PCA, no se usarán en prioridad.")

    # --- Calcular PCA extremeness (si aplica) ---
    is_extreme_pca = pd.Series(False, index=df_results.index)
    if use_pca:
        pca_std_devs = df_results[pca_cols].std()
        for i, col in enumerate(pca_cols):
            threshold = pca_threshold_std_dev * pca_std_devs[i]
            if pd.notna(threshold) and threshold > 0:
                 is_extreme_pca |= (df_results[col].abs() > threshold)
                 # print(f"Umbral PCA {col}: +/- {threshold:.4f}") # Descomentar para debug
            # else: print(f"Advertencia: Umbral PCA no válido para {col}")

    # --- Aplicar lógica de prioridad ---
    priorities = pd.Series(index=df_results.index, dtype=str)

    for index, row in df_results.iterrows():
        # Obtener valores de la fila (manejar posible NaN aunque no debería ocurrir)
        tipo_violencia = row[type_column_name] if pd.notna(row[type_column_name]) else 'TipoDesconocido'
        is_if_anomaly = row['Anomaly_IF'] == -1 if pd.notna(row['Anomaly_IF']) else False
        is_dbscan_outlier = row['Cluster_Label'] == -1 if pd.notna(row['Cluster_Label']) else False
        is_pca_extreme = is_extreme_pca.loc[index] if use_pca and index in is_extreme_pca.index else False

        # Regla 1: Infantil es siempre Alta
        if tipo_violencia == 'infantil':
            priorities.loc[index] = 'Alta'
            continue # Pasar a la siguiente fila

        # Regla 2: Ninguno_Detectado es siempre Baja
        if tipo_violencia == 'Ninguno_Detectado':
            priorities.loc[index] = 'Baja'
            continue # Pasar a la siguiente fila

        # Regla 3: Lógica para Género, Adulto Mayor, Intrafamiliar
        if tipo_violencia in ['género', 'adulto mayor', 'intrafamiliar']:
            # Sub-regla 3.1: Alta si es anomalía IF
            if is_if_anomaly:
                priorities.loc[index] = 'Alta'
            # Sub-regla 3.2: Media si es Inlier IF PERO outlier DBSCAN o extremo PCA
            elif is_dbscan_outlier or is_pca_extreme:
                 priorities.loc[index] = 'Media'
            # Sub-regla 3.3: Baja si es Inlier IF, NO outlier DBSCAN, y NO extremo PCA
            else:
                 priorities.loc[index] = 'Baja'
            continue # Pasar a la siguiente fila

        # Default para tipos no esperados (si los hubiera)
        priorities.loc[index] = 'PrioridadIndefinida_TipoDesconocido'


    print("Clasificación de prioridad v2 completada.")
    return priorities


# --- Proceso Principal (Modificado para llamar a la nueva classify_priority) ---
def main():
    with mlflow.start_run():
        run_uuid = mlflow.active_run().info.run_uuid
        print(f"MLflow Run started (Experiment: {MLFLOW_EXPERIMENT_NAME}, Run ID: {run_uuid})")

        # --- Parámetros (iguales) ---
        n_pca_components = 2; pca_extremeness_threshold = 1.5
        dbscan_eps = 0.5; dbscan_min_samples = 5
        iforest_contamination = 'auto'
        cols_excluded = ['edad', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']

        # Log Parámetros
        mlflow.log_param("n_pca_components", n_pca_components); mlflow.log_param("pca_extremeness_threshold", pca_extremeness_threshold)
        mlflow.log_param("dbscan_eps", dbscan_eps); mlflow.log_param("dbscan_min_samples", dbscan_min_samples)
        mlflow.log_param("iforest_contamination", iforest_contamination)
        mlflow.log_param("violence_type_column", VIOLENCE_TYPE_COLUMN_NAME); mlflow.log_param("violence_assignment_method", "rule_based_v2")
        mlflow.log_param("cols_excluded_from_scaling", cols_excluded)
        mlflow.log_param("priority_logic_version", "v2_type_sensitive") # Indicar versión de lógica
        print("Parámetros registrados en MLflow.")

        # --- Flujo ---
        conn = connect_to_sql_server()
        df_original = fetch_data_from_db(conn)
        if df_original is None or df_original.empty: print("Error crítico datos iniciales."); mlflow.end_run(status="FAILED"); return

        # Asignar tipo por reglas v2
        df_original = assign_violence_type_rule_based_v2(df_original, VIOLENCE_TYPE_COLUMN_NAME)
        if df_original is None: print("Error asignación tipo."); mlflow.end_run(status="FAILED"); return

        # Preprocesar y escalar (con exclusiones)
        df_scaled, scaler, valid_indices = preprocess_and_scale_data(df_original.copy(), cols_to_exclude_from_scaling=cols_excluded)
        if valid_indices is None: print("Fallo preprocesamiento."); mlflow.end_run(status="FAILED"); return

        features_scaled = list(df_scaled.columns) if df_scaled is not None else []
        mlflow.log_param("features_used_for_scaling", features_scaled if features_scaled else "None")

        # Aplicar Algoritmos
        if df_scaled is not None and not df_scaled.empty:
            df_pca, pca_model = apply_pca(df_scaled, n_components=n_pca_components)
            dbscan_labels, dbscan_model = apply_dbscan(df_scaled, eps=dbscan_eps, min_samples=dbscan_min_samples)
            isolation_forest_anomalies, iforest_model = apply_isolation_forest(df_scaled, contamination=iforest_contamination)
        else: print("No hay datos escalados para aplicar algoritmos."); df_pca, dbscan_labels, isolation_forest_anomalies = None, None, None

        # --- Combinar Resultados ---
        results_temp = pd.DataFrame(index=valid_indices)
        if df_pca is not None: results_temp = results_temp.join(df_pca)
        results_temp['Cluster_Label'] = dbscan_labels if dbscan_labels is not None else -99
        results_temp['Anomaly_IF'] = isolation_forest_anomalies if isolation_forest_anomalies is not None else -99
        # Añadir la columna de tipo de violencia a results_temp para pasarla a classify_priority
        # Asegurarse de tomarla del df_original *alineada por los índices válidos*
        if VIOLENCE_TYPE_COLUMN_NAME in df_original.columns:
            results_temp[VIOLENCE_TYPE_COLUMN_NAME] = df_original.loc[valid_indices, VIOLENCE_TYPE_COLUMN_NAME]
        else: # Si la columna de tipo no existe por alguna razón
             print(f"Error: Columna {VIOLENCE_TYPE_COLUMN_NAME} no encontrada para clasificación de prioridad.")
             results_temp[VIOLENCE_TYPE_COLUMN_NAME] = 'TipoDesconocido' # Crear columna dummy para evitar fallo


        # --- Clasificar Prioridad (usando la función revisada) ---
        if not results_temp.empty:
            priorities = classify_priority(
                results_temp, # Pasar el df con IF, DBSCAN, PCA y TIPO
                type_column_name=VIOLENCE_TYPE_COLUMN_NAME, # Pasar el nombre de la columna de tipo
                pca_threshold_std_dev=pca_extremeness_threshold
            )
            if priorities is not None:
                # Añadir la prioridad a results_temp ANTES de unir de vuelta
                results_temp['Priority'] = priorities
            else:
                results_temp['Priority'] = 'ErrorClasif'
        else:
            # No añadir columna de prioridad si no hubo nada que clasificar
             pass # Se manejará en el join/fillna

        # --- Unir al Original y Rellenar ---
        # Quitar la columna de tipo de results_temp si existe, para evitar duplicado en join
        cols_to_join = [col for col in results_temp.columns if col != VIOLENCE_TYPE_COLUMN_NAME]
        df_final = df_original.join(results_temp[cols_to_join]) # Unir resultados (sin tipo)

        # Rellenar NaNs en columnas generadas (para filas no procesadas)
        generated_cols_final = cols_to_join # Columnas que realmente se unieron
        fill_values = {}
        for col in generated_cols_final:
            if col.startswith('PC'): fill_values[col] = 0
            elif col in ['Cluster_Label', 'Anomaly_IF']: fill_values[col] = -99
            elif col == 'Priority': fill_values[col] = 'Indeterminada' # Default para filas no procesadas
        for col, val in fill_values.items():
             if col in df_final.columns: df_final[col] = df_final[col].fillna(val)

        # Asegurar tipos finales
        if 'Cluster_Label' in df_final.columns: df_final['Cluster_Label'] = df_final['Cluster_Label'].astype(int)
        if 'Anomaly_IF' in df_final.columns: df_final['Anomaly_IF'] = df_final['Anomaly_IF'].astype(int)
        if 'Priority' in df_final.columns: df_final['Priority'] = df_final['Priority'].astype(str)
        if VIOLENCE_TYPE_COLUMN_NAME in df_final.columns: df_final[VIOLENCE_TYPE_COLUMN_NAME] = df_final[VIOLENCE_TYPE_COLUMN_NAME].fillna('ErrorReglas').astype(str)


        print("\n--- Resultados Finales (Prioridad v2) ---")
        cols_orig_subset = [c for c in df_original.columns if c not in generated_cols_final and c != VIOLENCE_TYPE_COLUMN_NAME][:5]
        cols_rules_input = ['edad'] + [f'p{i}' for i in range(1,8)]
        # Asegurar que 'Priority' está al final
        cols_generated_final = [c for c in generated_cols_final if c != 'Priority'] + ['Priority'] if 'Priority' in generated_cols_final else []
        cols_to_show = cols_orig_subset + cols_rules_input + [VIOLENCE_TYPE_COLUMN_NAME] + cols_generated_final
        print(df_final[list(dict.fromkeys(cols_to_show))].head()) # Mostrar cols relevantes sin duplicados


        # --- Registrar Métricas (igual que antes, la lógica de cálculo no cambia) ---
        total_samples_original = len(df_original); total_samples_processed = len(valid_indices)
        mlflow.log_metric("total_samples_original", total_samples_original); mlflow.log_metric("total_samples_processed", total_samples_processed)
        mlflow.log_metric("samples_dropped_scaling_step", total_samples_original - total_samples_processed)
        # ... (Métricas DBSCAN, IF, PCA) ...
        if total_samples_processed > 0 and df_scaled is not None and not df_scaled.empty:
             if dbscan_labels is not None: db_out = sum(1 for l in dbscan_labels if l == -1); mlflow.log_metric("dbscan_outliers_count", db_out); mlflow.log_metric("dbscan_outliers_perc", (db_out / total_samples_processed) * 100)
             if isolation_forest_anomalies is not None: if_out = sum(1 for a in isolation_forest_anomalies if a == -1); mlflow.log_metric("iforest_anomalies_count", if_out); mlflow.log_metric("iforest_anomalies_perc", (if_out / total_samples_processed) * 100)
             if pca_model: mlflow.log_metric("pca_total_variance_explained", np.sum(pca_model.explained_variance_ratio_)); [mlflow.log_metric(f"pca_var_expl_PC{i+1}", v) for i, v in enumerate(pca_model.explained_variance_ratio_)]

        # Métricas de Prioridad (AHORA REFLEJAN LA NUEVA LÓGICA)
        if 'Priority' in df_final.columns:
            pri_counts = df_final['Priority'].value_counts(); print("\nDistribución Prioridades (v2):"); print(pri_counts)
            for level, count in pri_counts.items(): mlflow.log_metric(f"priority_{level}_count", count); mlflow.log_metric(f"priority_{level}_perc", (count / total_samples_original) * 100)

        # Métricas de Tipo de Violencia (igual que antes)
        if VIOLENCE_TYPE_COLUMN_NAME in df_final.columns:
            vt_counts = df_final[VIOLENCE_TYPE_COLUMN_NAME].value_counts(); print(f"\nDistribución Tipos ('{VIOLENCE_TYPE_COLUMN_NAME}'):"); print(vt_counts)
            for vtype, count in vt_counts.items(): mname = ''.join(e for e in str(vtype) if e.isalnum() or e in ['_','-']).lower() or "unknown"; mlflow.log_metric(f"vtype_{mname}_count", count); mlflow.log_metric(f"vtype_{mname}_perc", (count / total_samples_original) * 100)

        print("Métricas registradas en MLflow.")

        # --- Registrar Artefactos ---
        output_csv_path = f"resultados_prioridad_v2_{run_uuid[:8]}.csv"
        df_final.to_csv(output_csv_path, index=False)
        mlflow.log_artifact(output_csv_path)
        print(f"Resultados guardados y registrados como artefacto: {output_csv_path}")

        if conn: conn.close(); print("Conexión SQL Server cerrada.")
        print("MLflow Run completed.")


if __name__ == "__main__":
    main()
    print("\nPara ver los resultados, ejecuta 'mlflow ui' en tu terminal.")
    print(f"Busca el experimento '{MLFLOW_EXPERIMENT_NAME}'.")
