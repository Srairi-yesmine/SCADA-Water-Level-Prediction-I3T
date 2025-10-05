import pandas as pd
import sqlalchemy
import os

# Database connection string
connection_string = "mssql+pyodbc://localhost\\SQLEXPRESS/ScadaNetDb?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"

def get_engine(conn_str):
    try:
        engine = sqlalchemy.create_engine(conn_str)
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        print(" Database connection successful!")
        return engine
    except Exception as e:
        print(f" Database connection failed: {e}")
        return None

def load_or_read_csv(view_name, file_path, engine):
    if os.path.exists(file_path):
        print(f" Loading {view_name} from local file '{file_path}'")
        return pd.read_csv(file_path)
    else:
        print(f" Loading {view_name} from database...")
        df = pd.read_sql(f"SELECT * FROM {view_name}", engine)
        df.to_csv(file_path, index=False)
        print(f" Saved {view_name} locally as '{file_path}'")
        return df

def unify_timestamps(df_a, df_b, df_reservoir):
    # Extract and unify timestamps
    timestamps_a = df_a['Heure'].drop_duplicates().sort_values().reset_index(drop=True)
    timestamps_b = df_b['Heure'].drop_duplicates().sort_values().reset_index(drop=True)
    timestamps_reservoir = df_reservoir['datetime_hour'].drop_duplicates().sort_values().reset_index(drop=True)

    # Create master timestamp index (union)
    all_timestamps = pd.Series(
        pd.concat([timestamps_a, timestamps_b, timestamps_reservoir])
    ).drop_duplicates().sort_values().reset_index(drop=True)
    print(f"\n Unified timestamps count: {len(all_timestamps)}")

    # Reindex each dataframe on unified timestamps
    df_a_reindexed = df_a.set_index('Heure').reindex(all_timestamps).reset_index()
    df_b_reindexed = df_b.set_index('Heure').reindex(all_timestamps).reset_index()
    df_reservoir_reindexed = df_reservoir.set_index('datetime_hour').reindex(all_timestamps).reset_index()

    # Rename index column to 'Timestamp' for consistency
    df_a_reindexed.rename(columns={'index': 'Timestamp'}, inplace=True)
    df_b_reindexed.rename(columns={'index': 'Timestamp'}, inplace=True)
    df_reservoir_reindexed.rename(columns={'index': 'Timestamp'}, inplace=True)

    print(" DataFrames reindexed with unified timestamps.")
    return df_a_reindexed, df_b_reindexed, df_reservoir_reindexed

def clean_troza_columns(df_a, df_b):
    # Rename columns to standardize names for later merging/analysis
    df_a = df_a.rename(columns={
        'Débit': 'Débit_A',
        'Production (24h)': 'Production_A'
    })
    df_b = df_b.rename(columns={
        'Débit (l/s)': 'Débit_B',
        'Production (/24h) (m³)': 'Production_B'
    })
    return df_a, df_b

def main():
    engine = get_engine(connection_string)
    if engine is None:
        return

    # File paths to save/load local copies
    file_paths = {
        'vw_TROZA_A': 'troza_a.csv',
        'vw_TROZA_B': 'troza_b.csv',
        'View_Reservoir': 'reservoir.csv'
    }

    # Load data (from CSV if exists, else from DB and save CSV)
    df_troza_a = load_or_read_csv('vw_TROZA_A', file_paths['vw_TROZA_A'], engine)
    df_troza_b = load_or_read_csv('vw_TROZA_B', file_paths['vw_TROZA_B'], engine)
    df_reservoir = load_or_read_csv('View_Reservoir', file_paths['View_Reservoir'], engine)

    # Clean / rename columns to unify Troza views
    df_troza_a, df_troza_b = clean_troza_columns(df_troza_a, df_troza_b)

    # Align timestamps
    df_a_aligned, df_b_aligned, df_reservoir_aligned = unify_timestamps(df_troza_a, df_troza_b, df_reservoir)

    # Step: Delete rows with any NaN in any view, based on shared timestamps
    combined = pd.concat([
        df_a_aligned.set_index('Timestamp'),
        df_b_aligned.set_index('Timestamp'),
        df_reservoir_aligned.set_index('Timestamp')
    ], axis=1)

    # Drop rows with any NaNs
    combined_clean = combined.dropna()

    # Split back into individual DataFrames
    df_a_clean = combined_clean[df_a_aligned.columns.difference(['Timestamp'])].reset_index()
    df_b_clean = combined_clean[df_b_aligned.columns.difference(['Timestamp'])].reset_index()
    df_reservoir_clean = combined_clean[df_reservoir_aligned.columns.difference(['Timestamp'])].reset_index()

    # Preview cleaned data
    print("\n--- Cleaned vw_TROZA_A ---")
    print(df_a_clean.head(3))
    print("\n--- Cleaned vw_TROZA_B ---")
    print(df_b_clean.head(3))
    print("\n--- Cleaned View_Reservoir ---")
    print(df_reservoir_clean.head(3))

    # Print number of rows after cleaning
    print(f"\nRows after cleaning:")
    print(f" vw_TROZA_A: {df_a_clean.shape[0]} rows")
    print(f" vw_TROZA_B: {df_b_clean.shape[0]} rows")
    print(f" View_Reservoir: {df_reservoir_clean.shape[0]} rows")

    # Save cleaned CSV files for later use
    df_a_clean.to_csv("vw_TROZA_A_clean.csv", index=False)
    df_b_clean.to_csv("vw_TROZA_B_clean.csv", index=False)
    df_reservoir_clean.to_csv("View_Reservoir_clean.csv", index=False)
    print("\nCleaned CSVs saved successfully.")

if __name__ == "__main__":
    main()
