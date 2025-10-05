import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
from tensorflow.keras.models import load_model

# Charger les données 
df_res = pd.read_csv('View_Reservoir_clean.csv', parse_dates=['Timestamp'], index_col='Timestamp')
df_a = pd.read_csv('vw_TROZA_A_clean.csv', parse_dates=['Timestamp'], index_col='Timestamp')
df_b = pd.read_csv('vw_TROZA_B_clean.csv', parse_dates=['Timestamp'], index_col='Timestamp')

# Variables à conserver
features = [
    'niveau_avg', 'Débit_A', 'Débit_B',
    'debit_dn100_avg', 'debit_dn250_avg',
    'niveau_tres_bas', 'niveau_tres_haut',
    'Production_A', 'Production_B',
    'Marche_A', 'Marche_B'
]

# Fusionner les datasets sur l’index Timestamp
df = df_res.join([df_a, df_b], how='outer').sort_index()
df = df.asfreq('h').ffill().bfill()
df = df[features]

# Séparer les colonnes numériques et catégoriques
numerical_cols = [
    'niveau_avg', 'Débit_A', 'Débit_B',
    'debit_dn100_avg', 'debit_dn250_avg',
    'Production_A', 'Production_B'
]

categorical_cols = [
    'niveau_tres_bas', 'niveau_tres_haut',
    'Marche_A', 'Marche_B'
]

#  Appliquer le MinMaxScaler aux colonnes numériques
scaler = MinMaxScaler()
df_scaled_num = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols, index=df.index)

# Combiner les colonnes numériques normalisées + colonnes binaires 
df_final = pd.concat([df_scaled_num, df[categorical_cols]], axis=1)

#  Création des séquences pour l’entraînement LSTM 
SEQ_LEN = 24

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])  # Prédiction sur 'niveau_avg'
    return np.array(X), np.array(y)

X, y = create_sequences(df_final.values, SEQ_LEN)

# Séparation Entraînement/Test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# les timestamps du jeu de test
timestamps = df_final.index
test_timestamps = timestamps[SEQ_LEN + train_size : SEQ_LEN + train_size + len(y_test)]


# Vérification de la variabilité de la cible pendant la période de test
test_period_original = df['niveau_avg'].loc[test_timestamps[0] : test_timestamps[-1]]
print(test_period_original.describe())
test_period_original.plot(title='niveau_avg original pendant la période de test')
plt.show()

# Construire le modèle LSTM 
model = Sequential([
    Input(shape=(SEQ_LEN, X.shape[2])),
    LSTM(64, activation='tanh'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Entraîner le modèle avec arrêt anticipé
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# Fonction pour dénormaliser les prédictions
def inverse_scale_lstm(y_scaled, scaler, col_index=0):
    dummy = np.zeros((len(y_scaled), scaler.n_features_in_))
    dummy[:, col_index] = y_scaled
    return scaler.inverse_transform(dummy)[:, col_index]

y_pred_scaled = model.predict(X_test).flatten()
y_test_inv = inverse_scale_lstm(y_test, scaler, col_index=0)
y_pred_inv = inverse_scale_lstm(y_pred_scaled, scaler, col_index=0)

# Évaluer les performances du modèle
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
r2 = r2_score(y_test_inv, y_pred_inv)

print("\n=== Performances du modèle LSTM ===")
print(f"MAE :  {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R² :   {r2:.4f}")

# Afficher les prédictions vs valeurs réelles 
plt.figure(figsize=(14,6))
plt.plot(y_test_inv, label='Valeur réelle niveau_avg')
plt.plot(y_pred_inv, label='Prédiction niveau_avg (LSTM)', alpha=0.7)
plt.title("Prédiction LSTM avec variables numériques normalisées + variables binaires brutes")
plt.xlabel("Pas de temps")
plt.ylabel("Niveau d’eau")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Sauvegarder le modèle LSTM entraîné 
model.save('lstm_model.h5')
