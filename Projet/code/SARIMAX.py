import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import joblib

warnings.filterwarnings('ignore')

# Chargement des données
df_level = pd.read_csv('View_Reservoir_clean.csv', parse_dates=['Timestamp'], index_col='Timestamp')
df_A = pd.read_csv('vw_TROZA_A_clean.csv', parse_dates=['Timestamp'], index_col='Timestamp')
df_B = pd.read_csv('vw_TROZA_B_clean.csv', parse_dates=['Timestamp'], index_col='Timestamp')

#colonnes utiles
df_level = df_level[['niveau_avg']]
df_A = df_A[['Débit_A']]
df_B = df_B[['Débit_B']]

# Fusion et nettoyage
df = df_level.join([df_A, df_B], how='outer')
df = df.sort_index().asfreq('H')  # Note: uppercase 'H' is still fine here; change to 'h' if warning appears
df = df.ffill().bfill()

# variable cible
y = df['niveau_avg']
exog = df[['Débit_A', 'Débit_B']]

# Ajustement du modèle SARIMAX
print("Ajustement du modèle SARIMAX...")
model = SARIMAX(
    y,
    exog=exog,
    order=(1, 1, 2),
    seasonal_order=(1, 0, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)
print("Modèle ajusté.")

# Sauvegarde du modèle
joblib.dump(results, 'sarimax_model.pkl')
print("Modèle SARIMAX sauvegardé sous sarimax_model.pkl")

# Prédiction sur la période d’entraînement
pred = results.predict(start=y.index[0], end=y.index[-1], exog=exog)

# Évaluation du modèle
mae = mean_absolute_error(y, pred)
rmse = np.sqrt(mean_squared_error(y, pred))
r2 = r2_score(y, pred)

print("\nMétriques de performance")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# Visualisation des prédictions
plt.figure(figsize=(14, 6))
plt.plot(y, label='Valeurs réelles', color='blue')
plt.plot(pred, label='Valeurs prédites', color='red', alpha=0.7)
plt.title("Prédiction SARIMAX - niveau_avg")
plt.xlabel("Temps")
plt.ylabel("Niveau d'eau")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

#Analyse des résidus 
residuals = y - pred

plt.figure(figsize=(14, 4))
plt.plot(residuals)
plt.title("Résidus du modèle SARIMAX")
plt.xlabel("Temps")
plt.ylabel("Residu")
plt.grid(True)
plt.tight_layout()
plt.show()

plot_acf(residuals, lags=40)
plt.title("ACF oes résidus SARIMAX")
plt.tight_layout()
plt.show()

# test de Ljung-Box 
lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
print("\ntest de Ljung-Box  sur les résidus")
print(lb_test)
