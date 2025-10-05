import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class TSAAnalysis:
    def __init__(self):
        self.data = None
        self.target_col = 'niveau_avg'
        self.exog_cols = ['Débit_A', 'Débit_B']
        self.seasonal_period = 24
        self.model_results = []
        
    #chargement des données
    def load_data(self):

        print(" Chargement des données...")
        
        try:

            self.data = pd.read_csv("View_Reservoir_clean.csv")
            self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
            self.data.set_index('Timestamp', inplace=True)
            self.data = self.data.asfreq('H')  # fréquence horaire
            
            # variables exogènes
            df_A = pd.read_csv('vw_TROZA_A_clean.csv', parse_dates=['Timestamp'], index_col='Timestamp')
            df_B = pd.read_csv('vw_TROZA_B_clean.csv', parse_dates=['Timestamp'], index_col='Timestamp')
            
            # Fusion des variables exogènes
            self.data = self.data.join([df_A[['Débit_A']], df_B[['Débit_B']]], how='outer')
            self.data = self.data.ffill().bfill()  # Fill missing values
            
            print(f" Données chargées avec succès")
            print(f"   Dimensions: {self.data.shape}")
            print(f"   Période: {self.data.index.min()} to {self.data.index.max()}")
            print(f"   colonnes: {list(self.data.columns)}")
            
        except FileNotFoundError as e:
            print(f" Erreur lors du chargement {e}")


    #analyse exploratoire 
    def exploratory_analysis(self):
     
        print("\n analyse exploratoire ")
        print("="*50)
        

        print("\n Statistiques descriptives:")
        print(self.data.describe())
        
    
        print(f"\n Valeurs manquantes :")
        missing = self.data.isnull().sum()
        print(missing)
        
        # Tracé de la série temporelle
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Variable cible
        axes[0].plot(self.data.index, self.data[self.target_col], alpha=0.7, color='blue')
        axes[0].set_title(f'{self.target_col} - Niveau d\'eau au cours du temps')
        axes[0].set_ylabel('Niveau d\'eau')
        axes[0].grid(True, alpha=0.3)
        
        # variables exogènes
        axes[1].plot(self.data.index, self.data['Débit_A'], alpha=0.7, color='red', label='Débit_A')
        axes[1].plot(self.data.index, self.data['Débit_B'], alpha=0.7, color='green', label='Débit_B')
        axes[1].set_title('Variables exogènes - Débits')
        axes[1].set_ylabel('Débit')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Corrélation avec la variable cible
        correlation_A = self.data[self.target_col].corr(self.data['Débit_A'])
        correlation_B = self.data[self.target_col].corr(self.data['Débit_B'])
        
        axes[2].scatter(self.data['Débit_A'], self.data[self.target_col], alpha=0.5, 
                       label=f'Débit_A (corr: {correlation_A:.3f})')
        axes[2].scatter(self.data['Débit_B'], self.data[self.target_col], alpha=0.5, 
                       label=f'Débit_B (corr: {correlation_B:.3f})')
        axes[2].set_title('Niveau d\'eau vs Débitss')
        axes[2].set_xlabel('Débit')
        axes[2].set_ylabel('Niveau d\'eau')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n CCorrélations avec {self.target_col}:")
        print(f"   Débit_A: {correlation_A:.4f}")
        print(f"   Débit_B: {correlation_B:.4f}")

    #analyse de la stationnarité
    def stationarity_analysis(self):

        print("\n ANALYSE DE LA STATIONNARITÉ")
        print("="*50)
        
        series = self.data[self.target_col].dropna()
        
        # Test ADF 
        print("\n  Test ADF :")
        adf_result = adfuller(series)
        print(f"   Statistique ADF: {adf_result[0]:.4f}")
        print(f"   p-value: {adf_result[1]:.6f}")
        print(f"   Valeurs critiques :")
        for key, value in adf_result[4].items():
            print(f"      {key}: {value:.4f}")
        
        if adf_result[1] < 0.05:
            print("    La série est stationnaire (ADF)")
        else:
            print("    La série n'est pas stationnaire (ADF)")
            
        #Test KPSS 
        print("\n KPSS Test:")
        kpss_result = kpss(series)
        print(f"  Statistique KPSS : {kpss_result[0]:.4f}")
        print(f"   p-value: {kpss_result[1]:.6f}")
        print(f"   Valeurs critiques:")
        for key, value in kpss_result[3].items():
            print(f"      {key}: {value:.4f}")
            
        if kpss_result[1] > 0.05:
            print("    La série est stationnaire  (KPSS)")
        else:
            print("    La série n'est pas stationnaire (KPSS)")
            
        # Graphiques série originale et différenciée
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Série originale
        axes[0,0].plot(series, alpha=0.7, color='blue')
        axes[0,0].set_title('Série originale')
        axes[0,0].grid(True, alpha=0.3)
        
        # Première différence
        diff1 = series.diff().dropna()
        axes[0,1].plot(diff1, alpha=0.7, color='red')
        axes[0,1].set_title('Première différence')
        axes[0,1].grid(True, alpha=0.3)
        
        # ACF de la série originale
        plot_acf(series, ax=axes[1,0], lags=50, alpha=0.05)
        axes[1,0].set_title('ACF - Série originale')
        
        # ACF of differenced
        plot_acf(diff1, ax=axes[1,1], lags=50, alpha=0.05)
        axes[1,1].set_title('ACF - Première différence')
        
        plt.tight_layout()
        plt.show()
        
        return adf_result, kpss_result

    #Analyse des motifs saisonniers  
    def seasonal_analysis(self):

        print("\n ANALYSE SAISONNIÈRE")
        print("="*50)
        
        series = self.data[self.target_col].dropna()
        
        try:
            # Décomposition saisonnière
            decomposition = seasonal_decompose(series, model='additive', period=self.seasonal_period)
            
            # Tracé de la décomposition
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            decomposition.observed.plot(ax=axes[0], title='Série originale')
            decomposition.trend.plot(ax=axes[1], title='Composante tendance')
            decomposition.seasonal.plot(ax=axes[2], title=f'Composante saisonnière (Période: {self.seasonal_period}h)')
            decomposition.resid.plot(ax=axes[3], title='Résidus')
            
            for ax in axes:
                ax.grid(True, alpha=0.3)
                
            plt.tight_layout()
            plt.show()
            
            # Force saisonnière
            seasonal_strength = np.var(decomposition.seasonal) / np.var(decomposition.observed)
            print(f" Force saisonnière: {seasonal_strength:.4f}")
            
            # Motifs horaires moyens
            hourly_avg = series.groupby(series.index.hour).mean()
            hourly_std = series.groupby(series.index.hour).std()
            
            plt.figure(figsize=(12, 6))
            plt.errorbar(hourly_avg.index, hourly_avg.values, yerr=hourly_std.values, 
                        capsize=5, marker='o', linewidth=2)
            plt.title('Niveau d\'eau moyen selon l\'heure de la journée')
            plt.xlabel('Heure')
            plt.ylabel('Niveau d\'eau')
            plt.grid(True, alpha=0.3)
            plt.xticks(range(0, 24))
            plt.show()
            
        except Exception as e:
            print(f" La décomposition saisonnière a échoué: {e}")
    #Analyse des fonctions ACF et PACF        
    def acf_pacf_analysis(self):
    
        print("\n ACF/PACF ANALYSIS")
        print("="*50)
        
        series = self.data[self.target_col].dropna()
        diff_series = series.diff().dropna()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        #  serie originale
        plot_acf(series, ax=axes[0,0], lags=50, alpha=0.05)
        axes[0,0].set_title('ACF - erie originale')
        
        plot_pacf(series, ax=axes[0,1], lags=50, alpha=0.05)
        axes[0,1].set_title('PACF - erie originale')
        
        # Série différenciée
        plot_acf(diff_series, ax=axes[1,0], lags=50, alpha=0.05)
        axes[1,0].set_title('ACF - Série différenciée')
        
        plot_pacf(diff_series, ax=axes[1,1], lags=50, alpha=0.05)
        axes[1,1].set_title('PACF - First Difference')
        
        plt.tight_layout()
        plt.show()
        
    #Recherche des meilleurs paramètres SARIMAX
    def model_selection_grid_search(self):
 

    #Combinaison de modèles
        model_combinations = [  
    ((1,1,1), (1,0,1,24)),  
    ((1,1,1), (1,1,1,24)),  
    ((1,1,2), (1,0,1,24)),
    ((1,1,2), (1,1,1,24)),
]

        
        print(f" Test de {len(model_combinations)}")
        for i, (order, seasonal_order) in enumerate(model_combinations):
            print(f"   {i+1}. SARIMAX{order}x{seasonal_order}")
        print()
        
        series = self.data[self.target_col].dropna()
        exog = self.data[self.exog_cols].dropna()
        
      
        common_index = series.index.intersection(exog.index)
        series = series.loc[common_index]
        exog = exog.loc[common_index]
        
        best_aic = float('inf')
        best_params = None
        self.model_results = []
        

        
        for i, (order, seasonal_order) in enumerate(model_combinations):
            try:
                print(f"   Test de{i+1}/{len(model_combinations)}: SARIMAX{order}x{seasonal_order}...", end=" ")
                
                #  Ajustement du modèle
                model = SARIMAX(series, 
                               exog=exog,
                               order=order,
                               seasonal_order=seasonal_order,
                               enforce_stationarity=False,
                               enforce_invertibility=False)
                
                fitted = model.fit(disp=False, maxiter=50)
                
                # Stockage des résultats
                result = {
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'aic': fitted.aic,
                    'bic': fitted.bic,
                    'log_likelihood': fitted.llf,
                    'params': len(fitted.params)
                }
                
                self.model_results.append(result)
                
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_params = (order, seasonal_order)
                
                print(f"AIC: {fitted.aic:.2f} ✓")
                        
            except Exception as e:
                print(f"Failed: {str(e)[:50]}...")
                continue
        
        # Affichage des résultats
        if self.model_results:
            results_df = pd.DataFrame(self.model_results)
            results_df = results_df.sort_values('aic')
            
            print(f"\n TOUS LES RÉSULTATS (triés par AIC):")
            print("="*50)
            print(results_df.round(2))
            
            print(f"\n MEILLEUR MODÈLE :")
            print(f"   Ordre: {best_params[0]}")
            print(f"   Ordre saisonnier : {best_params[1]}")
            print(f"   AIC: {best_aic:.2f}")
            
            return best_params, results_df
        else:
            print(" Aucun modèle ajusté avec succès")
            return None, None
    
    #Comparaison des modèles avec et sans variables exogènes
    def compare_with_without_exog(self, best_order, best_seasonal_order):
       
        print(f"\n IMPACT DES VARIABLES EXOGÈNES")
        print("="*50)
        
        series = self.data[self.target_col].dropna()
        exog = self.data[self.exog_cols].dropna()
        
        
        common_index = series.index.intersection(exog.index)
        series = series.loc[common_index]
        exog = exog.loc[common_index]
        
        # Séparation entraînement/test
        split_idx = int(len(series) * 0.8)
        train_series = series.iloc[:split_idx]
        test_series = series.iloc[split_idx:]
        train_exog = exog.iloc[:split_idx]
        test_exog = exog.iloc[split_idx:]
        
        models = {}
        
        try:
            #  Modèle sans exogènes
            print(" Ajustement du modèle SANS variables exogènes...")
            model_no_exog = SARIMAX(train_series, 
                                   order=best_order,
                                   seasonal_order=best_seasonal_order)
            fitted_no_exog = model_no_exog.fit(disp=False)
            
            # Prédiction
            forecast_no_exog = fitted_no_exog.forecast(steps=len(test_series))
            rmse_no_exog = np.sqrt(mean_squared_error(test_series, forecast_no_exog))
            mae_no_exog = mean_absolute_error(test_series, forecast_no_exog)
            
            models['without_exog'] = {
                'model': fitted_no_exog,
                'aic': fitted_no_exog.aic,
                'bic': fitted_no_exog.bic,
                'rmse': rmse_no_exog,
                'mae': mae_no_exog
            }
            
        except Exception as e:
            print(f"  Échec modèle sans exogènes {e}")
            
        try:
            #  modèle AVEC variables exogènes
            print(" Ajustement du modèle AVEC variables exogènes...")
            model_with_exog = SARIMAX(train_series, 
                                     exog=train_exog,
                                     order=best_order,
                                     seasonal_order=best_seasonal_order)
            fitted_with_exog = model_with_exog.fit(disp=False)
            
            #  Prédiction
            forecast_with_exog = fitted_with_exog.forecast(steps=len(test_series), exog=test_exog)
            rmse_with_exog = np.sqrt(mean_squared_error(test_series, forecast_with_exog))
            mae_with_exog = mean_absolute_error(test_series, forecast_with_exog)
            
            models['with_exog'] = {
                'model': fitted_with_exog,
                'aic': fitted_with_exog.aic,
                'bic': fitted_with_exog.bic,
                'rmse': rmse_with_exog,
                'mae': mae_with_exog
            }
            
        except Exception as e:
            print(f" Model with exog failed: {e}")
        
        # Comparaison
        if len(models) == 2:
            print(f"\n COMPARAISON DES MODÈLES")
            print("-"*50)
            print(f"{'Métrique':<15} {'Sans Exogène':<15} {'Avec Exogène':<15} {'Improvement':<15}")
            print("-"*60)
            
            aic_improve = models['without_exog']['aic'] - models['with_exog']['aic']
            bic_improve = models['without_exog']['bic'] - models['with_exog']['bic']
            rmse_improve = models['without_exog']['rmse'] - models['with_exog']['rmse']
            mae_improve = models['without_exog']['mae'] - models['with_exog']['mae']
            
            print(f"{'AIC':<15} {models['without_exog']['aic']:<15.2f} {models['with_exog']['aic']:<15.2f} {aic_improve:<15.2f}")
            print(f"{'BIC':<15} {models['without_exog']['bic']:<15.2f} {models['with_exog']['bic']:<15.2f} {bic_improve:<15.2f}")
            print(f"{'RMSE':<15} {models['without_exog']['rmse']:<15.4f} {models['with_exog']['rmse']:<15.4f} {rmse_improve:<15.4f}")
            print(f"{'MAE':<15} {models['without_exog']['mae']:<15.4f} {models['with_exog']['mae']:<15.4f} {mae_improve:<15.4f}")
            
            # Recommendation
            if aic_improve > 0 and rmse_improve > 0:
                print(f"\n RECOMMANDATION : Utiliser le modèle AVEC variables exogènes.")
            else:
                print(f"\n RECOMMANDATION : Utiliser le modèle SANS variables exogènes.")
                
        return models
        
    #Analyse complete  
    def run_complete_analysis(self):
        
        print(" STARTING COMPLETE TIME SERIES ANALYSIS")
        print("="*70)

        self.load_data()
        if self.data is None:
            return
            
      
        self.exploratory_analysis()
        
    
        adf_result, kpss_result = self.stationarity_analysis()

        self.seasonal_analysis()
       
        self.acf_pacf_analysis()
        
        best_params, results_df = self.model_selection_grid_search()
        
        if best_params:
    
            models = self.compare_with_without_exog(best_params[0], best_params[1])
            
            # 8. Sauvegarde des résultats
            self.save_analysis_results(best_params, results_df, models)
            
        print(f"\n ANALYSE TERMINÉE")
        print("="*70)
        
        return {
            'best_params': best_params,
            'results_df': results_df,
            'stationarity': {'adf': adf_result, 'kpss': kpss_result}
        }
        
    def save_analysis_results(self, best_params, results_df, models):
        """Sauvegarde les résultats de l’analyse dans des fichiers."""
        print(f"\n Saving analysis results...")
        
        
        if results_df is not None:
            results_df.to_csv('tsa_model_selection_results.csv', index=False)
            print("    Model selection results saved to 'tsa_model_selection_results.csv'")
        
        
        with open('tsa_best_parameters.txt', 'w') as f:
            f.write("MEILLEURS PARAMÈTRES SARIMAX\n")
            f.write("="*30 + "\n")
            f.write(f"Ordre : {best_params[0]}\n")
            f.write(f"Ordre saisonnier :  {best_params[1]}\n")
            f.write(f"Recommandé :{best_params[0]}x{best_params[1]}\n")
        print("    Best parameters saved to 'tsa_best_parameters.txt'")
        

        with open('tsa_analysis_summary.txt', 'w') as f:
            f.write("RÉSUMÉ DE L'ANALYSE DES SÉRIES TEMPORELLES\n")
            f.write("="*40 + "\n")
            f.write(f"Taille des données: {self.data.shape}\n")
            f.write(f"Variable cible :  {self.target_col}\n")
            f.write(f"Variables exogènes : {self.exog_cols}\n")
            f.write(f"Période saisonnière  {self.seasonal_period} hours\n")
            f.write(f"Modèle optimal : SARIMAX{best_params[0]}x{best_params[1]}\n")
        print("     Résumé de l’analyse enregistré dans'tsa_analysis_summary.txt'")

# Usage
if __name__ == "__main__":
    analyzer = TSAAnalysis()
    results = analyzer.run_complete_analysis()