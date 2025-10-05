# Prédiction du Niveau d'Eau d'un Réservoir par Machine Learning

Projet réalisé chez **I3T** pour optimiser la gestion d’un réservoir agricole via des modèles prédictifs.

**Auteure :** Yesmine Srairi  
**Encadrant :** M. Sahbi Gsouma  
**Période :** Juillet - Août 2025  

---

## À propos

Ce projet transforme la gestion réactive d’un réservoir en approche prédictive basée sur l’IA. Les modèles anticipent les variations du niveau d’eau pour éviter les débordements et optimiser le contrôle des pompes.

- **Données :** 30 621 observations horaires (janvier 2022 - juillet 2025)  

---

## Structure du projet



```project/```

```│```

```├── code/                           # Scripts Python du projet```

```│   ├── Data.py                     # Préparation et fusion des données```

```│   ├── TSA_Analysis.py             # Analyse exploratoire des séries temporelles```

```│   ├── SARIMAX.py                  # Modèle SARIMAX```

```│   └── LSTM.py                     # Modèle LSTM```

```│```

```├── data/                           # Données brutes et nettoyées (*.csv)```

```│```

```├── figures d'analyse exploratoire/ # Visualisations et graphiques```

```│```

```├── modeles/                        # Modèles entraînés```

```│   ├── sarimax_model.pkl           # Modèle SARIMAX sauvegardé```

```│   └── lstm_model.h5               # Modèle LSTM sauvegardé```

```│```

```├── rapports/                       # Documentation et rapports```

```│```

```└── Résultats/                      # Résultats des prédictions et analyses```

---

## Technologies

- **Langages :** Python , SQL, LaTeX  
- **Bibliothèques principales :** `pandas`, `numpy`, `statsmodels`, `tensorflow/keras`, `scikit-learn`, `matplotlib`, `seaborn`, `sqlalchemy`

---

## Phases du projet

1. **Analyse & Recommandation** : Étude méthodologique et choix technologiques  
2. **Business Intelligence** : Nettoyage des données SCADA et création de vues SQL agrégées  
3. **Modélisation Prédictive** : Développement et évaluation des modèles SARIMAX et LSTM  

---

## Résultats

| Métrique | SARIMAX | LSTM |
|----------|---------|------|
| MAE      | 0.1143  | 0.0802 |
| RMSE     | 0.2178  | 0.1764 |
| R²       | 0.8279  | 0.8431 |

> Les deux modèles expliquent >82% de la variance. LSTM est légèrement plus performant, SARIMAX offre une meilleure interprétabilité.

---
## Documentation
Rapports détaillés disponibles dans les fichiers PDF du projet :

-Analyse & Recommandation

-BI : Préparation des données SCADA

-Modélisation Prédictive


## Contact
yesminesr123@gmail.com

