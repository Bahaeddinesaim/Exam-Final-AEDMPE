# Rapport Management - Forecast & Promotion Touristique

## Besoin metier
Identifier les pays et destinations a promouvoir en priorite, sous contrainte de budget et de risque externe, sans supposer que les donnees brutes sont parfaites.

## Problematique
La demande touristique future est estimee a partir d'un signal mensuel pays. La recommandation destination combine cette prevision avec qualite, sentiment, cout, meteo et historique campagne.

## Architecture
Le projet suit une architecture bronze/raw, processed et gold. Les fichiers bruts sont conserves intacts dans `data/raw`; les controles et transformations sont versionnables dans `src`.

## Nettoyage et gouvernance
Les pays sont harmonises, les destinations anonymisees sont mappees vers des noms metier, la colonne `destination_original` conserve la tracabilite source, les formats numeriques sont parses, les budgets non numeriques restent manquants, et les contradictions de campagne sont signalees plutot que masquees.

## GOLD DATA
La GOLD DATA est a granularite pays-destination. Les variables derivees principales sont `demand_growth`, `tourism_score`, `quality_score`, `sentiment_score`, `weather_penalty`, `campaign_efficiency`, `forecasted_demand` et `marketing_priority`.

## Choix IA
Le probleme est une prevision supervisee de serie temporelle mensuelle. Le split est temporel afin d'eviter une fuite d'information. Deux baselines sont comparees a Random Forest et Gradient Boosting.

## Resultats modeles
| Model                            | Approach type                            |       MAE |      RMSE |      MAPE |         R2 | Business interpretation                                                      | Status      |
|:---------------------------------|:-----------------------------------------|----------:|----------:|----------:|-----------:|:-----------------------------------------------------------------------------|:------------|
| Gradient Boosting + lag features | Supervised forecasting with lag features |   6.85324 |   8.67352 |   7.30196 |   0.855826 | Candidate model for country demand forecasting and marketing prioritization. | OK          |
| Random Forest + lag features     | Supervised forecasting with lag features |   7.69015 |   9.63958 |   8.15029 |   0.821921 | Candidate model for country demand forecasting and marketing prioritization. | OK          |
| Baseline Persistence             | Temporal baseline                        |   9.93929 |  12.1561  |  10.7011  |   0.716806 | Reference point used to prove whether advanced AI adds value.                | OK          |
| Baseline Moving Average          | Temporal baseline                        |  13.2375  |  14.9413  |  14.3113  |   0.572171 | Reference point used to prove whether advanced AI adds value.                | OK          |
| SARIMAX                          | Statistical time-series model            |  16.8858  |  20.3908  |  19.1977  |   0.203169 | Candidate model for country demand forecasting and marketing prioritization. | OK          |
| Prophet                          | Additive time-series model               | nan       | nan       | nan       | nan        | Model unavailable or no valid predictions.                                   | Unavailable |

## Data Quality
Les controles couvrent valeurs manquantes, types, cardinalites et contradictions. Extrait:
| dataset      | column               | dtype          |   rows |   missing |   missing_rate |   unique_values |
|:-------------|:---------------------|:---------------|-------:|----------:|---------------:|----------------:|
| destinations | country              | object         |    160 |         0 |              0 |               8 |
| destinations | destination          | object         |    160 |         0 |              0 |             160 |
| destinations | attractiveness       | float64        |    160 |         0 |              0 |             123 |
| destinations | cost                 | int64          |    160 |         0 |              0 |             155 |
| destinations | rating               | float64        |    160 |         0 |              0 |             103 |
| destinations | visitors             | int64          |    160 |         0 |              0 |             160 |
| destinations | destination_original | object         |    160 |         0 |              0 |              50 |
| market       | country              | object         |    287 |         0 |              0 |               8 |
| market       | month                | datetime64[ns] |    287 |         0 |              0 |              36 |
| market       | demand_index         | float64        |    287 |         0 |              0 |             243 |
| reviews      | country              | object         |    800 |         0 |              0 |               8 |
| reviews      | destination          | object         |    800 |         0 |              0 |             183 |

## Recommandations
Top recommandations:
| country   | destination   | destination_original   |   rank_country |   marketing_priority |   forecasted_demand |   quality_score |   sentiment_score |   weather_penalty |   campaign_efficiency |   allocated_budget | recommendation_reason                                                        |
|:----------|:--------------|:-----------------------|---------------:|---------------------:|--------------------:|----------------:|------------------:|------------------:|----------------------:|-------------------:|:-----------------------------------------------------------------------------|
| Morocco   | Fès           | City_39                |              1 |             0.65615  |         2.07459e+07 |        0.558633 |          1        |              0    |               0       |              12100 | Demand forecast=20745947, quality=0.56, sentiment=1.00, weather penalty=0.00 |
| Japan     | Hiroshima     | City_3                 |              1 |             0.580756 |         1.66319e+07 |        0.672121 |          1        |              0.25 |               0       |              10700 | Demand forecast=16631943, quality=0.67, sentiment=1.00, weather penalty=0.25 |
| Spain     | Madrid        | City_10                |              1 |             0.554049 |         2.05248e+07 |        0.601535 |          0        |              0.25 |               2.93333 |              10300 | Demand forecast=20524842, quality=0.60, sentiment=0.00, weather penalty=0.25 |
| Portugal  | Faro          | City_43                |              1 |             0.540911 |         1.88256e+07 |        0.412457 |          0        |              0.25 |               0       |              10000 | Demand forecast=18825614, quality=0.41, sentiment=0.00, weather penalty=0.25 |
| USA       | Las Vegas     | City_17                |              1 |             0.538001 |         1.97061e+07 |        0.77496  |          0        |              0    |               0       |              10000 | Demand forecast=19706125, quality=0.77, sentiment=0.00, weather penalty=0.00 |
| Italy     | Venice        | City_37                |              1 |             0.53161  |         1.64617e+07 |        0.451757 |          1        |              0.25 |               0       |               9800 | Demand forecast=16461692, quality=0.45, sentiment=1.00, weather penalty=0.25 |
| Japan     | Osaka         | City_6                 |              2 |             0.528362 |         1.63783e+07 |        0.51241  |          1        |              0.55 |               0       |               9800 | Demand forecast=16378337, quality=0.51, sentiment=1.00, weather penalty=0.55 |
| Japan     | Kobe          | City_31                |              3 |             0.522484 |         1.43191e+07 |        0.48746  |          1        |              0.25 |               0       |               9700 | Demand forecast=14319135, quality=0.49, sentiment=1.00, weather penalty=0.25 |
| Portugal  | Coimbra       | City_8                 |              2 |             0.521188 |         1.37002e+07 |        0.868498 |          0.666667 |              0.25 |               0       |               9600 | Demand forecast=13700188, quality=0.87, sentiment=0.67, weather penalty=0.25 |
| Morocco   | Tanger        | City_14                |              2 |             0.515389 |         1.85371e+07 |        0.619178 |          0        |              0.25 |               0       |               9500 | Demand forecast=18537125, quality=0.62, sentiment=0.00, weather penalty=0.25 |

## Limites, biais et risques
Les destinations sont anonymisees, la cible officielle est un texte non tabulaire, les avis sociaux sont sujets a biais de representation, et la meteo est categorielle. Les scores doivent donc etre utilises comme aide a la decision, pas comme verite automatique.

## Pistes futures
Connecter des donnees calendaires reelles, ajouter elasticite prix, saisonnalite par destination, tests A/B campagne, et monitoring MLOps des performances.
