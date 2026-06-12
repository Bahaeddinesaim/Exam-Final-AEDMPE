# Best Model Summary

Le modele retenu est **Gradient Boosting + lag features** car il obtient la meilleure performance RMSE sur l'ensemble de test temporel: MAE=6.85, RMSE=8.67, MAPE=7.30% et R2=0.86. Il doit etre lu avec prudence: la performance statistique ne remplace pas l'analyse de robustesse, l'interpretabilite, la qualite des donnees et la validation metier marketing.

Le split temporel respecte l'ordre chronologique des observations et evite la fuite de donnees. Les modeles Random Forest et XGBoost/Gradient Boosting ont ete adaptes au forecasting via la creation de variables de retard et de moyennes glissantes. Ils ne sont donc pas utilises comme simples modeles tabulaires, mais comme approches de prevision supervisee temporelle.

Une validation croisee temporelle TimeSeriesSplit a egalement ete appliquee aux modeles ML afin de mesurer leur robustesse sur plusieurs fenetres chronologiques successives sans fuite d'information.
