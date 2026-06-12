# Tourism Forecast AI - Gouvernance des Donnees & IA

Projet de Master visant a prevoir la demande touristique future par pays et a recommander les destinations a promouvoir sous contraintes metier.

## Architecture

```text
project/
├── data/
│   ├── raw/          # sources conservees intactes
│   ├── processed/    # donnees nettoyees
│   └── gold/         # GOLD DATA et recommandations
├── notebooks/        # EDA, cleaning, gold data, modelisation
├── src/              # modules industrialises
├── dashboard/        # application Streamlit
├── reports/          # KPI, qualite, modeles, rapport management
├── docs/             # dictionnaire de donnees
└── run_pipeline.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Execution

```bash
python run_pipeline.py
streamlit run dashboard/app.py
```

## Workflow

1. Chargement multi-format avec detection CSV, Excel et JSON.
2. Profilage qualite: types, dimensions, colonnes, NA, cardinalites et anomalies.
3. Nettoyage gouverne: pays, destinations, textes, numeriques, dates et contradictions.
4. Construction GOLD DATA pays-destination.
5. Prevision mensuelle avec split temporel, deux baselines et deux modeles ML.
6. Recommandation metier sous contraintes: budget, meteo, qualite, sentiment et historique campagne.
7. Restitution via dashboard Streamlit et rapport management.

## Variables derivees cles

- `demand_growth`: dynamique recente du signal marche pays.
- `tourism_score`: potentiel touristique combine.
- `quality_score`: qualite percue par notes et avis.
- `sentiment_score`: tonalite moyenne des reviews sociales.
- `weather_penalty`: penalite de risque externe.
- `campaign_efficiency`: ROI rapporte au budget connu.
- `forecasted_demand`: projection destination derivee du signal pays.
- `marketing_priority`: score final de priorisation.

## Resultats attendus

- `data/gold/gold_tourism_data.csv`
- `data/gold/business_recommendations.csv`
- `reports/model_metrics.csv`
- `reports/data_quality_summary.csv`
- `reports/management_report.md`
- `reports/management_report.pdf`
- `docs/data_dictionary.md`
- `docs/data_dictionary.csv`

## Points de vigilance

La cible fournie est un texte de consigne et non une table analytique. Les donnees contiennent volontairement des incoherences: casse pays, budget inconnu, ROI contradictoire avec statut campagne, granularites pays vs destination et avis sociaux potentiellement biaises. Le projet documente ces limites et transforme les donnees en aide a la decision plutot qu'en automatisation aveugle.
