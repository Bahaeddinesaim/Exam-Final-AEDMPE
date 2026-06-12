"""Generate professional, executable notebooks for the tourism project."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


SETUP_CELL = """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path("..").resolve()
sys.path.append(str(ROOT))

sns.set_theme(style="whitegrid", palette="Set2")
pd.set_option("display.max_columns", 80)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
"""


def md(text: str):
    """Create a Markdown cell."""
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    """Create a code cell."""
    return nbf.v4.new_code_cell(dedent(text).strip())


def write_notebook(path: Path, cells: list) -> None:
    """Write one notebook with deterministic metadata."""
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(notebook, path)


def generate_eda_notebook(path: Path) -> None:
    """Generate a complete EDA notebook."""
    cells = [
        md(
            """
            # 01 - Analyse exploratoire des donnees

            Objectif : comprendre les sources avant de modeliser. Le sujet indique volontairement des incoherences, biais et granularites differentes. Ce notebook ne cherche donc pas a "faire confiance" aux donnees : il les audite.

            Decisions attendues :
            - distinguer faits observes et hypotheses ;
            - identifier les ruptures de granularite pays / destination / review / campagne ;
            - documenter les risques avant nettoyage.
            """
        ),
        code(SETUP_CELL),
        md("## 1. Chargement multi-source et dimensions"),
        code(
            """
            raw = ROOT / "data" / "raw"

            destinations = pd.read_csv(raw / "01_destinations_brut.csv")
            market = pd.read_excel(raw / "02_signaux_marche.xlsx")
            reviews = pd.read_json(raw / "03_reviews_reseaux.json")
            external = pd.read_csv(raw / "04_facteurs_externes.csv", sep=";")
            campaigns = pd.read_json(raw / "06_campaign_history.json")

            datasets = {
                "destinations": destinations,
                "market": market,
                "reviews": reviews,
                "external": external,
                "campaigns": campaigns,
            }

            overview = []
            for name, df in datasets.items():
                overview.append({
                    "dataset": name,
                    "rows": len(df),
                    "columns": df.shape[1],
                    "duplicate_rows": df.duplicated().sum(),
                    "missing_cells": df.isna().sum().sum(),
                })
            pd.DataFrame(overview)
            """
        ),
        md(
            """
            Interpretation : les fichiers n'ont pas la meme granularite. `market` est mensuel au niveau pays, `destinations` est pays-destination, `reviews` est transactionnel, `external` repete des signaux externes, et `campaigns` est un historique court. Une jointure directe sans aggregation creerait des doublons artificiels.
            """
        ),
        code(
            """
            for name, df in datasets.items():
                print(f"\\n{name.upper()} - shape={df.shape}")
                display(pd.DataFrame({"dtype": df.dtypes, "missing": df.isna().sum(), "nunique": df.nunique(dropna=True)}))
                display(df.head(3))
            """
        ),
        md("## 2. Statistiques descriptives et distributions numeriques"),
        code(
            """
            numeric_dest = destinations.select_dtypes(include="number")
            display(numeric_dest.describe().T)

            fig, axes = plt.subplots(2, 2, figsize=(13, 8))
            for ax, col in zip(axes.ravel(), numeric_dest.columns):
                sns.histplot(destinations[col], kde=True, ax=ax)
                ax.set_title(f"Distribution - {col}")
            plt.tight_layout()
            """
        ),
        md(
            """
            Interpretation : `visitors` a une echelle beaucoup plus large que les scores. Toute combinaison de variables doit donc normaliser les grandeurs pour eviter qu'un volume historique ecrase la qualite ou le sentiment.
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            for ax, col in zip(axes, ["attractiveness", "cost", "rating", "visitors"]):
                sns.boxplot(y=destinations[col], ax=ax)
                ax.set_title(f"Boxplot - {col}")
            plt.tight_layout()
            """
        ),
        md("## 3. Correlations et relations utiles"),
        code(
            """
            corr = numeric_dest.corr(numeric_only=True)
            plt.figure(figsize=(7, 5))
            sns.heatmap(corr, annot=True, cmap="vlag", center=0)
            plt.title("Correlation des variables destinations")
            plt.show()
            """
        ),
        code(
            """
            sns.pairplot(destinations[["attractiveness", "cost", "rating", "visitors"]], corner=True)
            plt.suptitle("Pairplot des variables quantitatives", y=1.02)
            plt.show()
            """
        ),
        md(
            """
            Interpretation : les correlations ne doivent pas etre confondues avec des causalites. Elles servent ici a reperer les redondances et les risques de scoring, pas a justifier seules une recommandation business.
            """
        ),
        md("## 4. Variables categorielles, cardinalites et incoherences de libelles"),
        code(
            """
            categorical_checks = []
            for name, df in datasets.items():
                for col in df.select_dtypes(include="object").columns:
                    categorical_checks.append({
                        "dataset": name,
                        "column": col,
                        "nunique": df[col].nunique(dropna=True),
                        "examples": sorted(df[col].dropna().astype(str).unique())[:8],
                    })
            pd.DataFrame(categorical_checks)
            """
        ),
        code(
            """
            country_values = pd.concat([
                destinations["country"].astype(str),
                market["country"].astype(str),
                reviews["country"].astype(str),
                external["country"].astype(str),
                campaigns["country"].astype(str),
            ], ignore_index=True)
            country_values.value_counts().head(25)
            """
        ),
        md(
            """
            Interpretation : les pays apparaissent avec des casses differentes (`France`, `france`, `FRANCE`). Le nettoyage doit harmoniser la cle, mais cette anomalie doit rester documentee car elle prouve que les sources ne partagent pas de referentiel commun.
            """
        ),
        md("## 5. Valeurs manquantes, doublons et anomalies metier"),
        code(
            """
            quality_rows = []
            for name, df in datasets.items():
                for col in df.columns:
                    quality_rows.append({
                        "dataset": name,
                        "column": col,
                        "missing_rate": df[col].isna().mean(),
                        "nunique": df[col].nunique(dropna=True),
                    })
            quality = pd.DataFrame(quality_rows)
            display(quality.sort_values("missing_rate", ascending=False))

            plt.figure(figsize=(11, 5))
            sns.barplot(data=quality, x="column", y="missing_rate", hue="dataset")
            plt.xticks(rotation=45, ha="right")
            plt.title("Taux de valeurs manquantes par colonne")
            plt.tight_layout()
            """
        ),
        code(
            """
            campaign_anomalies = campaigns.assign(
                budget_numeric=pd.to_numeric(campaigns["campaign_budget"], errors="coerce"),
                contradiction=lambda d: ((d["status"].str.upper() == "SUCCESS") & (d["roi"] <= 0))
                | ((d["status"].str.upper() == "FAIL") & (d["roi"] > 0))
            )
            campaign_anomalies[campaign_anomalies["contradiction"] | campaign_anomalies["budget_numeric"].isna()]
            """
        ),
        md(
            """
            Interpretation : un budget `unknown` et des statuts contradictoires avec le ROI ne doivent pas etre corriges arbitrairement. La bonne posture gouvernance est de les flagger, puis de limiter leur poids dans le scoring.
            """
        ),
        md("## 6. Relations entre fichiers et couverture des jointures"),
        code(
            """
            def key_frame(df):
                return df.assign(
                    country_key=df["country"].astype(str).str.strip().str.title(),
                    destination_key=df["destination"].astype(str).str.strip().str.replace("city_", "City_", case=False, regex=False),
                )[["country_key", "destination_key"]].drop_duplicates()

            destination_keys = key_frame(destinations)
            review_keys = key_frame(reviews)
            external_keys = key_frame(external)
            campaign_keys = key_frame(campaigns)

            coverage = pd.DataFrame([
                {"source": "reviews", "matched_destination_keys": len(destination_keys.merge(review_keys)), "source_keys": len(review_keys)},
                {"source": "external", "matched_destination_keys": len(destination_keys.merge(external_keys)), "source_keys": len(external_keys)},
                {"source": "campaigns", "matched_destination_keys": len(destination_keys.merge(campaign_keys)), "source_keys": len(campaign_keys)},
            ])
            coverage["coverage_rate"] = coverage["matched_destination_keys"] / coverage["source_keys"]
            coverage
            """
        ),
        md(
            """
            Conclusion EDA : le projet doit construire une GOLD DATA par aggregation controlee. Les signaux pays ne remplacent pas les signaux destination ; ils enrichissent la recommandation avec une hypothese explicite de potentiel marche.
            """
        ),
    ]
    write_notebook(path, cells)


def generate_cleaning_notebook(path: Path) -> None:
    """Generate a cleaning and governance notebook."""
    cells = [
        md(
            """
            # 02 - Nettoyage industriel et gouvernance

            Ce notebook documente les transformations appliquees. Regle directrice : corriger les problemes de format, mais ne pas inventer une verite metier lorsque la source est contradictoire.
            """
        ),
        code(SETUP_CELL),
        code(
            """
            from src.preprocessing import (
                clean_campaigns, clean_destinations, clean_external, clean_market,
                clean_reviews, load_table, quality_summary
            )

            raw = ROOT / "data" / "raw"
            destinations_raw = load_table(raw / "01_destinations_brut.csv")
            market_raw = load_table(raw / "02_signaux_marche.xlsx")
            reviews_raw = load_table(raw / "03_reviews_reseaux.json")
            external_raw = load_table(raw / "04_facteurs_externes.csv")
            campaigns_raw = load_table(raw / "06_campaign_history.json")
            """
        ),
        md("## 1. Diagnostic avant nettoyage"),
        code(
            """
            raw_datasets = {
                "destinations_raw": destinations_raw,
                "market_raw": market_raw,
                "reviews_raw": reviews_raw,
                "external_raw": external_raw,
                "campaigns_raw": campaigns_raw,
            }
            pd.concat([quality_summary(df, name) for name, df in raw_datasets.items()], ignore_index=True)
            """
        ),
        md(
            """
            Decision : les controles sont faits avant transformation pour garder une trace des anomalies sources. C'est important pour un jury : le nettoyage n'est pas une boite noire.
            """
        ),
        md("## 2. Application des fonctions de nettoyage"),
        code(
            """
            destinations = clean_destinations(destinations_raw)
            market = clean_market(market_raw)
            reviews = clean_reviews(reviews_raw)
            external = clean_external(external_raw)
            campaigns = clean_campaigns(campaigns_raw)

            cleaned_datasets = {
                "destinations": destinations,
                "market": market,
                "reviews": reviews,
                "external": external,
                "campaigns": campaigns,
            }

            for name, df in cleaned_datasets.items():
                print(f"{name}: {df.shape}")
                display(df.head(3))
            """
        ),
        md("## 3. Justification des transformations"),
        code(
            """
            transformations = pd.DataFrame([
                ["country", "strip + title case", "Referentiel pays commun", "Evite pertes de jointure France/france/FRANCE"],
                ["destination_original", "conservation du code City_n", "Lineage gouvernance", "Permet d'auditer le mapping"],
                ["destination", "mapping vers nom de ville", "Restitution metier lisible", "Le mapping est documente dans src/city_mapping.py"],
                ["month", "conversion datetime", "Split temporel fiable", "Les dates invalides sont exclues"],
                ["campaign_budget", "parse numerique", "Calcul ROI/budget", "unknown reste manquant, pas impute arbitrairement"],
                ["status vs roi", "flag contradiction", "Gouvernance", "L'anomalie est exposee au lieu d'etre masquee"],
                ["reviews", "aggregation ulterieure", "Granularite destination", "Evite duplication artificielle"],
            ], columns=["Variable", "Transformation", "Justification metier", "Regle de prudence"])
            transformations
            """
        ),
        md("## 4. Controle avant / apres"),
        code(
            """
            before_after = []
            for raw_name, clean_name in [
                ("destinations_raw", "destinations"),
                ("market_raw", "market"),
                ("reviews_raw", "reviews"),
                ("external_raw", "external"),
                ("campaigns_raw", "campaigns"),
            ]:
                raw_df = raw_datasets[raw_name]
                clean_df = cleaned_datasets[clean_name]
                before_after.append({
                    "dataset": clean_name,
                    "rows_before": len(raw_df),
                    "rows_after": len(clean_df),
                    "duplicates_before": raw_df.duplicated().sum(),
                    "duplicates_after": clean_df.duplicated().sum(),
                    "missing_before": raw_df.isna().sum().sum(),
                    "missing_after": clean_df.isna().sum().sum(),
                })
            pd.DataFrame(before_after)
            """
        ),
        code(
            """
            campaigns[campaigns["campaign_contradiction"] | campaigns["campaign_budget"].isna()]
            """
        ),
        md(
            """
            Interpretation : les valeurs budget non numeriques et contradictions campagne sont conservees sous forme de signaux de qualite. Elles pourront penaliser ou limiter le scoring, mais elles ne doivent pas etre supprimees sans mandat metier.
            """
        ),
        md("## 5. Sauvegarde processed"),
        code(
            """
            processed = ROOT / "data" / "processed"
            processed.mkdir(parents=True, exist_ok=True)
            for name, df in cleaned_datasets.items():
                df.to_csv(processed / f"{name}_clean.csv", index=False)
            sorted(p.name for p in processed.glob("*_clean.csv"))
            """
        ),
    ]
    write_notebook(path, cells)


def generate_gold_notebook(path: Path) -> None:
    """Generate a GOLD DATA notebook."""
    cells = [
        md(
            """
            # 03 - Construction de la GOLD DATA

            Objectif : produire une table analytique unique a granularite pays-destination, utilisable par la BI, le scoring metier et le dashboard.
            """
        ),
        code(SETUP_CELL),
        code(
            """
            from src.feature_engineering import (
                aggregate_campaigns, aggregate_external, aggregate_reviews,
                build_country_market_features, build_gold_data
            )

            processed = ROOT / "data" / "processed"
            destinations = pd.read_csv(processed / "destinations_clean.csv")
            market = pd.read_csv(processed / "market_clean.csv", parse_dates=["month"])
            reviews = pd.read_csv(processed / "reviews_clean.csv")
            external = pd.read_csv(processed / "external_clean.csv")
            campaigns = pd.read_csv(processed / "campaigns_clean.csv")
            """
        ),
        md("## 1. Aggregations par granularite"),
        code(
            """
            review_agg = aggregate_reviews(reviews)
            external_agg = aggregate_external(external)
            campaign_agg = aggregate_campaigns(campaigns)
            market_features = build_country_market_features(market)

            display(review_agg.head())
            display(external_agg.head())
            display(campaign_agg.head())
            display(market_features.head())
            """
        ),
        md(
            """
            Decision : les reviews, facteurs externes et campagnes sont agreges avant jointure. Cela evite de multiplier les lignes destination par le nombre d'avis ou de signaux externes.
            """
        ),
        md("## 2. Variables derivees et valeur metier"),
        code(
            """
            variable_rationale = pd.DataFrame([
                ["demand_growth", "Mesure la dynamique recente du marche pays", "Permet de pousser les pays en acceleration"],
                ["tourism_score", "Combine attractivite, visiteurs, demande et volume d'avis", "Mesure le potentiel global"],
                ["quality_score", "Combine rating source et score reviews", "Evite de recommander une destination mal percue"],
                ["sentiment_score", "Convertit le sentiment social en signal numerique", "Capte la tonalite reputationale"],
                ["attractiveness_cost_ratio", "Attractivite par unite de cout", "Integre l'efficience budgetaire"],
                ["weather_penalty", "Risque meteo externe", "Evite les destinations defavorables"],
                ["campaign_efficiency", "ROI rapporte au budget connu", "Valorise les apprentissages marketing"],
                ["marketing_priority", "Score final pondere", "Classement actionnable pour le comite marketing"],
            ], columns=["Variable", "Definition", "Interet business"])
            variable_rationale
            """
        ),
        md("## 3. Generation GOLD DATA"),
        code(
            """
            gold = build_gold_data(destinations, market, reviews, external, campaigns)
            display(gold.head())
            display(gold.describe().T)
            """
        ),
        code(
            """
            selected = ["forecasted_demand", "quality_score", "sentiment_score", "weather_penalty", "campaign_efficiency", "marketing_priority"]
            plt.figure(figsize=(9, 6))
            sns.heatmap(gold[selected].corr(), annot=True, cmap="vlag", center=0)
            plt.title("Correlation des variables de scoring")
            plt.show()
            """
        ),
        code(
            """
            plt.figure(figsize=(12, 6))
            sns.scatterplot(
                data=gold,
                x="quality_score",
                y="forecasted_demand",
                hue="country",
                size="marketing_priority",
                sizes=(30, 250),
            )
            plt.title("Qualite vs demande prevue, taille = priorite marketing")
            plt.tight_layout()
            """
        ),
        md(
            """
            Interpretation : une destination prioritaire n'est pas seulement une destination populaire. Elle doit combiner demande, qualite, sentiment, efficience cout/campagne et risque meteo acceptable.
            """
        ),
        md("## 4. Sauvegarde et controles finaux"),
        code(
            """
            gold_path = ROOT / "data" / "gold" / "gold_tourism_data.csv"
            gold_path.parent.mkdir(parents=True, exist_ok=True)
            gold.to_csv(gold_path, index=False)

            pd.DataFrame({
                "rows": [len(gold)],
                "columns": [gold.shape[1]],
                "missing_cells": [gold.isna().sum().sum()],
                "unique_countries": [gold["country"].nunique()],
                "unique_destinations": [gold["destination"].nunique()],
            })
            """
        ),
    ]
    write_notebook(path, cells)


def generate_model_notebook(path: Path) -> None:
    """Generate the professional Part F notebook: AI model choice and training."""
    cells = [
        md(
            """
            # Partie F - Choix du modele IA et entrainement

            ## Objectif metier

            L'objectif est de prevoir la demande touristique future par pays pour le prochain mois ou trimestre, afin d'aider l'equipe marketing a prioriser les pays et destinations a promouvoir.

            La variable cible retenue est `target_next_period`, construite comme la demande du mois suivant par pays:

            `target_next_period = demand_index shifted by -1 period per country`

            Cette cible est une approximation gouvernee de la demande touristique future, car le fichier temporel disponible porte un `demand_index` mensuel par pays. Les volumes `visitors` sont disponibles au niveau pays-destination mais ne disposent pas d'un historique mensuel; ils sont donc utilises comme variables explicatives contextuelles et non comme cible temporelle directe.
            """
        ),
        code(SETUP_CELL),
        code(
            """
            import numpy as np
            import plotly.express as px
            import plotly.graph_objects as go
            from sklearn.model_selection import TimeSeriesSplit

            from src.modeling import (
                prepare_modeling_dataset,
                temporal_train_test_split,
                train_full_model_suite,
            )

            processed = ROOT / "data" / "processed"
            market = pd.read_csv(processed / "market_clean.csv", parse_dates=["month"])
            gold = pd.read_csv(ROOT / "data" / "gold" / "gold_tourism_data.csv")
            """
        ),
        md(
            """
            ## 1. Type de probleme IA

            Il s'agit d'une **regression supervisee** appliquee a une **prevision temporelle** sur des **donnees structurees**.

            Le modele apprend a partir des observations historiques pour predire une demande future. Le caractere temporel impose un split chronologique afin d'eviter toute fuite de donnees.

            Ce n'est pas une classification: la cible est une variable quantitative continue.
            """
        ),
        code(
            """
            model_components = pd.DataFrame([
                ["Donnees d'entree X", "Pays, periode, demande courante, lags, moyenne mobile, visiteurs, attractivite, cout, note, sentiment, meteo, budget campagne"],
                ["Variable cible y", "target_next_period = demand_index du mois suivant par pays"],
                ["Algorithmes", "Persistence, moyenne mobile, SARIMAX, Prophet, Random Forest + lag features, XGBoost ou Gradient Boosting + lag features"],
                ["Fonction de perte", "Erreur de regression; optimisation indirecte via MAE/RMSE/MAPE"],
                ["Split temporel", "80 % premieres periodes en train, 20 % dernieres periodes en test"],
                ["Baselines", "Naive persistence et moving average 3 periodes"],
                ["Metriques", "MAE, RMSE, MAPE, R2"],
                ["Interpretation metier", "Mesurer la fiabilite d'une prediction utilisee pour prioriser les budgets marketing"],
                ["Limites", "Historique court, destination-level visitors non temporel, signaux sociaux potentiellement biaises"],
            ], columns=["Composante", "Choix et justification"])
            model_components
            """
        ),
        md("## 2. Preparation des donnees temporelles"),
        code(
            """
            modeling_df = prepare_modeling_dataset(gold, market)
            display(modeling_df.head())
            display(modeling_df[[
                "country", "period", "visitors", "target_next_period",
                "attractiveness", "cost", "rating", "sentiment_score",
                "weather_score", "campaign_budget", "market_signal"
            ]].head())
            modeling_df.isna().sum().to_frame("missing_values").T
            """
        ),
        md(
            """
            Interpretation : la colonne `period` existe dans les signaux marche et est convertie en datetime. La table de modelisation est triee par pays et periode. Les lignes sans `target_next_period` sont supprimees uniquement parce qu'elles correspondent a la derniere periode observee, pour laquelle le futur n'est pas encore connu.
            """
        ),
        md("## 2.1 Feature engineering temporel sans fuite de donnees"),
        code(
            """
            time_features = [
                "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30",
                "rolling_mean_3", "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
                "rolling_std_7", "rolling_std_30", "rolling_min_30", "rolling_max_30",
                "month", "quarter", "year",
            ]
            modeling_df[["country", "period", "market_signal", "target_next_period"] + time_features].head(12)
            """
        ),
        md(
            """
            Toutes les variables glissantes utilisent `shift(1)` avant le calcul des moyennes, ecarts-types, minimums et maximums. Cela garantit qu'une observation future ou courante ne fuit pas dans les variables explicatives servant a predire la periode suivante.
            """
        ),
        md("## 2.2 Validation croisee temporelle avec TimeSeriesSplit"),
        code(
            """
            tscv = TimeSeriesSplit(n_splits=5, test_size=None)
            ordered_for_cv = modeling_df.sort_values(["period", "country"]).reset_index(drop=True)

            fold_rows = []
            for fold, (train_idx, test_idx) in enumerate(tscv.split(ordered_for_cv), start=1):
                fold_rows.append({
                    "Fold": fold,
                    "Train start": ordered_for_cv.loc[train_idx, "period"].min(),
                    "Train end": ordered_for_cv.loc[train_idx, "period"].max(),
                    "Test start": ordered_for_cv.loc[test_idx, "period"].min(),
                    "Test end": ordered_for_cv.loc[test_idx, "period"].max(),
                    "Train rows": len(train_idx),
                    "Test rows": len(test_idx),
                })
            folds = pd.DataFrame(fold_rows)
            display(folds)

            fig = go.Figure()
            for _, row in folds.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["Train start"], row["Train end"]],
                    y=[f"Fold {row['Fold']}", f"Fold {row['Fold']}"],
                    mode="lines",
                    line=dict(color="#2563EB", width=10),
                    name="Train" if row["Fold"] == 1 else None,
                    showlegend=row["Fold"] == 1,
                ))
                fig.add_trace(go.Scatter(
                    x=[row["Test start"], row["Test end"]],
                    y=[f"Fold {row['Fold']}", f"Fold {row['Fold']}"],
                    mode="lines",
                    line=dict(color="#F59E0B", width=10),
                    name="Test" if row["Fold"] == 1 else None,
                    showlegend=row["Fold"] == 1,
                ))
            fig.update_layout(template="plotly_white", title="Visualisation des folds TimeSeriesSplit", xaxis_title="Periode", yaxis_title="")
            fig.show()
            """
        ),
        md(
            """
            TimeSeriesSplit respecte l'ordre chronologique des observations et permet de valider le modele sur plusieurs fenetres temporelles successives sans fuite d'information.

            Cette validation est appliquee aux modeles de Machine Learning avec lag features : Random Forest et XGBoost/Gradient Boosting. Les modeles statistiques comme SARIMAX sont generalement evalues sur un decoupage temporel chronologique ou via une approche rolling forecast plutot qu'avec une validation croisee standard. Prophet est conserve sur une evaluation hold-out; si la librairie est disponible, ses outils de diagnostics peuvent etre ajoutes via `prophet.diagnostics.cross_validation`.
            """
        ),
        md("## 3. Split temporel"),
        code(
            """
            train, test = temporal_train_test_split(modeling_df, test_ratio=0.2)
            split_summary = pd.DataFrame({
                "train_start": [train["period"].min()],
                "train_end": [train["period"].max()],
                "test_start": [test["period"].min()],
                "test_end": [test["period"].max()],
                "train_rows": [len(train)],
                "test_rows": [len(test)],
                "countries_train": [train["country"].nunique()],
                "countries_test": [test["country"].nunique()],
            })
            split_summary
            """
        ),
        md(
            """
            Le split temporel respecte l'ordre chronologique des observations et simule une situation reelle ou le futur n'est pas connu au moment de l'entrainement. Un `train_test_split` aleatoire serait une fuite de donnees et surestimerait la performance.
            """
        ),
        md("## 4. Baselines et modeles avances"),
        code(
            """
            suite = train_full_model_suite(
                gold=gold,
                market=market,
                reports_dir=ROOT / "reports",
                gold_dir=ROOT / "data" / "gold",
                models_dir=ROOT / "models",
            )

            performance = suite["performance"]
            predictions = suite["predictions"]
            feature_importance = suite["feature_importance"]
            cv_results = suite["cv_results"]
            cv_summary = suite["cv_summary"]
            limitations = suite["limitations"]

            performance
            """
        ),
        md(
            """
            Modeles entraines :

            - **Baseline Persistence** : prevision(t+1) = valeur observee a t.
            - **Baseline Moving Average** : prevision(t+1) = moyenne mobile des 3 dernieres periodes.
            - **SARIMAX** : modele statistique de serie temporelle entraine par pays avec variables exogenes et `try/except`.
            - **Prophet** : modele additif de serie temporelle par pays, avec regressors si disponible et fallback si la librairie n'est pas installee.
            - **RandomForestRegressor + lag features** : pipeline sklearn avec `ColumnTransformer`, `OneHotEncoder`, imputation, `StandardScaler`, lags et rolling features.
            - **XGBoost ou Gradient Boosting + lag features** : XGBoost est utilise si installe, sinon fallback propre vers `GradientBoostingRegressor`, toujours avec variables temporelles.

            Les metriques suivies sont `MAE`, `RMSE`, `MAPE` et `R2`. Le R2 mesure la part de variance expliquee par le modele; il peut etre negatif si le modele fait moins bien qu'une prediction moyenne naive.
            """
        ),
        md("## 5.1 Resultats TimeSeriesSplit par fold"),
        code(
            """
            display(cv_results)
            display(cv_summary)
            """
        ),
        md("## 5.2 Visualisations de robustesse TimeSeriesSplit"),
        code(
            """
            fig = px.box(
                cv_results,
                x="Model",
                y="RMSE",
                color="Model",
                color_discrete_sequence=["#2563EB", "#14B8A6"],
                title="Boxplot RMSE par fold",
            )
            fig.update_layout(template="plotly_white", xaxis_title="", yaxis_title="RMSE", showlegend=False)
            fig.show()

            fig = px.box(
                cv_results,
                x="Model",
                y="MAE",
                color="Model",
                color_discrete_sequence=["#2563EB", "#14B8A6"],
                title="Boxplot MAE par fold",
            )
            fig.update_layout(template="plotly_white", xaxis_title="", yaxis_title="MAE", showlegend=False)
            fig.show()

            cv_long = cv_results.melt(
                id_vars=["Model", "Fold"],
                value_vars=["MAE", "RMSE", "MAPE"],
                var_name="Metric",
                value_name="Value",
            )
            fig = px.line(
                cv_long,
                x="Fold",
                y="Value",
                color="Model",
                line_dash="Metric",
                markers=True,
                title="Evolution des performances selon les folds",
            )
            fig.update_layout(template="plotly_white", xaxis_title="Fold chronologique", yaxis_title="Erreur")
            fig.show()
            """
        ),
        md(
            """
            Interpretation : la validation croisee temporelle mesure la stabilite des modeles ML sur plusieurs fenetres successives. Une faible dispersion des RMSE/MAE indique une meilleure robustesse temporelle. Cette verification complete le hold-out final et reduit le risque de selectionner un modele performant uniquement sur une seule fenetre de test.
            """
        ),
        md("## 5. Tableau comparatif des performances"),
        code(
            """
            display(performance)
            if not limitations.empty:
                display(limitations)
            """
        ),
        md("## 6. Graphique de comparaison MAE / RMSE / MAPE"),
        code(
            """
            fig = px.bar(
                performance.dropna(subset=["RMSE"]).sort_values("RMSE", ascending=False),
                x="RMSE",
                y="Model",
                color="Approach type",
                orientation="h",
                color_discrete_sequence=["#2563EB", "#14B8A6", "#F59E0B", "#64748B"],
                title="Comparaison des modeles par RMSE",
            )
            fig.update_layout(template="plotly_white", xaxis_title="RMSE", yaxis_title="")
            fig.show()

            fig = px.bar(
                performance.dropna(subset=["MAPE"]).sort_values("MAPE", ascending=False),
                x="MAPE",
                y="Model",
                color="Approach type",
                orientation="h",
                color_discrete_sequence=["#2563EB", "#14B8A6", "#F59E0B", "#64748B"],
                title="Comparaison des modeles par MAPE",
            )
            fig.update_layout(template="plotly_white", xaxis_title="MAPE (%)", yaxis_title="")
            fig.show()
            """
        ),
        md(
            """
            Interpretation : les baselines sont le point de comparaison minimal. Un modele avance doit les battre pour justifier sa complexite. Le RMSE penalise fortement les grosses erreurs, ce qui est pertinent pour une decision budgetaire.
            """
        ),
        md("## 7. Reel vs predit pour le meilleur modele"),
        code(
            """
            model_column_map = {
                "Baseline Persistence": "baseline_persistence",
                "Baseline Moving Average": "baseline_moving_average",
                "Random Forest + lag features": "random_forest",
                "XGBoost + lag features": "gradient_boosting",
                "Gradient Boosting + lag features": "gradient_boosting",
                "SARIMAX": "sarimax",
                "Prophet": "prophet",
            }
            best_model = performance[performance["Status"] == "OK"].iloc[0]["Model"]
            best_column = model_column_map[best_model]
            sample_country = predictions["country"].iloc[0]
            country_plot = predictions[predictions["country"] == sample_country].sort_values("period")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=country_plot["period"], y=country_plot["actual"], mode="lines+markers", name="Reel", line=dict(color="#1E293B", width=3)))
            fig.add_trace(go.Scatter(x=country_plot["period"], y=country_plot[best_column], mode="lines+markers", name=best_model, line=dict(color="#2563EB", width=3)))
            fig.update_layout(template="plotly_white", title=f"Reel vs predit - {sample_country}", xaxis_title="Periode", yaxis_title="Demande future")
            fig.show()
            """
        ),
        md("Interpretation : cette courbe permet de verifier si le modele suit correctement la dynamique temporelle sur les periodes de test, et pas uniquement une moyenne globale."),
        md("## 8. Scatter plot y_true vs y_pred et residus"),
        code(
            """
            eval_df = predictions[["actual", best_column]].dropna().rename(columns={best_column: "prediction"})
            eval_df["residual"] = eval_df["actual"] - eval_df["prediction"]

            fig = px.scatter(
                eval_df,
                x="actual",
                y="prediction",
                color="residual",
                color_continuous_scale="RdBu",
                title=f"Actual vs predicted - {best_model}",
            )
            fig.add_trace(go.Scatter(x=eval_df["actual"], y=eval_df["actual"], mode="lines", name="Ideal", line=dict(color="#1E293B", dash="dash")))
            fig.update_layout(template="plotly_white", xaxis_title="Valeur reelle", yaxis_title="Prediction")
            fig.show()

            fig = px.histogram(
                eval_df,
                x="residual",
                nbins=20,
                color_discrete_sequence=["#2563EB"],
                title=f"Distribution des residus - {best_model}",
            )
            fig.update_layout(template="plotly_white", xaxis_title="Erreur reelle - predite", yaxis_title="Frequence")
            fig.show()
            """
        ),
        md("Interpretation : un bon modele doit produire des points proches de la diagonale et des residus centres autour de zero. Des residus asymetriques indiqueraient un biais de sur- ou sous-prediction."),
        md("## 9. Feature importance Random Forest"),
        code(
            """
            display(feature_importance.head(15))
            fig = px.bar(
                feature_importance.head(15).sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                color_discrete_sequence=["#14B8A6"],
                title="Top feature importance - Random Forest",
            )
            fig.update_layout(template="plotly_white", xaxis_title="Importance", yaxis_title="")
            fig.show()
            """
        ),
        md("Interpretation : l'importance des variables aide a expliquer les drivers de prediction. Elle doit etre lue comme une indication de contribution statistique, pas comme une causalite metier."),
        md("## 10. Choix du meilleur modele et lecture critique"),
        code(
            """
            best_row = performance[performance["Status"] == "OK"].iloc[0]
            generated_conclusion = (
                f"Le modele retenu est {best_row['Model']} car il obtient la meilleure performance "
                f"sur l'ensemble de test avec un RMSE de {best_row['RMSE']:.2f}, un MAPE de {best_row['MAPE']:.2f}% "
                f"et un R2 de {best_row['R2']:.2f}. "
                "Il sera utilise pour alimenter la GOLD DATA et le moteur de recommandation, sous reserve de validation metier."
            )
            print(generated_conclusion)
            """
        ),
        md(
            """
            Lecture critique :

            - La meilleure performance statistique ne suffit pas : il faut verifier la robustesse dans le temps.
            - Random Forest et Gradient Boosting ne sont acceptables ici que parce qu'ils utilisent des lags, rolling features et variables calendaires.
            - SARIMAX et Prophet sont methodologiquement pertinents, mais limites par la longueur d'historique disponible.
            - L'interpretabilite doit etre renforcee par feature importance, analyse des residus et validation metier.
            - Les donnees sociales, meteo et campagnes contiennent des biais et incoherences documentes dans la gouvernance.

            Formulation de synthese : les modeles Random Forest et XGBoost ont ete adaptes au forecasting via la creation de variables de retard et de moyennes glissantes. Ils ne sont donc pas utilises comme simples modeles tabulaires, mais comme approches de prevision supervisee temporelle.
            """
        ),
        md("## 11. Sorties generees"),
        code(
            """
            outputs = [
                ROOT / "reports" / "model_performance.csv",
                ROOT / "reports" / "forecast_predictions.csv",
                ROOT / "reports" / "time_series_cv_results.csv",
                ROOT / "reports" / "time_series_cv_summary.csv",
                ROOT / "data" / "gold" / "forecast_predictions.csv",
                ROOT / "data" / "gold" / "gold_tourism_data_with_forecast.csv",
                ROOT / "reports" / "best_model_summary.md",
                ROOT / "models" / "best_model.pkl",
                ROOT / "reports" / "feature_importance.csv",
                ROOT / "reports" / "time_series_model_limits.csv",
            ]
            pd.DataFrame({"output": [str(path) for path in outputs], "exists": [path.exists() for path in outputs]})
            """
        ),
        md(
            """
            Recommandations d'amelioration :

            - Collecter un historique mensuel de visiteurs par destination, pas seulement par pays.
            - Ajouter des variables calendaires: vacances scolaires, evenements, saisons, prix moyens.
            - Monitorer le drift des donnees et la degradation des metriques.
            - Mettre en place un backtesting multi-fenetres.
            - Valider les recommandations par tests A/B marketing.
            """
        ),
    ]
    write_notebook(path, cells)


def generate_all_notebooks(notebooks_dir: Path) -> None:
    """Generate the four final notebooks."""
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    generate_eda_notebook(notebooks_dir / "01_EDA.ipynb")
    generate_cleaning_notebook(notebooks_dir / "02_Cleaning.ipynb")
    generate_gold_notebook(notebooks_dir / "03_GoldData.ipynb")
    generate_model_notebook(notebooks_dir / "04_Model.ipynb")
