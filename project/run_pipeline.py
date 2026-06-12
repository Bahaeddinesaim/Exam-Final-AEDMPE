"""End-to-end pipeline for the Master tourism forecasting project."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

sys.path.append(str(Path(__file__).parent))

from src.feature_engineering import build_gold_data
from src.modeling import train_and_evaluate, train_full_model_suite
from src.notebook_factory import generate_all_notebooks
from src.preprocessing import (
    clean_campaigns,
    clean_destinations,
    clean_external,
    clean_market,
    clean_reviews,
    configure_logging,
    load_table,
    quality_summary,
)
from src.recommendation import rank_destinations


ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
GOLD = ROOT / "data" / "gold"
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
NOTEBOOKS = ROOT / "notebooks"
MODELS = ROOT / "models"


def save_pdf(markdown_text: str, path: Path) -> None:
    """Create a simple professional PDF from the generated management report."""
    styles = getSampleStyleSheet()
    story = []
    for line in markdown_text.splitlines():
        if line.startswith("#"):
            story.append(Paragraph(line.replace("#", "").strip(), styles["Heading1"]))
        elif line.strip():
            story.append(Paragraph(line.strip().replace("&", "&amp;"), styles["BodyText"]))
        story.append(Spacer(1, 6))
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def build_data_dictionary(gold: pd.DataFrame) -> pd.DataFrame:
    """Document key variables with governance ownership and quality rules."""
    definitions = [
        ("country", "Pays de promotion touristique", "string", "Title case", "N/A", "01/02/03/04/06", "Non nul, harmonise", "Marketing Data Owner", "Variantes de casse observees", "France"),
        ("destination_original", "Code destination anonymise conserve pour lineage", "string", "City_n", "N/A", "01/03/04/06", "Conserve avant mapping", "Data Governance Owner", "Ne pas utiliser en restitution executive", "City_3"),
        ("destination", "Destination candidate fiabilisee a promouvoir", "string", "Nom ville", "N/A", "01/03/04/06 + mapping gouverne", "Non nul avec pays", "Tourism Product Owner", "Mapping a maintenir comme referentiel", "Annecy"),
        ("attractiveness", "Indice d'attractivite fourni", "float", "0-10", "score", "01", "Valeur plausible 0-10", "Data Steward", "Source non auditee", "9.38"),
        ("cost", "Cout moyen de promotion/sejour", "float", "numeric", "EUR proxy", "01", "Strictement positif", "Finance Owner", "Unite implicite", "287"),
        ("rating", "Note moyenne destination", "float", "0-5", "score", "01", "Controle bornes 0-5", "CX Owner", "Peut etre biaisee", "4.22"),
        ("visitors", "Volume historique de visiteurs", "float", "integer", "visiteurs", "01", "Non negatif", "BI Owner", "Pas de date fournie", "9376241"),
        ("demand_index", "Indice mensuel de demande pays", "float", "numeric", "index", "02", "Non nul par pays/mois", "Market Intelligence", "Granularite pays", "101.2"),
        ("demand_growth", "Croissance recente de la demande", "float", "ratio", "%", "02 derive", "Borne clippee en scoring", "Data Science", "Sensible au dernier mois", "0.04"),
        ("quality_score", "Score qualite combine note et avis", "float", "0-1", "score", "01/03 derive", "Normalise", "Data Science", "Avis sociaux non representatifs", "0.82"),
        ("sentiment_score", "Sentiment moyen des reviews", "float", "-1 a 1", "score", "03 derive", "Mapping documente", "Social Media Owner", "Ironie/langue non detectees", "0.18"),
        ("weather_penalty", "Penalite meteo externe", "float", "0-0.7", "penalite", "04 derive", "Mapping good/average/bad", "Risk Owner", "Donnee agregee grossiere", "0.25"),
        ("campaign_efficiency", "ROI par euro investi", "float", "ratio", "ROI/budget", "06 derive", "Budget numerique requis", "Marketing Ops", "Budget unknown exclu", "2.4"),
        ("forecasted_demand", "Demande future estimee destination", "float", "numeric", "visiteurs proxy", "Gold derive", "Pas de fuite future", "Data Science", "Forecast proxy destination", "1240000"),
        ("marketing_priority", "Score final de priorisation", "float", "0-1", "score", "Gold derive", "Contraintes documentees", "Steering Committee", "Depend des ponderations", "0.74"),
        ("allocated_budget", "Budget recommande alloue", "float", "numeric", "EUR proxy", "Recommendation", "Somme controlee", "Finance Owner", "A valider en comite", "18000"),
    ]
    columns = [
        "Nom technique",
        "Definition metier",
        "Type",
        "Format",
        "Unite",
        "Source",
        "Regles qualite",
        "Data Owner",
        "Points de vigilance",
        "Exemple",
    ]
    return pd.DataFrame(definitions, columns=columns)


def create_notebook(path: Path, title: str, sections: list[str]) -> None:
    """Generate a readable notebook scaffold with executable imports."""
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\nProjet tourisme - version consultant.")]
    cells.append(
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "ROOT = Path('..')\n"
            "gold = pd.read_csv(ROOT / 'data/gold/gold_tourism_data.csv')\n"
            "gold.head()"
        )
    )
    for section in sections:
        cells.append(nbf.v4.new_markdown_cell(section))
    nb["cells"] = cells
    nbf.write(nb, path)


def build_reports(quality: pd.DataFrame, metrics: pd.DataFrame, recommendations: pd.DataFrame) -> str:
    """Return the management report as Markdown."""
    return f"""# Rapport Management - Forecast & Promotion Touristique

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
{metrics.to_markdown(index=False)}

## Data Quality
Les controles couvrent valeurs manquantes, types, cardinalites et contradictions. Extrait:
{quality.head(12).to_markdown(index=False)}

## Recommandations
Top recommandations:
{recommendations.head(10).to_markdown(index=False)}

## Limites, biais et risques
Les destinations sont anonymisees, la cible officielle est un texte non tabulaire, les avis sociaux sont sujets a biais de representation, et la meteo est categorielle. Les scores doivent donc etre utilises comme aide a la decision, pas comme verite automatique.

## Pistes futures
Connecter des donnees calendaires reelles, ajouter elasticite prix, saisonnalite par destination, tests A/B campagne, et monitoring MLOps des performances.
"""


def main() -> None:
    configure_logging(ROOT / "logs" / "pipeline.log")
    for path in [PROCESSED, GOLD, DOCS, REPORTS, NOTEBOOKS, MODELS]:
        path.mkdir(parents=True, exist_ok=True)

    destinations = clean_destinations(load_table(RAW / "01_destinations_brut.csv"))
    market = clean_market(load_table(RAW / "02_signaux_marche.xlsx"))
    reviews = clean_reviews(load_table(RAW / "03_reviews_reseaux.json"))
    external = clean_external(load_table(RAW / "04_facteurs_externes.csv"))
    campaigns = clean_campaigns(load_table(RAW / "06_campaign_history.json"))

    datasets = {
        "destinations": destinations,
        "market": market,
        "reviews": reviews,
        "external": external,
        "campaigns": campaigns,
    }
    for name, frame in datasets.items():
        frame.to_csv(PROCESSED / f"{name}_clean.csv", index=False)

    quality = pd.concat([quality_summary(frame, name) for name, frame in datasets.items()], ignore_index=True)
    quality.to_csv(REPORTS / "data_quality_summary.csv", index=False)

    gold = build_gold_data(destinations, market, reviews, external, campaigns)
    gold.to_csv(GOLD / "gold_tourism_data.csv", index=False)

    metrics, predictions = train_and_evaluate(market)
    metrics.to_csv(REPORTS / "model_metrics.csv", index=False)
    predictions.to_csv(REPORTS / "forecast_predictions.csv", index=False)
    full_model_suite = train_full_model_suite(gold, market, REPORTS, GOLD, MODELS)

    recommendations = rank_destinations(gold)
    recommendations.to_csv(GOLD / "business_recommendations.csv", index=False)

    dictionary = build_data_dictionary(gold)
    dictionary.to_csv(DOCS / "data_dictionary.csv", index=False)
    (DOCS / "data_dictionary.md").write_text(
        "# Dictionnaire de donnees\n\n" + dictionary.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )

    report = build_reports(quality, full_model_suite["performance"], recommendations)
    (REPORTS / "management_report.md").write_text(report, encoding="utf-8")
    save_pdf(report, REPORTS / "management_report.pdf")

    create_notebook(
        NOTEBOOKS / "01_EDA.ipynb",
        "01 - Analyse exploratoire",
        [
            "## Lecture critique\nLes distributions, valeurs extremes, doublons et cardinalites doivent etre interpretes avant toute decision.",
            "## Graphiques attendus\nHistogrammes, boxplots, heatmap de correlation, analyse des variables categorielles et relations entre fichiers.",
        ],
    )
    create_notebook(
        NOTEBOOKS / "02_Cleaning.ipynb",
        "02 - Nettoyage industriel",
        [
            "## Decisions\nChaque transformation conserve une justification: harmonisation pays, parsing numerique, gestion NA et signalement des contradictions.",
            "## Principe de gouvernance\nUne anomalie n'est pas automatiquement corrigee lorsqu'elle peut porter un sens metier.",
        ],
    )
    create_notebook(
        NOTEBOOKS / "03_GoldData.ipynb",
        "03 - GOLD DATA",
        [
            "## Variables derivees\nLes scores combinent demande, qualite, sentiment, cout, meteo et campagne.",
            "## Usage metier\nLa table finale sert de socle commun BI, IA et recommandation.",
        ],
    )
    create_notebook(
        NOTEBOOKS / "04_Model.ipynb",
        "04 - Modelisation",
        [
            "## Split temporel\nLa validation se fait sur les derniers mois pour simuler une vraie prevision.",
            "## Baselines\nNaive persistence et moving average encadrent la valeur ajoutee des modeles ML.",
        ],
    )
    generate_all_notebooks(NOTEBOOKS)

    print("Pipeline complete")
    print(f"Gold rows: {len(gold)}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
