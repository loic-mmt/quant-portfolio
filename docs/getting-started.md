# Installation et commandes

## Prérequis

- Python 3.10 ou supérieur ;
- un compilateur C++17 uniquement pour l'extension C++ optionnelle ;
- une connexion réseau uniquement pour l'ingestion Yahoo Finance.

## Installation reproductible

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

Pour développer et exécuter les tests :

```bash
python -m pip install -e '.[dev]'
pytest
```

Le fichier `requirements.lock` verrouille les dépendances directes de l'environnement Python 3.10 de référence. `pyproject.toml` reste la source des contraintes de compatibilité du package.

## Interface en ligne de commande

```bash
quant-portfolio --help
```

Commandes disponibles :

```bash
quant-portfolio ingest
quant-portfolio features
quant-portfolio regimes
quant-portfolio mc
quant-portfolio optimize --run-id experiment-001
quant-portfolio backtest --run-id experiment-001
quant-portfolio report --run-id experiment-001
```

`ingest` télécharge actifs et séries FX définis par fichier d'univers. `features`
refuse calcul multi-devise si série FX requise manque. Audits sont écrits sous
`data/quality/`.

Le pipeline complet peut être lancé avec :

```bash
quant-portfolio run-all --run-id experiment-001
```

Options utiles :

- `--skip-ingest` sur `run-all` réutilise les prix locaux ;
- `--existing-data-behavior overwrite` force la réécriture d'un dataset ;
- `--verbose` active les diagnostics détaillés ;
- `--with-report` ajoute le résumé Markdown à la fin de `run-all`.

## Exécution sans installation du script console

Depuis la racine du dépôt :

```bash
PYTHONPATH=src python -m quant_portfolio --help
```

L'installation éditable reste recommandée, car elle rend les imports identiques en développement, dans les tests et en production.

## Vérifications rapides

```bash
python -m compileall -q src/quant_portfolio
pytest tests/test_cli.py tests/test_ids.py tests/test_backtest.py tests/test_features.py
```

Les tests de backtest utilisent des données synthétiques locales et ne téléchargent rien.
