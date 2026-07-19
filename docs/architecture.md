# Architecture du pipeline

## Organisation

```text
config/                     paramètres versionnés
data/
  _meta.db                  fraîcheur et résumés SQLite
  parquet/                  séries et artefacts lourds
docs/                       conventions et guides
sql/                        schéma SQLite et vues
src/
  quant_portfolio/
    core/                   configuration, IDs, SQL et stockage
    models/                 HMM, GARCH et covariance
    pipeline/               étapes exécutables
    main.py                 CLI
  cpp/                      accélération optionnelle
tests/                      tests déterministes
```

## Flux de données

```text
Yahoo Finance
    ↓
prices (Parquet) + fraîcheur (SQLite)
    ↓
features/assets + features/regime
    ↓
regimes
    ↓
Monte-Carlo conditionnel
    ↓
target weights
    ↓ décision à t, exécution à t+1
backtests + trades + positions
    ↓
report
```

## Responsabilités

### `core/settings.py`

Charge `config/base.yaml`, résout les chemins par rapport à la racine du dépôt et configure les logs. Les modules ne doivent pas dépendre du répertoire courant du shell.

### `core/ids.py`

Valide les identifiants utilisés dans les partitions et crée des IDs UTC triables. Les chaînes contenant des séparateurs de chemin sont rejetées.

### `core/storage.py`

Lit et écrit les datasets Parquet partitionnés. Un run d'optimisation ou de backtest est toujours sélectionné par son `run_id`. En l'absence d'ID explicite, le dernier ID disponible est choisi.

### `pipeline/backtest.py`

Le backtest part de 100 % cash. Il transforme les poids cibles datés en ordres exécutés, applique coûts et turnover, calcule les rendements puis fait dériver les poids jusqu'au prochain ordre.

## Artefacts d'un backtest

| Dataset | Contenu |
|---|---|
| `weights` | Poids cibles à la date de décision |
| `trades` | Deltas exécutés, sens et notionnel |
| `positions` | Poids et valeur de chaque actif après clôture, cash inclus |
| `backtests` | Valeur, rendement net, coût, turnover et exposition quotidienne |
| SQLite `backtests` | Résumé de performance par `run_id` |

Les fichiers historiques déjà présents restent lisibles. Les nouveaux backtests ajoutent des colonnes sans supprimer les anciennes.
