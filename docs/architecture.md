# Architecture du pipeline

## Organisation

```text
config/                     paramètres versionnés
  universes/                constituants, devises, provenance et FX
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
prices + fx (Parquet) + fraîcheur (SQLite) + audit qualité
    ↓
features/assets + features/regime
    ↓
HMM recalibré walk-forward → régimes filtrés et confirmés
    ↓
allocation contrainte + MC pondéré + overlay quotidien
    ↓
target weights + décisions + risques (même run_id)
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

Snapshots features sont écrits dans répertoire temporaire, puis remplacés par
renommage atomique. Clés `(date)` et `(ticker, date)` restent uniques.

### `core/universe.py`

Valide fichier d'univers, couverture devises et séries FX. Expose fingerprint
SHA-256 permettant d'identifier exactement définition utilisée.

### `pipeline/data_quality.py`

Nettoie prix, convertit vers devise de référence sans taux futur et produit audits
prix/features sous `data/quality/`.

### `pipeline/regimes.py`

Recalibre scaler et HMM sur passé strict, filtre prochaine période, transforme états
bruts en `calm/choppy/stress`, applique confirmation 20/60 et écrit paramètres JSON.
Voir [Régimes walk-forward](regimes.md).

### `pipeline/backtest.py`

Le backtest part de 100 % cash. Il transforme les poids cibles datés en ordres exécutés, applique coûts et turnover, calcule les rendements puis fait dériver les poids jusqu'au prochain ordre.

### `pipeline/optimize.py`, `pipeline/mc.py` et `models/allocation.py`

L'optimiseur suit les positions effectives avec la comptabilité partagée de
`core/portfolio.py`. Il calibre MC sur le passé, résout les contraintes puis
applique l'overlay quotidien. Les risques du portefeuille détenu, du candidat
et de la cible sont distincts. La commande `mc --run-id ...` rejoue les poids
sauvegardés ; ses résultats ne sont jamais réinjectés dans les décisions passées.
Voir [Risque et allocation](risk-allocation.md).

## Artefacts d'un backtest

| Dataset | Contenu |
|---|---|
| `weights` | Poids cibles à la date de décision |
| `decisions` | Historique quotidien des expositions, contrôles de risque et statuts solveur |
| `mc` | Risques pondérés effectifs/candidats/cibles par horizon et distribution |
| `risk_weights` | Poids exacts et univers de calibration utilisés par MC |
| `mc_replay` | Réévaluation séparée des poids sauvegardés |
| `trades` | Deltas exécutés, sens et notionnel |
| `positions` | Poids et valeur de chaque actif après clôture, cash inclus |
| `backtests` | Valeur, rendement net, coût, turnover et exposition quotidienne |
| SQLite `backtests` | Résumé de performance par `run_id` |

Les fichiers historiques déjà présents restent lisibles. Les nouveaux backtests ajoutent des colonnes sans supprimer les anciennes.
