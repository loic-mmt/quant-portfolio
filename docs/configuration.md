# Configuration

## Configuration commune

`config/base.yaml` contient :

| Champ | Rôle |
|---|---|
| `data_dir` | Racine des données, relative au dépôt si le chemin n'est pas absolu |
| `start_date` | Première date demandée lors d'une ingestion complète |
| `end_date` | Borne optionnelle de téléchargement |
| `log_level` | Niveau de log par défaut |
| `random_seed` | Seed commun destiné aux modèles stochastiques |
| `universe_file` | Fichier YAML versionné définissant constituants, devises, FX et provenance |

Fichier d'univers possède `universe_id`, version entière et fingerprint SHA-256.
Tickers et devises sont normalisés en majuscules. Doublons, liste vide ou taux FX
manquant provoquent erreur avant lancement. Voir [Données et features](data-and-features.md).

## Configuration du backtest

`config/backtest.yaml` contient :

| Champ | Valeur par défaut | Rôle |
|---|---:|---|
| `rebal_freq` | `W` | Fréquence attendue des décisions de poids |
| `transaction_bps` | `1.0` | Frais proportionnels au turnover |
| `slippage_bps` | `2.0` | Slippage proportionnel au turnover |
| `turnover_cap` | `null` | Turnover maximal par exécution |
| `exposure_change_cap` | `null` | Variation maximale d'exposition par exécution ; combinée avec la limite de l'overlay |
| `initial_capital` | `100000` | Valeur initiale du portefeuille |
| `cash_rate_annual` | `0.0` | Rendement annualisé de la poche cash |
| `execution_lag_days` | `1` | Nombre de jours de marché après la décision ; minimum obligatoire : 1 |
| `return_type` | `simple` | Convention des rendements d'entrée : `simple` ou `log` |
| `missing_return_policy` | `cash` | Traitement d'un rendement manquant : `cash`, `zero` ou `error` |
| `start_date` | `null` | Borne de début optionnelle du backtest |
| `end_date` | `null` | Borne de fin optionnelle du backtest |

La convention canonique du projet est le rendement simple. Quand `return_type: log` est choisi, le simulateur reconvertit chaque rendement en rendement simple avant la comptabilité du portefeuille.

## Identifiants de run

Un identifiant peut être fourni explicitement :

```bash
quant-portfolio optimize --run-id ablation-no-mc-001
quant-portfolio backtest --run-id ablation-no-mc-001
```

Sans identifiant, un ID UTC est créé. Un même ID doit relier les poids, trades, positions, résultats et rapports d'une expérience.

## Données existantes

Les modes d'écriture Parquet sont :

- `overwrite_or_ignore` : fusion incrémentale idempotente par défaut ;
- `overwrite` : snapshot complet ;
- `delete_matching` : comportement Arrow explicite de remplacement ;
- `error` : échoue si la destination existe.

Pour features, écriture finale remplace snapshot complet de façon atomique après fusion.
Utiliser `overwrite` lorsque recalcul historique complet est intentionnel.

## Configuration des régimes

`config/regimes.yaml` contient :

| Champ | Rôle |
|---|---|
| `n_states` | Nombre d'états HMM bruts, minimum 3 |
| `random_seed` | Initialisation déterministe |
| `covariance_type` | `diag`, `full`, `tied` ou `spherical` |
| `n_iter`, `tol`, `min_covar` | Convergence et stabilité numérique |
| `min_train_size` | Historique minimal avant première prédiction |
| `train_window` | Taille fenêtre roulante ; `null` pour expansive |
| `recalibration_frequency` | `D`, `20B`, `W`, `M`, `Q`, etc. |
| `confidence_threshold` | Confiance instantanée minimale pour transition |
| `confirmation` | Fenêtres et seuils courts/longs 20/60 |
| `regime_features` | Colonnes ordonnées utilisées par scaler/HMM |

Changer seed, fenêtre, features ou fréquence définit expérience différente. Chaque
calibration conserve copie exacte configuration dans artefact JSON.

## Risque, solveur et overlay

`config/mc.yaml` définit simulations, horizons 5/20, fenêtre, observations minimales,
seed, distribution principale, distributions de comparaison, degrés de liberté
Student-t et seuils de perte/drawdown. `sparse_regime_policy` choisit le repli
historique explicite `pooled` ou le mode strict `error`.

`config/optimize.yaml` définit bornes individuelles, fréquence de recomposition,
cash, volatilité cible, hysteresis, vitesse, turnover, pénalités du solveur et
politiques par régime. Les options `use_regimes`, `use_mc` et `use_overlay`
permettent des variantes séparées. Les contraintes sectorielles exigent un champ
`sector` pour chaque actif de l'univers versionné.

Le détail des unités, contraintes dures, objectifs souples et cas infaisables est
documenté dans [Risque et allocation](risk-allocation.md). Les frais de l'objectif
proviennent de la même configuration que ceux du backtest.

Les nouveaux runs d'optimisation sont immuables. Un nouveau calcul nécessite un
nouvel ID, même avec `--existing-data-behavior overwrite`. Leur configuration
d'exécution sauvegardée est réutilisée par le backtest. La commande MC exige un ID
existant et écrit un replay séparé ; elle ne modifie pas les risques du run initial.

## Configuration du rapport

`config/report.yaml` contient :

| Champ | Défaut | Rôle |
|---|---:|---|
| `trading_days` | `252` | Annualisation vol/ratios |
| `extreme_loss_threshold` | `0.02` | Seuil de fréquence des pertes journalières extrêmes |
| `capacity_adv_window` | `20` | Fenêtre ADV en séances |
| `capacity_participation_rate` | `0.10` | Fraction maximale théorique d'ADV |
| `capacity_quantile` | `0.05` | Quantile prudent des limites par trade |

Rapport utilise config/seed sauvegardés par optimisation, pas fichiers live MC/
optimiseur. Rapport peut être régénéré avec nouveau `report.yaml`; manifeste garde
valeurs exactes. Voir [Évaluation et rapport](evaluation-and-reporting.md).
