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
| `tickers` | Univers fixe et versionné de l'expérience |

Les tickers sont normalisés en majuscules. Les doublons et listes vides provoquent une erreur avant le lancement du pipeline.

## Configuration du backtest

`config/backtest.yaml` contient :

| Champ | Valeur par défaut | Rôle |
|---|---:|---|
| `rebal_freq` | `W` | Fréquence attendue des décisions de poids |
| `transaction_bps` | `1.0` | Frais proportionnels au turnover |
| `slippage_bps` | `2.0` | Slippage proportionnel au turnover |
| `turnover_cap` | `null` | Turnover maximal par exécution |
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

- `overwrite_or_ignore` : mode incrémental par défaut ;
- `overwrite` : remplace les partitions correspondantes ;
- `delete_matching` : comportement Arrow explicite de remplacement ;
- `error` : échoue si la destination existe.

Utiliser `overwrite` uniquement lorsque le recalcul de la période concernée est intentionnel.
