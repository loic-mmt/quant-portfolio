# Évaluation, ablations et rapport autonome

Le jalon 5 compare six variantes dans un protocole commun, persiste toutes les
données nécessaires au recalcul et produit un fichier HTML autonome.

## Une période comparable

Le rapport charge un run d'optimisation/backtest terminé. La première décision de
la stratégie complète définit `decision_date`; sa première exécution strictement
postérieure définit le début d'évaluation.

- **Calibration/warmup** : début des rendements → première décision. Cette période
  sert aux estimateurs ; aucun score de performance « in-sample » n'est publié.
- **Walk-forward out-of-sample** : première exécution → borne figée du run. Tous
  tableaux, classements et conclusions utilisent uniquement cette période.

Chaque variante voit exactement les mêmes dates, colonnes de rendement, univers,
capital, taux cash, politique de données manquantes, frais, slippage et délai
d'exécution. Les données postérieures à `input_end` sauvegardé sont ignorées.

Les baselines equal-weight désactivent seulement les gouverneurs de turnover et de
vitesse : elles doivent réaliser leur allocation standard dès l'exécution. Elles
conservent les mêmes frais et slippage. Cette différence est inscrite dans le
manifeste et ne doit pas être confondue avec une réduction de coût.

## Variantes

| ID | Définition |
|---|---|
| `equal_weight_buy_hold` | Allocation égale unique, puis dérive buy-and-hold |
| `equal_weight_rebalanced` | Allocation égale à la fréquence du run |
| `minimum_variance_no_regimes` | Minimum variance, sans régimes, MC ni overlay |
| `strategy_no_overlay` | Régimes et MC actifs, overlay quotidien désactivé |
| `strategy_no_mc` | Régimes et vol targeting actifs, stress cut MC désactivé |
| `strategy_full` | Poids et exécutions réellement sauvegardés par le run |

Les trois ablations optimisées sont recalculées avec le seed, la covariance,
l'univers, les secteurs et les configurations sauvegardés. La stratégie complète
n'est pas recalculée : le rapport lit ses artefacts d'exécution réels.

## Métriques

### Performance et risque

- CAGR sur jours calendaires ;
- volatilité, Sharpe et Sortino annualisés sur 252 séances par défaut ;
- downside deviation : RMS annualisée des excès de rendement négatifs ;
- max drawdown, dates pic/creux, récupération et jours calendaires pic → reprise ;
- VaR/CVaR réalisées à 1 % et 5 % sur rendements journaliers nets ;
- fréquence de perte sous `-extreme_loss_threshold`.

VaR et CVaR sont des pertes positives. Les métriques réalisées décrivent
l'échantillon du backtest ; elles ne sont pas les scénarios MC prospectifs.

### Praticabilité et concentration

- turnover total et annualisé ;
- coûts cumulés en devise de référence et fraction du capital initial ;
- HHI moyen/maximal des actifs risqués normalisés par leur exposition ;
- HHI total incluant le cash ;
- poids actif maximal ;
- exposition cash moyenne/maximale.

Capacité approximative par trade :

```text
NAV maximale = participation_ADV × ADV_moyen / |delta_weight|
capacité stratégie = quantile configuré des NAV maximales
```

`ADV_moyen` utilise `adj_close × volume` dans la devise de référence sur
`capacity_adv_window`. Le nombre d'observations ADV est conservé avec chaque trade.
Cette métrique ignore impact non linéaire, spread variable, profondeur intraday,
shortfall et suspensions : diagnostic de premier ordre, jamais garantie d'exécution.

### Attribution par régime

Le régime confirmé disponible à chaque date est aligné causalement. Pour chaque
variante/régime : jours, rendement composé, rendement/volatilité annualisés,
Sharpe à taux zéro, downside deviation, turnover, coûts et exposition.

La somme des log-rendements par régime reconstitue exactement le log-rendement
total. La `variance_share` répartit la somme des écarts quadratiques à la moyenne
globale. Ce partage de risque est descriptif, pas une décomposition factorielle.

## Artefacts recalculables

Chaque chemin est isolé sous `run_id=<id>` :

| Dataset | Contenu |
|---|---|
| `evaluations` | NAV, rendements, drawdown calculable, turnover, coûts, cash/exposition, sample |
| `evaluation_positions` | Poids/valeurs quotidiens de chaque actif et cash |
| `evaluation_trades` | Ordres, deltas, notionnels, ADV et limite de capacité |
| `evaluation_metrics` | Métriques de synthèse des six variantes |
| `regime_attribution` | Performance, risque, turnover et coûts par régime |

`data/runs/<id>/report.json` conserve config du rapport, overrides exacts des
variantes, période, univers, frais, chemins et provenance. `config.json` conserve
config/seed de stratégie, fingerprint d'univers/données, dates, commit Git,
état dirty et hash du code/config réellement présent. Un ancien run sans provenance
J5 reste lisible mais affiche cette donnée comme indisponible.

Toutes les métriques HTML se recalculent depuis datasets d'évaluation et
`config.json`; HTML n'est pas source de vérité.

## Commandes

Run existant :

```bash
quant-portfolio report --run-id experiment-001
```

Sortie par défaut : `artifacts/reports/experiment-001.html`. Un chemin personnalisé
doit finir par `.html` ou `.htm` :

```bash
quant-portfolio report --run-id experiment-001 --output artifacts/my-report.html
```

Pipeline complet en une commande :

```bash
quant-portfolio run-all --run-id experiment-001 --with-report
```

Le HTML embarque CSS et cinq graphiques SVG : equity, drawdowns, exposition,
probabilités filtrées des régimes et rendements par régime. Aucun CDN, JavaScript,
image ou police externe.

## Interprétation

Les conclusions automatiques indiquent uniquement meilleure CAGR observée, plus
faible drawdown observé et écarts de la stratégie complète à equal-weight hold.
Elles n'établissent ni causalité, ni significativité, ni performance future.
Survivorship bias, sélection d'univers, faible nombre d'événements extrêmes et
incertitude des coûts/capacité restent applicables.
