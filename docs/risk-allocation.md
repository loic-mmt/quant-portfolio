# Risque, allocation contrainte et overlay quotidien

Le jalon 4 remplace les prototypes Monte-Carlo et inverse-variance par une chaîne
causale. `optimize` calcule désormais le risque **pendant** la construction des
poids. Il ne lit jamais un dataset MC calculé sur une autre allocation ou un autre run.

## Chronologie et poids effectifs

À chaque séance `t` :

1. exécuter les décisions arrivées à échéance, avec le plafond de turnover du backtest ;
2. appliquer le rendement simple observé à `t`, puis faire dériver les positions et le cash ;
3. estimer les modèles avec les rendements strictement antérieurs à `t` ;
4. utiliser le régime filtré disponible à `t` et mesurer le risque des positions réelles après clôture ;
5. calculer une nouvelle cible, exécutée à `t + execution_lag_days` séances.

Les nouveaux poids ne capturent donc jamais le rendement de leur jour de décision.
La comptabilité de dérive est partagée entre stratégie et backtest dans
`core/portfolio.py`. Le shadow portfolio utilise les mêmes dates, cash, rendement
manquant, délai et turnover d'exécution. Les frais étant prélevés au prorata du
portefeuille, ils modifient la valeur mais pas les poids relatifs. Les gouverneurs
de turnover et de vitesse sont aussi réappliqués contre les positions réelles à
l'exécution : les rendements intermédiaires d'un délai supérieur à une séance ne
peuvent donc pas les contourner. Les limites effectives combinent celles de
l'optimiseur et du backtest ; elles sont sauvegardées avec le run.

La composition est recalculée à la première séance observée de chaque semaine,
quinzaine ou mois (`D`, `W`, `2W`, `M`). L'ancrage des quinzaines est fixe. Cette
convention n'a pas besoin de connaître le dernier jour d'un historique encore
incomplet. Une transition de régime, un changement d'éligibilité ou un dépassement
des bornes de composition provoque aussi une nouvelle optimisation. L'overlay reste
quotidien ; hors recomposition, il conserve les proportions dérivées des actifs.

## Monte-Carlo conditionnel

### Calibration

Les rendements simples sont convertis en log-rendements. À chaque décision, les
dernières `window` observations **antérieures** portant le régime courant fournissent
la moyenne et la covariance. Les horizons ne servent jamais à choisir des données
futures. `models/covariance.py` applique Ledoit-Wolf ou un shrinkage diagonal et
conditionne les valeurs propres par `eps`.

Les actifs pondérés ne sont jamais supprimés silencieusement. En présence de NaN,
la covariance paire à paire reste shrinkée et n'est acceptée que si chaque paire
dispose d'assez d'observations. Son nom devient `shrink_diag_pairwise` dans les
artefacts. Un historique insuffisant entraîne une décision `history_hold` : aucun
ordre nouveau, portefeuille existant conservé et dérivant normalement, motif journalisé.

`sparse_regime_policy: pooled` autorise explicitement un historique tous régimes
confondus lorsque le régime courant est trop court. Il reste strictement passé et
est marqué `pooled_sparse_regime` dans chaque métrique, avec un log. Choisir `error`
interdit ce repli : la stratégie suspend alors les nouvelles décisions tant que
la calibration conditionnelle est impossible. Ce choix est une hypothèse de
recherche importante, pas une estimation conditionnelle équivalente.

### Simulation pondérée

Pour chaque scénario, la valeur relative à la NAV initiale vaut :

```text
V_h / V_0 = Σ_i w_i × exp(Σ_{d=1..h} log_return_i,d)
            + cash_weight × (1 + cash_return_daily)^h
```

Les positions sont buy-and-hold durant le scénario, donc leurs poids dérivent.
Il ne s'agit ni d'une somme non pondérée des actifs, ni d'un portefeuille rééquilibré
chaque jour. Les transactions futures et changements de régime durant le scénario
ne sont pas simulés.

Trois rôles sont séparés, avec les poids exacts sérialisés :

- `effective` : positions réellement détenues après clôture, avant la décision ;
- `candidate` : allocation candidate avant réduction d'exposition ;
- `target` : allocation finale après overlay et contraintes.

Le stress cut utilise `candidate`, pour mesurer le risque d'un réinvestissement
même lorsque le portefeuille actuel est principalement cash. Le risque réellement
détenu reste disponible sous `effective`. Tous utilisent les mêmes scénarios à une
date donnée, permettant une comparaison directe.

Les distributions disponibles sont `gaussian` et `student_t`. Pour Student-t, une
échelle chi-deux commune aux actifs de chaque scénario/jour préserve la dépendance
des queues. Le facteur `sqrt((df-2)/chi2(df))` conserve la même covariance que la
gaussienne ; `student_df` doit dépasser 2. Les distributions de
`compare_distributions` produisent des diagnostics supplémentaires mais seul `dist`
pilote les décisions. Il s'agit d'une comparaison de modèles, pas d'une sélection
optimisée sur leurs performances futures.

Chaque générateur est local, initialisé par un hash stable de seed/date/distribution.
L'ordre d'exécution, l'état de `numpy.random` et l'ajout de données futures ne
modifient pas les scénarios passés.

### Métriques et conventions

Les horizons par défaut sont 5 et 20 séances. Toutes les métriques sont des fractions
de NAV, pas des montants en euros :

- `var_01`, `var_05` : opposé des quantiles de PnL à 1 % et 5 %, borné à zéro ;
- `cvar_01`, `cvar_05` : perte moyenne dans la queue correspondante, bornée à zéro ;
- `pnl_q01`, `pnl_q05`, `pnl_q50`, `pnl_q95`, `pnl_q99` : quantiles **signés** ;
- `expected_pnl` : PnL moyen ;
- `probability_loss` : fréquence de `PnL < -loss_threshold` ;
- `probability_drawdown` : fréquence du drawdown maximal intra-horizon supérieur
  à `drawdown_threshold`, pic initial de NAV inclus.

Les seuils, horizons, poids, seed, taille des simulations, dates de calibration,
état et estimateur sont enregistrés avec les métriques. À 2 000 simulations, la
queue 1 % ne contient qu'environ 20 scénarios : l'incertitude d'estimation reste
importante. Les queues Student-t des log-rendements peuvent aussi produire des
scénarios extrêmes ; aucun clipping financier silencieux n'est appliqué.

## Optimisation contrainte

Le solveur CVXPY/CLARABEL minimise :

```text
252 × wᵀΣw
+ (turnover_penalty + frais_effectifs) × turnover
+ distance_penalty × ||w - poids_effectifs||²
+ exposure_penalty × (Σw - exposition_demandée)²
+ target_vol_penalty × max(vol_annualisée(w) - target_vol, 0)²
```

`frais_effectifs = (transaction_bps + slippage_bps) / 10000` provient de la
configuration du backtest. Le turnover inclut le cash :
`0.5 × (Σ|Δw| + |Δcash|)`. Les coûts sont estimés dans l'objectif puis facturés
uniquement à l'exécution par le backtest.

Contraintes dures :

- long-only ; borne `min_weight` sur **chaque actif éligible**, zéro sur les autres ;
- plafond individuel `max_weight × regime_cap_scale[régime]` ;
- exposition totale entre `min_exposure` et `max_exposure`, ou exactement 1 sans cash ;
- plafonds sectoriels absolus sur la NAV, si configurés ;
- turnover de décision inférieur au minimum de `turnover_governor` et du plafond d'exécution ;
- variation absolue d'exposition inférieure à `max_exposure_change` avec overlay/cash.

Il n'y a pas de sélection mixte-entière « zéro ou poids minimal ». Un minimum positif
force donc la détention de tous les actifs éligibles. La valeur par défaut zéro
permet une réduction profonde de l'exposition. Un minimum positif trop élevé,
un plafond sectoriel trop bas ou des limites de vitesse incompatibles rendent le
problème infaisable ; aucune renormalisation ne contourne ces contraintes. Sans
cash, un portefeuille initialement liquide exige un turnover initial autorisant
l'investissement complet (`turnover_governor: null`, plafond backtest compatible).

`sector_caps` reste vide par défaut, car l'univers existant n'a pas de classification
sectorielle. Pour activer cette option, ajouter `sector` à **chaque** actif du fichier
d'univers versionné, puis un mapping tel que `sector_caps: {Technology: 0.25}`.
Les métadonnées manquantes ou secteurs inconnus sont des erreurs.

Le statut du solveur et toutes les contraintes sont contrôlés numériquement avec
une tolérance de `1e-7`, avant persistance. Les poussières numériques sont traitées
sans renormaliser la solution. `solver_fallback: error` arrête le run par défaut.
`hold` conserve les poids précédents uniquement s'ils satisfont encore **toutes**
les contraintes ; le statut `fallback_hold` inclut la cause de l'échec.

## Overlay quotidien

La volatilité annualisée prévisionnelle du candidat est
`sqrt(252 × w_candidateᵀΣw_candidate)`. Le multiplicateur
`target_vol / forecast_vol` est borné par `min_multiplier` et `max_multiplier`.
L'exposition demandée combine ce multiplicateur avec :

- `regime_exposure` : 1 / 0,7 / 0,4 pour calm / choppy / stress par défaut ;
- un multiplicateur MC : minimum de 1, limite VaR / VaR 5 %, limite CVaR / CVaR 5 %.

Les écarts inférieurs à `hysteresis` conservent l'exposition observée ; hors
recomposition, cette exposition est fixée pour éviter des ajustements résiduels.
L'objectif d'exposition et le ciblage de volatilité sont **souples** : bornes de
poids, turnover et vitesse restent prioritaires. `governed` indique une différence
supérieure à 0,001 entre exposition demandée et obtenue. Un stress cut peut donc
prendre plusieurs séances. Les poids de marché peuvent dépasser une borne après
leur dérive ; les contraintes certifiées concernent les nouvelles **cibles**, pas
une garantie permanente sur les positions après rendement ou exécution plafonnée.

`decisions` journalise réduction/restauration/maintien, exposition réelle/demandée/
cible, cash, volatilités, statut MC, composition, turnover, frais estimés et statut
solveur. `use_regimes: false` retire conditionnement et politiques de régime ;
`use_mc: false` retire simulation et stress cut MC ; `use_overlay: false` conserve
les recompositions périodiques sans overlay quotidien. Chaque variante doit avoir
son propre `run_id`.

## Commandes, artefacts et reproductibilité

```bash
quant-portfolio features
quant-portfolio regimes
quant-portfolio optimize --run-id risk-v1
quant-portfolio backtest --run-id risk-v1
# Optionnel : relecture des poids sauvegardés avec config/mc.yaml courant.
quant-portfolio mc --run-id risk-v1
```

`run-all` appelle features → regimes → optimize (MC intégré) → backtest. Aucun MC
préliminaire ne doit être exécuté avant l'optimisation.

Sous `data/parquet/`, les datasets `weights`, `decisions`, `mc` et `risk_weights`
sont isolés par `run_id`, puis année. Les risques rejoués vont dans `mc_replay`
pour ne jamais écraser les risques ayant piloté les décisions. `risk_weights`
conserve aussi l'ordre/univers de calibration, même pour des poids nuls.
Les paramètres de la dernière réévaluation sont enregistrés dans
`data/runs/<run_id>/mc_replay.json`.

`data/runs/<run_id>/config.json` sauvegarde les configurations, le fingerprint
d'univers et celui des historiques utilisés. La borne de fin du backtest est figée
sur les données du run. Backtest et replay vérifient l'historique ; le backtest
reprend la configuration d'exécution sauvegardée. Les runs d'optimisation sont
immuables : réutiliser un ID existant échoue, quel que soit le mode d'écriture.

Les anciens artefacts MC, non pondérés et calibrés globalement, ne sont pas
compatibles avec ce contrat. Ils ne sont ni consommés ni supprimés automatiquement.
Recalculer régimes et créer un nouveau run avant toute interprétation de résultats.

Les tests déterministes couvrent covariance, solution analytique minimum-variance,
contraintes, queues Student-t, réduction d'exposition, invariance au futur,
ablations, comptabilité décalée et replay Parquet identique. Ils ne valident pas
une performance financière ni la pertinence empirique des hyperparamètres.
