# Contrat temporel du backtest

Ce document définit la convention anti-look-ahead utilisée par le projet.

> Cette convention est maintenant appliquée au moteur de backtest et à la fenêtre de l'optimiseur. Le HMM et la calibration Monte-Carlo historiques doivent encore être convertis en walk-forward dans les jalons 3 et 4 ; les anciens artefacts de régime et de backtest ne deviennent donc pas automatiquement out-of-sample.

## Trois dates distinctes

### `information_date`

Dernière date dont les données sont complètement observées. Une estimation réalisée à `t` ne peut utiliser que des rendements d'indice strictement antérieurs à `t`.

### `decision_date`

Date attachée aux poids cibles. L'optimiseur produit une décision après avoir observé l'information autorisée.

### `execution_date`

Date à laquelle le portefeuille adopte effectivement les nouveaux poids. Avec `execution_lag_days: 1`, il s'agit du premier jour de marché strictement postérieur à la décision.

```text
clôture t-1       décision t          exécution t+1       clôture t+1
information ────────┼────────────────────┼────────────────────┤
                    poids cibles         coûts/trades         rendement détenu
```

Le rendement de la `decision_date` n'est jamais capturé par les poids décidés ce jour-là.

## Ordre comptable quotidien

Pour chaque date de marché :

1. mémoriser la valeur de clôture précédente ;
2. exécuter l'éventuelle décision arrivée à échéance ;
3. limiter le turnover si un plafond est configuré ;
4. déduire frais et slippage ;
5. appliquer les rendements des actifs et du cash ;
6. calculer le rendement net quotidien ;
7. faire dériver les poids selon les performances relatives ;
8. enregistrer résultats, trades et positions de clôture.

## Turnover et cash

Le cash est un actif explicite. Le turnover one-way inclut la jambe cash :

```text
turnover = 0,5 × (Σ |Δ poids actifs| + |Δ poids cash|)
```

Passer de 100 % cash à 60 % investi produit donc un turnover de 60 %, pas de 30 %.

Les coûts sont calculés ainsi :

```text
coût = turnover × (transaction_bps + slippage_bps) / 10 000
```

L'allocation initiale est une transaction et supporte les mêmes coûts que les rebalancements suivants.

## Dérive des poids

Entre deux transactions, les poids ne restent pas fixes. Pour un actif `i` :

```text
w_i,t+1 = w_i,t × (1 + r_i,t+1) / (1 + r_portefeuille,t+1)
```

Le cash dérive de la même manière avec son rendement quotidien.

## Rendements manquants

La politique est configurable :

- `cash` : le poids concerné reçoit le rendement du cash pour cette date ;
- `zero` : rendement nul ;
- `error` : arrêt immédiat si l'actif manquant est détenu.

`error` est le mode le plus strict pour les validations. `cash` est le mode opérationnel par défaut et doit être complété plus tard par une véritable gestion des actifs non négociables.

## Tests de non-régression

Les tests imposent notamment que :

- un rendement du jour de décision ne soit pas capturé ;
- ajouter ou modifier des données futures ne change pas le passé ;
- un portefeuille mono-actif réconcilie un buy-and-hold ;
- les positions et le cash somment à la valeur totale ;
- l'allocation initiale soit facturée.
