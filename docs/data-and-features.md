# Données et features

## Univers reproductible

`config/base.yaml` référence un fichier sous `config/universes/`. Ce fichier est
versionné dans Git et contient : identifiant, version, date du snapshot, source,
biais de survivance connu, devise de référence, constituants, devise locale,
multiplicateur de prix et séries FX.

Le loader calcule un fingerprint SHA-256 du fichier. Identifiant, version et
fingerprint sont copiés dans les rapports qualité. Modifier un constituant, une
devise ou une série FX crée donc une définition d'expérience différente.

L'univers `global-equities-legacy-2024` provient de l'ancienne liste
`tickers_2024`. Sa méthodologie de sélection et son historique point-in-time ne
sont pas disponibles. Il contient donc un biais de sélection et potentiellement
un biais de survivance. Ce biais est documenté, pas corrigé rétroactivement.

## Devise et FX

Devise de référence : EUR. Les cours locaux sont convertis avant tout calcul de
feature. Pour chaque devise étrangère, configuration fournit une série Yahoo de
type `EURXXX=X`, cotée en unités de devise locale par EUR :

```text
prix_EUR(t) = prix_local(t) × multiplicateur / FX_local_par_EUR(t)
```

Le taux utilisé est dernier taux disponible à `t`, jamais un taux futur, avec
tolérance maximale de sept jours calendaires. Une ligne sans taux compatible
reste manquante et est exclue des calculs. Listings londoniens cotés en pence
utilisent multiplicateur `0.01` avant conversion.

## Qualité des prix

Ingestion conserve données ajustées, OHLCV, dividendes et splits quand Yahoo les
fournit. Audit produit `data/quality/ingest_latest.json`. Pipeline features ajoute
même audit dans `data/quality/features_latest.json`.

Contrôles :

- doublons `(ticker, date)` ; dernier enregistrement conservé ;
- dates invalides ;
- prix nuls ou négatifs ; lignes rejetées ;
- couverture et dates absentes par ticker sur calendrier observé de l'univers ;
- ticker absent, partiel, stale ou configuré comme délisté ;
- dividendes et splits déclarés ;
- changement supérieur à 20 % du ratio `close / adj_close` sans action déclarée ;
- couverture FX et nombre de lignes non convertibles.

Une absence ne devient jamais rendement nul. Ticker absent ou délisté reste
explicite dans rapport. Calendrier observé mélange places boursières : compteur de
dates manquantes est diagnostic de couverture, pas calendrier officiel de marché.

## Construction du proxy marché

Proxy marché utilise rendement simple equal-weight quotidien des actifs disponibles
en EUR. Série commence à 100 et compose ces rendements. Elle est donc invariante à
l'unité nominale de chaque cours. Breadth divise uniquement par actifs ayant une
observation valide à date donnée ; NaN ne compte ni comme hausse ni comme baisse.

## Catalogue des features

Fréquence de toutes features : quotidienne, close-to-close. Rendements de momentum
sont log. Volatilités et vol-of-vol restent quotidiennes, non annualisées. Skew,
kurtosis, corrélations, beta, distances et breadth sont sans unité.

| Features | Fenêtre / convention |
|---|---|
| `mom_mkt_20/60/252`, `mom_i_20/60/252` | somme log-returns sur 20/60/252 observations |
| `trend_slope_60`, `trend_slope_i_60` | pente OLS quotidienne du log-prix sur 60 observations |
| `dist_mkt_ma_50/200`, `dist_ma_i_50/200` | prix / moyenne mobile 50/200 - 1 |
| `vol_mkt_20/60`, `vol_i_20/60` | écart-type log-return 20/60, non annualisé |
| `ewma_vol_mkt`, `ewma_vol_i` | écart-type EWMA, alpha 0,06, non annualisé |
| `vol_of_vol_20`, `vol_of_vol_i_20` | écart-type 20 jours de volatilité 20 jours |
| `dd_mkt_60`, `dd_i_60` | drawdown depuis maximum roulant 60 |
| `mdd_mkt_252`, `mdd_i_252` | maximum roulant 252 du drawdown calculé sur 252 ; historique causal maximal 503 observations |
| `skew_mkt_60`, `skew_i_60` | skewness log-return sur 60 |
| `kurt_mkt_60`, `kurt_i_60` | excess kurtosis log-return sur 60 |
| `avg_corr_20/60` | moyenne des corrélations par paire disponibles sur 20/60 |
| `d_avg_corr` | `avg_corr_20 - avg_corr_60` |
| `disp_20/60` | moyenne 20/60 de dispersion cross-sectionnelle quotidienne |
| `breadth_up` | fraction des rendements disponibles strictement positifs |
| `breadth_up_20` | moyenne 20 de `breadth_up` |
| `breadth_ma50` | fraction des actifs valides au-dessus de moyenne mobile 50 |
| `downside_vol_i_60` | écart-type 60 des seuls log-returns négatifs, non annualisé |
| `beta_i_60` | covariance actif/marché divisée par variance marché sur 60 |
| `idio_vol_i_60` | écart-type 60 du résidu actif moins beta × marché, non annualisé |
| `adv_i_20` | volume moyen sur 20 |
| `dollar_volume_i_20` | prix EUR × volume, moyenne 20 |

## Incrémental et idempotence

Features longues incluent EWMA et double fenêtre du maximum drawdown. Pour garantir
égalité exacte, run incrémental recharge historique causal complet, recalcule snapshot,
ne fusionne que dates inédites avec snapshot existant, déduplique clés, puis remplace
dataset via répertoire temporaire et renommage atomique. Rejouer même entrée ne crée
aucune ligne supplémentaire.

`overwrite_or_ignore` effectue fusion incrémentale exacte. `overwrite` et
`delete_matching` forcent snapshot calculé complet. `error` refuse destination existante.

Rapport features contient, pour chaque colonne : couverture, nombre de NaN, outliers
robustes au-delà de huit MAD normalisées et shift de moyenne récent exprimé en écarts-
types de première moitié.
