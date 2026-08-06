# Régimes walk-forward

## Contrat temporel

À chaque date de recalibration `c` :

1. scaler ajusté uniquement avec lignes `date < c` ;
2. HMM ajusté uniquement sur cette même fenêtre historique ;
3. modèle figé produit probabilités de `c` jusqu'à veille de recalibration suivante ;
4. filtre forward calcule `P(S_t | x_1, ..., x_t)` ; aucune observation après `t` ;
5. prochaine recalibration recommence avec historique alors disponible.

Première prédiction apparaît après `min_train_size`. `train_window: null` donne fenêtre
expansive ; entier donne fenêtre roulante. Fréquence accepte `D`, pas observationnel
comme `20B`, ou période Pandas comme `W`, `M`, `Q`.

`GaussianHMM.predict()` et `predict_proba()` ne sont pas utilisés : ils reposent sur
algorithme forward-backward et fournissent états/postérieurs lissés par observations
futures. Implémentation utilise récursion forward normalisée en log-space.

## Interprétation économique

Numéros d'états HMM sont arbitraires et peuvent permuter entre recalibrations. Pour
chaque calibration, profil de chaque état brut est moyenne des features de train
pondérée par probabilités filtrées. Un score de risque standardisé donne :

- volatilité, drawdown, corrélation, dispersion et kurtosis : signe positif ;
- momentum, tendance et breadth : signe négatif.

État score minimum devient `calm`, maximum `stress`, états intermédiaires `choppy`.
Si profils extrêmes ne sont pas distincts, pipeline échoue explicitement. Mapping ne
dépend jamais du numéro brut.

Compatibilité aval :

| Régime | `state` | Probabilité |
|---|---:|---|
| `calm` | 0 | `p_calm`, alias `p_state_0` |
| `choppy` | 1 | `p_choppy`, alias `p_state_1` |
| `stress` | 2 | `p_stress`, alias `p_state_2` |

`raw_state` et `p_raw_*` restent disponibles pour audit, jamais pour politique
économique.

## Confiance, confirmation 20/60 et hysteresis

Chaque date possède régime candidat de probabilité sémantique maximale. Régime
confirmé reste inchangé tant que candidat différent ne satisfait pas simultanément :

- probabilité courante ≥ `confidence_threshold` ;
- sur 20 derniers jours, candidat occupe au moins `short_share` ;
- sur 60 derniers jours, probabilité moyenne du candidat ≥ `long_probability`.

Valeurs par défaut : confiance 55 %, présence 20 jours 60 %, probabilité 60 jours
40 %. Historique final du train initialise filtre de confirmation au début OOS.
Chaque ligne indique `candidate_regime`, `regime`, `transition` et
`transition_reason`.

## Artefacts et diagnostics

Chaque recalibration écrit sous `data/artifacts/regimes/DATE.json` :

- dates train/prédiction et tailles ;
- configuration, seed, features et fingerprint univers ;
- moyenne/écart-type scaler ;
- `startprob`, matrice de transition, moyennes et covariances HMM ;
- log-vraisemblance, convergence et itérations ;
- mapping brut/sémantique et profils économiques.

Répertoire est remplacé atomiquement. `data/quality/regimes_latest.json` contient
occupation, durées moyennes/maximales, matrice de transition empirique, profils OOS,
confiance et nombre de transitions confirmées.

Dataset `data/parquet/regimes/` est snapshot OOS complet et idempotent. Ajouter des
données futures ne modifie ni probabilités filtrées, ni états confirmés passés.
