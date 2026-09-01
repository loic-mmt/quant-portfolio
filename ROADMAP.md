# Roadmap — Regime-Aware Dynamic Equity Portfolio

## Objectif

Construire un pipeline quantitatif reproductible capable de :

- détecter les régimes de marché sans utiliser d'information future ;
- estimer le risque futur conditionnellement au régime ;
- produire une allocation long-only contrainte avec une poche cash éventuelle ;
- appliquer quotidiennement un contrôle de volatilité et de risque extrême ;
- évaluer la stratégie dans un backtest walk-forward réaliste ;
- générer automatiquement les diagnostics, ablations et rapports finaux.

La priorité est la validité du protocole de recherche. L'optimisation des performances et l'accélération C++ viennent après l'élimination des biais temporels et la validation du backtest.

## État actuel

| Composant | État | Commentaire |
|---|---:|---|
| Ingestion Yahoo Finance | ✅ Robuste | Univers YAML versionné, actifs/FX incrémentaux et audit qualité explicite |
| Cache SQLite | ✅ Opérationnel | Schéma et vues versionnés, fraîcheur prix/features centralisée |
| Feature engineering | ✅ Robuste | Proxy rebasé sur rendements EUR, breadth corrigé et incrémental idempotent |
| Modèle de covariance | ✅ Intégré | Shrinkage, conditionnement et contrôle des historiques paire à paire |
| Régimes HMM | ✅ Walk-forward | Filtrage causal, mapping économique, confirmation 20/60 et artefacts par calibration |
| Monte-Carlo | ✅ Causal et pondéré | Poids effectifs/candidats/cibles, horizons 5/20 et comparaison gaussienne/Student-t |
| Optimisation | ✅ Contrainte | Minimum-variance CVXPY, coûts, turnover, secteurs et statuts vérifiés |
| Risk overlay quotidien | ✅ Implémenté | Ciblage de volatilité, stress cut, cash, hysteresis et gouverneurs décision/exécution |
| Backtest | ✅ Fiable | Exécution décalée, cash, coûts, turnover et dérive des poids testés |
| Ablations et reporting | ✅ Autonome | Six variantes comparables, métriques recalculables, attribution et HTML inline SVG |
| C++ | 🚧 Prototype | MC compilable, binding Python non importable et optimisation C++ absente |
| Tests et CI | 🚧 Partiel | Tests unitaires déterministes présents ; workflow CI reste à faire |

## Principes obligatoires

1. Toute décision prise à la date `t` utilise uniquement l'information disponible à `t`.
2. Les poids décidés à `t` sont exécutés au plus tôt à `t+1`, selon une convention documentée.
3. Modifier des données postérieures à une date ne doit jamais modifier les résultats antérieurs à cette date.
4. Chaque run doit être reproductible à partir de sa configuration, de son univers, de son seed et de la version du code.
5. Aucun échec du modèle de régime, du Monte-Carlo ou de l'optimiseur ne doit être ignoré silencieusement.
6. Une accélération C++ n'est intégrée qu'après validation de la version Python de référence.

---

## Jalon 0 — Projet exécutable et reproductible

### Packaging et dépendances

- [x] Compléter `pyproject.toml` avec la version de Python et les dépendances.
- [x] Transformer les sources Python en package `src/quant_portfolio` importable de manière cohérente.
- [x] Corriger les imports relatifs/absolus des pipelines.
- [x] Ajouter un lockfile de dépendances.
- [x] Documenter l'installation dans le README.

### Configuration et CLI

- [x] Déplacer l'univers de tickers vers `config/base.yaml`.
- [x] Compléter `config/backtest.yaml` avec dates, coûts, capital, taux cash et convention d'exécution.
- [x] Centraliser le chargement et la validation de la configuration dans `core/settings.py`.
- [x] Centraliser la génération des identifiants de run dans `core/ids.py`.
- [x] Implémenter un point d'entrée dans `main.py` avec les commandes :
  - `ingest` ;
  - `features` ;
  - `regimes` ;
  - `mc` ;
  - `optimize` ;
  - `backtest` ;
  - `report` ;
  - `run-all`.
- [x] Ajouter des logs structurés et remplacer les `print` de diagnostic.
- [x] Remplacer les `except Exception: pass` par des erreurs explicites ou des fallbacks tracés.

### Critères d'acceptation

- [x] Une installation depuis un clone propre est définie par une séquence documentée et verrouillée.
- [x] Chaque commande expose une aide utilisable avec `--help`.
- [x] Un smoke test exécute le moteur sur un petit jeu de données local.

---

## Jalon 1 — Contrat temporel et moteur de backtest fiable

### Convention temporelle

- [x] Définir explicitement `information_date`, `decision_date` et `execution_date`.
- [x] Utiliser les rendements simples comme convention canonique et convertir explicitement les log-returns.
- [x] Décaler les signaux et les poids pour interdire l'utilisation du rendement du jour de décision.
- [x] Ajouter des tests automatiques anti-look-ahead.

### Comptabilité du portefeuille

- [x] Faire dériver les poids entre deux rebalancements.
- [x] Facturer l'allocation initiale et chaque transaction.
- [x] Modéliser explicitement la poche cash et son rendement.
- [x] Définir une politique configurable pour les rendements manquants.
- [x] Appliquer les coûts, le slippage et le turnover sur les poids réellement exécutés.
- [x] Séparer poids cibles, poids exécutés, positions et trades.
- [x] Limiter le backtest à la période de test configurée.

### Critères d'acceptation

- [x] Un portefeuille mono-actif reproduit exactement le buy-and-hold attendu.
- [x] Un portefeuille sans frais réconcilie sa valeur avec ses positions et son cash.
- [x] Ajouter ou modifier des données futures ne change aucun résultat passé.
- [x] Les trades reconstruisent le passage des poids précédents aux poids exécutés.

---

## Jalon 2 — Données et features robustes

### Univers et qualité des données

- [x] Définir un univers fixe et versionné pour chaque expérience.
- [x] Traiter explicitement les tickers sans données ou délistés.
- [x] Documenter le survivorship bias et la source des constituants.
- [x] Définir la devise de référence du portefeuille.
- [x] Ajouter les conversions FX nécessaires ou limiter l'univers à une devise homogène.
- [x] Contrôler doublons, dates manquantes, prix non positifs et corporate actions.

### Feature engineering

- [x] Construire le proxy marché à partir de rendements ou de séries rebasées, pas de niveaux de prix bruts.
- [x] Corriger les dénominateurs des mesures de breadth quand des actifs sont manquants.
- [x] Utiliser une fenêtre incrémentale suffisante pour les features à 200/252 jours.
- [x] Garantir l'idempotence des écritures Parquet.
- [x] Vérifier que recomputation complète et mise à jour incrémentale donnent les mêmes résultats.
- [x] Produire un rapport de qualité des features : couverture, NaN, outliers et stabilité.

### Critères d'acceptation

- [x] Les résultats incrémentaux sont identiques aux résultats d'une recomputation complète.
- [x] Toutes les features documentent leur fenêtre, fréquence et convention d'annualisation.
- [x] Le proxy marché est invariant au niveau nominal des cours et cohérent entre devises.

---

## Jalon 3 — Régimes walk-forward interprétables

### Modèle

- [x] Paramétrer le nombre d'états, le seed, le type de covariance et les fenêtres d'entraînement.
- [x] Entraîner le scaler et le HMM uniquement sur la fenêtre historique autorisée.
- [x] Recalibrer le modèle selon une fréquence configurable.
- [x] Utiliser des probabilités filtrées disponibles en temps réel, pas des états lissés avec le futur.
- [x] Produire les états et probabilités uniquement pour la période suivante.
- [x] Sauvegarder les paramètres et métadonnées de chaque recalibration.

### Interprétation et stabilité

- [x] Mapper chaque état brut vers `calm`, `choppy` ou `stress` à partir de ses statistiques.
- [x] Garantir que le mapping ne dépend pas du numéro arbitraire attribué par le HMM.
- [x] Ajouter un seuil de confiance et une règle de transition/hysteresis.
- [x] Implémenter la confirmation 20 jours / 60 jours annoncée dans le README.
- [x] Produire durée moyenne, matrice de transition, occupation et profil de chaque régime.

### Critères d'acceptation

- [x] Les régimes passés ne changent pas lorsque des données futures sont ajoutées.
- [x] Chaque état possède une interprétation économique calculée et documentée.
- [x] Deux runs avec le même seed et les mêmes données produisent les mêmes résultats.

---

## Jalon 4 — Risque, optimisation et overlay quotidien

### Monte-Carlo conditionnel au régime

- [x] Calibrer `mu` et `Sigma` pour chaque date uniquement avec les observations passées du régime concerné.
- [x] Utiliser la covariance shrinkée et conditionnée du module `models/covariance.py`.
- [x] Simuler la distribution du portefeuille avec ses poids effectifs.
- [x] Supporter les horizons 5 jours et 20 jours.
- [x] Ajouter un seed configurable.
- [x] Calculer et stocker :
  - [x] VaR 1 % et 5 % ;
  - [x] CVaR 1 % et 5 % ;
  - [x] quantiles de PnL ;
  - [x] probabilité de perte au-delà d'un seuil ;
  - [x] probabilité de drawdown au-delà d'un seuil.
- [x] Évaluer une loi Student-t ou un bootstrap historique comme alternative gaussienne.

### Optimisation contrainte

- [x] Remplacer l'heuristique inverse-variance par un vrai problème `min wᵀΣw`.
- [x] Respecter exactement `min_weight`, `max_weight`, long-only et exposition totale.
- [x] Intégrer turnover, coûts et distance aux poids précédents dans l'objectif.
- [x] Ajouter les contraintes sectorielles si les métadonnées sont disponibles.
- [x] Ajouter une contrainte ou pénalité de volatilité cible.
- [x] Définir les politiques de concentration et d'exposition par régime.
- [x] Vérifier le statut du solveur et rendre explicite toute solution de repli.

### Risk overlay quotidien

- [x] Estimer la volatilité prévisionnelle ou réalisée à chaque date.
- [x] Calculer un multiplicateur d'exposition `target_vol / forecast_vol` borné.
- [x] Appliquer un stress cut fondé sur le régime et les métriques MC.
- [x] Autoriser une poche cash pour rendre le de-risking effectif.
- [x] Ajouter hysteresis, vitesse maximale de variation et turnover governor.
- [x] Journaliser chaque décision de réduction ou de restauration de l'exposition.

### Critères d'acceptation

- [x] Toutes les contraintes sont vérifiées numériquement après optimisation.
- [x] Le risque MC correspond au portefeuille pondéré, pas à la somme non pondérée des actifs.
- [x] Une hausse de volatilité réduit effectivement l'exposition lorsque le cash est autorisé.
- [x] Désactiver régimes ou MC produit des variantes distinctes et reproductibles.

Contrat et limites : [Risque et allocation](docs/risk-allocation.md). Le repli
`pooled` sur régime trop court est configurable et explicitement tracé ; `error`
impose le conditionnement strict. Secteurs optionnels, objectifs de risque souples
sous gouverneurs durs. La comparaison empirique des stratégies reste au jalon 5.

---

## Jalon 5 — Évaluation, ablations et rapport final

### Baselines et ablations

- [x] Equal-weight buy-and-hold.
- [x] Equal-weight rebalancé.
- [x] Minimum variance sans régimes.
- [x] Stratégie sans overlay.
- [x] Stratégie sans Monte-Carlo.
- [x] Stratégie complète.
- [x] Utiliser exactement les mêmes dates, univers et coûts pour toutes les variantes.

### Métriques

- [x] CAGR, volatilité, Sharpe et Sortino.
- [x] Max drawdown et temps de récupération.
- [x] VaR/CVaR réalisées, downside deviation et fréquence des événements extrêmes.
- [x] Turnover, coûts cumulés et capacité approximative.
- [x] Concentration HHI, poids maximum et exposition cash.
- [x] Attribution de performance et de risque par régime.
- [x] Comparaison aux baselines avec tableaux et graphiques communs.

### Reporting

- [x] Implémenter `pipeline/report.py`.
- [x] Générer une equity curve, les drawdowns et l'exposition.
- [x] Visualiser les régimes et leurs probabilités.
- [x] Produire les performances, risques et turnovers par régime.
- [x] Générer un tableau d'ablations.
- [x] Exporter un rapport HTML ou PDF autonome.
- [x] Enregistrer pour chaque run la configuration, le seed, la version Git et les dates de données.

### Critères d'acceptation

- [x] Le rapport complet est généré par une seule commande.
- [x] Toutes les métriques sont recalculables depuis les artefacts du run.
- [x] Les conclusions séparent clairement performance in-sample et out-of-sample.

Contrat, formules et limites : [Évaluation et rapport](docs/evaluation-and-reporting.md).
Période calibration/warmup ne publie aucun score de performance ; comparaisons et
conclusions utilisent uniquement première exécution → fin figée du run.

---

## Jalon 6 — Tests, CI et gestion des artefacts

### Tests

- [ ] Tests unitaires des fonctions de rendement, covariance, turnover, coûts, VaR et contraintes.
- [ ] Tests d'intégration de chaque pipeline.
- [ ] Test end-to-end sur un fixture déterministe.
- [ ] Tests de non-régression sur un petit jeu de résultats de référence.
- [ ] Tests spécifiques anti-look-ahead et idempotence incrémentale.

### CI et qualité

- [ ] Ajouter linting, formatage et vérification des types.
- [ ] Exécuter les tests automatiquement en CI.
- [ ] Ajouter une matrice minimale de versions Python supportées.
- [ ] Vérifier les migrations SQLite et les schémas Parquet.

### Données et artefacts

- [ ] Décider quels Parquet et bases SQLite doivent rester versionnés dans Git.
- [ ] Déplacer les données générées vers un stockage d'artefacts ou DVC si nécessaire.
- [ ] Conserver dans Git uniquement les petits fixtures requis par les tests.
- [ ] Ajouter des manifests avec checksum, source et date de collecte.

### Critères d'acceptation

- [ ] La CI passe depuis un clone propre sans dépendre d'un environnement local préexistant.
- [ ] Les données utilisées par un run peuvent être identifiées sans ambiguïté.

---

## Jalon 7 — Accélération C++ optionnelle

- [ ] Corriger le nom du module : cible CMake et symbole `PYBIND11_MODULE` doivent correspondre.
- [ ] Remplacer les chemins Python/pybind11 codés en dur par `find_package`.
- [ ] Connecter le moteur C++ au pipeline Python derrière une interface commune.
- [ ] Ajouter des tests de parité numérique Python/C++.
- [ ] Benchmarker MC et optimisation avant/après accélération.
- [ ] Implémenter l'optimisation C++ uniquement si le benchmark justifie sa maintenance.
- [ ] Conserver une implémentation Python de référence pour la validation.

### Critères d'acceptation

- [ ] Le module compilé est importable sur les plateformes supportées.
- [ ] Les résultats C++ respectent les tolérances définies face à la version Python.
- [ ] Le gain de performance est mesuré et documenté.

---

## Ordre d'exécution

```text
Jalon 0 : projet exécutable
    ↓
Jalon 1 : timing et comptabilité fiables
    ↓
Jalon 2 : données et features robustes
    ↓
Jalon 3 : régimes walk-forward
    ↓
Jalon 4 : MC + optimisation + overlay
    ↓
Jalon 5 : ablations et rapport
    ↓
Jalon 6 : durcissement CI/artefacts
    ↓
Jalon 7 : accélération C++ si nécessaire
```

Les tests du jalon 6 doivent être ajoutés progressivement dès chaque jalon, même si leur industrialisation complète intervient plus tard.

## Definition of Done du projet

Le projet est considéré terminé lorsque :

- [ ] le pipeline complet s'exécute depuis un clone propre avec une commande documentée ;
- [ ] aucune étape n'utilise d'information future ;
- [ ] le backtest modélise correctement positions, cash, trades, coûts et exécution ;
- [ ] les régimes sont walk-forward, interprétables et reproductibles ;
- [ ] le Monte-Carlo mesure le risque du portefeuille réellement détenu ;
- [ ] l'optimisation respecte toutes les contraintes après résolution ;
- [ ] le vol targeting et le stress cut fonctionnent quotidiennement ;
- [ ] les baselines et ablations utilisent un protocole identique ;
- [ ] le rapport final expose performance, risque, turnover et attribution par régime ;
- [ ] les tests automatiques et la CI passent ;
- [ ] chaque résultat peut être relié à sa configuration, ses données, son seed et sa version Git.
