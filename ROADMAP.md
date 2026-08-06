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
| Modèle de covariance | 🚧 Partiel | Estimateurs présents, validation et intégration walk-forward à compléter |
| Régimes HMM | ✅ Walk-forward | Filtrage causal, mapping économique, confirmation 20/60 et artefacts par calibration |
| Monte-Carlo | 🚧 Prototype | Simulation gaussienne présente, mais pas encore appliquée au portefeuille réel |
| Optimisation | 🚧 Prototype | Heuristique inverse-variance, contraintes et coûts non intégrés dans le solveur |
| Risk overlay quotidien | ⬜ À faire | Vol targeting, stress cut et gestion de la poche cash |
| Backtest | ✅ Fiable | Exécution décalée, cash, coûts, turnover et dérive des poids testés |
| Ablations et reporting | ⬜ À faire | Rapport, attribution par régime et comparaison des variantes absents |
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

- [ ] Calibrer `mu` et `Sigma` pour chaque date uniquement avec les observations passées du régime concerné.
- [ ] Utiliser la covariance shrinkée et conditionnée du module `models/covariance.py`.
- [ ] Simuler la distribution du portefeuille avec ses poids effectifs.
- [ ] Supporter les horizons 5 jours et 20 jours.
- [ ] Ajouter un seed configurable.
- [ ] Calculer et stocker :
  - [ ] VaR 1 % et 5 % ;
  - [ ] CVaR 1 % et 5 % ;
  - [ ] quantiles de PnL ;
  - [ ] probabilité de perte au-delà d'un seuil ;
  - [ ] probabilité de drawdown au-delà d'un seuil.
- [ ] Évaluer une loi Student-t ou un bootstrap historique comme alternative gaussienne.

### Optimisation contrainte

- [ ] Remplacer l'heuristique inverse-variance par un vrai problème `min wᵀΣw`.
- [ ] Respecter exactement `min_weight`, `max_weight`, long-only et exposition totale.
- [ ] Intégrer turnover, coûts et distance aux poids précédents dans l'objectif.
- [ ] Ajouter les contraintes sectorielles si les métadonnées sont disponibles.
- [ ] Ajouter une contrainte ou pénalité de volatilité cible.
- [ ] Définir les politiques de concentration et d'exposition par régime.
- [ ] Vérifier le statut du solveur et rendre explicite toute solution de repli.

### Risk overlay quotidien

- [ ] Estimer la volatilité prévisionnelle ou réalisée à chaque date.
- [ ] Calculer un multiplicateur d'exposition `target_vol / forecast_vol` borné.
- [ ] Appliquer un stress cut fondé sur le régime et les métriques MC.
- [ ] Autoriser une poche cash pour rendre le de-risking effectif.
- [ ] Ajouter hysteresis, vitesse maximale de variation et turnover governor.
- [ ] Journaliser chaque décision de réduction ou de restauration de l'exposition.

### Critères d'acceptation

- [ ] Toutes les contraintes sont vérifiées numériquement après optimisation.
- [ ] Le risque MC correspond au portefeuille pondéré, pas à la somme non pondérée des actifs.
- [ ] Une hausse de volatilité réduit effectivement l'exposition lorsque le cash est autorisé.
- [ ] Désactiver régimes ou MC produit des variantes distinctes et reproductibles.

---

## Jalon 5 — Évaluation, ablations et rapport final

### Baselines et ablations

- [ ] Equal-weight buy-and-hold.
- [ ] Equal-weight rebalancé.
- [ ] Minimum variance sans régimes.
- [ ] Stratégie sans overlay.
- [ ] Stratégie sans Monte-Carlo.
- [ ] Stratégie complète.
- [ ] Utiliser exactement les mêmes dates, univers et coûts pour toutes les variantes.

### Métriques

- [ ] CAGR, volatilité, Sharpe et Sortino.
- [ ] Max drawdown et temps de récupération.
- [ ] VaR/CVaR réalisées, downside deviation et fréquence des événements extrêmes.
- [ ] Turnover, coûts cumulés et capacité approximative.
- [ ] Concentration HHI, poids maximum et exposition cash.
- [ ] Attribution de performance et de risque par régime.
- [ ] Comparaison aux baselines avec tableaux et graphiques communs.

### Reporting

- [ ] Implémenter `pipeline/report.py`.
- [ ] Générer une equity curve, les drawdowns et l'exposition.
- [ ] Visualiser les régimes et leurs probabilités.
- [ ] Produire les performances, risques et turnovers par régime.
- [ ] Générer un tableau d'ablations.
- [ ] Exporter un rapport HTML ou PDF autonome.
- [ ] Enregistrer pour chaque run la configuration, le seed, la version Git et les dates de données.

### Critères d'acceptation

- [ ] Le rapport complet est généré par une seule commande.
- [ ] Toutes les métriques sont recalculables depuis les artefacts du run.
- [ ] Les conclusions séparent clairement performance in-sample et out-of-sample.

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
