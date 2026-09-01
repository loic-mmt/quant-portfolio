# Documentation

Cette documentation décrit le fonctionnement exécutable du projet. Le `README.md` présente l'objectif de recherche ; les documents ci-dessous définissent les conventions nécessaires pour reproduire un run.

## Guides

- [Installation et commandes](getting-started.md) — créer l'environnement et lancer chaque étape.
- [Architecture du pipeline](architecture.md) — responsabilités des modules et artefacts produits.
- [Configuration](configuration.md) — univers, backtest et paramètres communs.
- [Données et features](data-and-features.md) — univers versionné, FX, contrôles qualité et catalogue des features.
- [Régimes walk-forward](regimes.md) — filtrage causal, mapping économique, confirmation 20/60 et artefacts.
- [Risque et allocation](risk-allocation.md) — MC pondéré, solveur contraint, overlay quotidien, fallbacks et replay.
- [Évaluation et rapport](evaluation-and-reporting.md) — baselines, ablations, métriques, attribution, provenance et HTML autonome.
- [Contrat temporel du backtest](temporal-contract.md) — information, décision, exécution, cash et coûts.

## Règle de lecture

En cas d'ambiguïté entre un exemple exploratoire dans `notebooks/` et cette documentation, les conventions de `docs/` et les validations du code de production font foi.
