# quant-portfolio

⸻

1) Vision du projet

But : construire un système quant complet qui :
	1.	identifie l’état du marché (régime),
	2.	estime le risque futur (distributions),
	3.	alloue les poids du portefeuille en conséquence,
	4.	contrôle le risque au quotidien (overlay),
	5.	backteste proprement en walk-forward,
	6.	sort un rapport clair.

⸻

2) Périmètre
	•	Univers : 40–50 actions.
	•	Fréquence : quotidienne (close-to-close).
	•	Horizon d’investissement : 1 an (janvier → décembre) pour la période d’évaluation “projet”.
	•	Horizon de décision : multi-horizon :
	•	estimation régime : 20 jours + 60 jours
	•	rebalancing : hebdo ou bi-hebdo
	•	overlay risque : quotidien

⸻

3) Entrées / données

Données requises
	•	Prix (Close ajusté idéalement) pour chaque action.
	•	Calendrier de trading.
	•	Optionnel : secteurs/industries (pour contraintes de diversification).

⸻

4) Pipeline global (de bout en bout)

Étape A — Ingestion & normalisation (Data Layer)

Objectif : avoir des séries propres, alignées, sans trous bizarres.
	•	Téléchargement incremental.
	•	Normalisation :
	•	alignement des dates,
	•	gestion des valeurs manquantes (drop ou forward-fill encadré),
	•	calcul des retours log ou simples,
	•	filtrage (période d’entraînement vs test).

Sorties :
	•	prices (tableau date × ticker)
	•	returns (date × ticker)

⸻

1) Feature engineering (Market + Cross-sectional)

Tu ne veux pas détecter des régimes sur 50 actions séparément. Tu construis un “Market State” via des agrégats.

Features “marché”
	•	Momentum : retours 20j, 60j
	•	Volatilité réalisée : std 20j, 60j
	•	Drawdown 20j/60j
	•	Vol-of-vol : variation de la vol (ex : std(Δvol))

Features cross-actions (très importantes)
	•	Corrélation moyenne (rolling) entre actions
→ en stress, les corrélations montent.
	•	Dispersion : écart-type des retours cross-sectionnels (winners vs losers)
	•	Breadth : % d’actions au-dessus de leur moyenne 50j/200j

Sorties :
	•	un dataframe features (date × features)

⸻

6) Détection de régimes (le cœur “macro”)

Objectif : produire un state S_t (ex : 3 états) + les probabilités par état.

Modèles conseillés
	•	HMM (Hidden Markov Model) à 3 états : le plus “quant pro” et interprétable.
	•	Alternative : GMM / clustering si tu veux plus simple.

Exemples d’états attendus (interprétation)
	•	Régime 1 : Calme / Trend
vol basse, momentum positif, corr modérée
	•	Régime 2 : Bruit / Mean reversion
vol moyenne, momentum faible, dispersion élevée
	•	Régime 3 : Stress / Risk-off
vol haute, drawdown, corr élevée

Sorties :
	•	regime_state[t]
	•	regime_probabilities[t, k]
	•	matrice de transition (utile pour anticiper la persistance)

⸻

7) Modèle de risque conditionnel (Monte-Carlo “utile”)

L’intérêt de Monte-Carlo ici : quantifier le risque futur, pas générer des chemins pour faire joli.

Calibration conditionnelle au régime

Pour chaque régime k :
	•	\mu_k (moyenne des retours, optionnel / peut être 0)
	•	\Sigma_k (covariance) avec shrinkage (stabilité indispensable)

Simulation multivariée
	•	Simule des retours à horizon H (5j, 20j) pour le portefeuille, pas action par action isolée.
	•	Distribution :
	•	Gaussienne (OK MVP)
	•	Student-t multivariée (plus réaliste : queues épaisses)

Ce que tu stockes (pas les chemins)
	•	VaR / CVaR (1%, 5%)
	•	prob(perte > x%)
	•	prob(drawdown > y%) sur 20j
	•	quantiles du PnL futur

Sorties :
	•	mc_summary[t, horizon]

⸻

8) Allocation dynamique (portfolio engine)

Tu crées une règle “régime → policy”.

Exemple de policies (long-only)
	•	Trend/Calme : risk-on
allocation type momentum + budget risque plus élevé, poids plus concentrés
	•	Mean reversion : diversification + rotation
plus de contraintes, poids plus répartis
	•	Stress : désendettement / réduction expo
baisse de l’exposition totale (cash), cap sur les poids, turnover réduit

Optimisation (pro et réaliste)

Tu optimises sous contraintes :
	•	Somme des poids = 1 (ou ≤1 si cash)
	•	0 ≤ w_i ≤ w_max
	•	turnover max (sinon backtest mensonger)
	•	coûts de transaction (bps) + slippage simple
	•	target vol (important)

Objectifs possibles :
	•	min variance (MVP robuste)
	•	max Sharpe (plus fragile)
	•	min CVaR (très pro)

Sorties :
	•	weights[t, asset]
	•	trades[t, asset] (deltas)
	•	portfolio_exposure[t]

⸻

9) Overlay de risque (la couche “institutionnelle”)

Même si tu rebalance hebdo, tu peux contrôler le risque tous les jours :
	•	Vol targeting : scale l’exposition pour coller à une vol cible (ex 10–15% annualisé)
	•	Stress cut : si prob(drawdown) ou CVaR dépasse un seuil → réduction automatique de l’exposition
	•	Turnover governor : limite les changements pour éviter de surtrader

C’est ce qui rend ton projet “pro” et pas juste académique.

⸻

10) Backtest : walk-forward + séparation train/test

Objectif : éviter l’auto-illusion.
	•	Période d’entraînement : plusieurs années (5–10 ans si possible)
	•	Période de test “projet” : l’année cible (jan→déc) en out-of-sample
	•	Walk-forward :
	•	tu recalibres (régimes, cov) sur une fenêtre glissante
	•	tu trades la fenêtre suivante

Métriques attendues
	•	Perf : CAGR, Sharpe, Sortino
	•	Risque : max drawdown, CVaR, VaR
	•	Stabilité : turnover, concentration (HHI), sensibilité aux paramètres
	•	Comparaisons :
	•	vs Equal Weight
	•	vs Buy&Hold market proxy
	•	vs stratégie sans régimes (ablation)

⸻

11) Stockage (concret, simple, efficace)

Vu que ta DB est minimaliste (cache prix), fais comme ça :
	•	Parquet / fichiers : prix, retours, features, covariances, weights, résultats backtest
	•	DB : seulement last_price_date (MVP)

Ensuite si tu veux “industrialiser” :
	•	DB pour runs, metrics, allocations, regime_state (audit + comparaison)

Réponse claire à ta question implicite :
➡️ Non, tu n’es pas obligé de stocker toutes les features en SQL. En pratique, Parquet est mieux pour des tableaux temporels.

⸻

12) Livrables finaux (ce qui fait “qualitatif”)
	1.	Code structuré + reproductible
	2.	Backtest walk-forward propre
	3.	Rapport final :
	•	description méthode
	•	régimes identifiés + interprétation
	•	règles d’allocation par régime
	•	résultats + risques + ablations
	•	limites + améliorations

⸻

13) Roadmap (ordre le plus intelligent)
	1.	Ingestion + cache DB last_date
	2.	Features + affichages diagnostics
	3.	HMM 3 régimes + validation “ça fait sens”
	4.	Allocation simple (min-variance + contraintes) + backtest baseline
	5.	Overlay vol targeting
	6.	Monte-Carlo conditionnel + stress cut
	7.	Walk-forward complet + rapport

⸻

Si tu me dis quelle source de données tu utilises (yfinance ? autre ?) et si ton univers est US ou EU, je te propose une structure de fichiers exacte (scripts + noms + signatures des fonctions) + un “pipeline runner” (un seul main.py qui exécute tout dans le bon ordre).