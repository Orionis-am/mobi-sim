# MobiSim — Rapport technique

**Plateforme hybride de services de communication mobile — Simulation × APIs gratuites ×
Algorithmique évolutionnaire**
MSc Télécommunications — Projet individuel — Python 3.10+

---

## 1. Introduction

MobiSim est une plateforme qui combine, sur six modules, trois façons de produire de la donnée
télécom : simulation logicielle pure, appel à de vraies APIs gratuites, et optimisation par
algorithmes évolutionnaires. Aucun matériel physique n'est requis — un seul ordinateur, une
connexion internet.

| Module | Domaine | APIs / libs réelles |
|---|---|---|
| A | Codecs audio & qualité perceptive | OpenAI Whisper, gTTS |
| B | SMS hybride (SMSC/HLR/VLR simulés) | Twilio (trial) |
| C | Géolocalisation LBS | OpenCelliD (offline), Nominatim/Overpass, ipinfo.io |
| D | QoS/QoE (modèle E ITU-T) | `stun.l.google.com:19302` |
| E | Passerelle API REST | FastAPI, SQLite |
| F | Algorithmique évolutionnaire | DEAP, pymoo, scipy, pyswarm + ABC/CMA-ES faits maison (§8.8) |

Fil conducteur double, tenu sur les six modules : (1) l'écart entre ce qu'une simulation prédit et
ce qu'une vraie API/mesure produit ; (2) l'apport et les limites des algorithmes évolutionnaires
sur des problèmes télécom concrets (Module F).

**État du projet** : les six modules sont complets — code, tests, validation contre au moins une
vraie API par module quand le module en dépend. **511 tests**, **97 % de couverture globale**
(pytest-cov). Détail par module dans les sections suivantes.

---

## 2. Architecture générale

```
        ┌──────────────┐     fitness()      ┌──────────────────────┐
        │  Module A     │ ─────────────────▶ │                       │
        │  Codecs       │                     │                       │
        ├──────────────┤     fitness()       │      Module F         │
        │  Module C     │ ─────────────────▶ │  AG (DEAP) / NSGA-II   │
        │  LBS          │                     │  MOEA/D (pymoo) / DE   │
        ├──────────────┤     fitness()       │  (scipy) / PSO (pyswarm)│
        │  Module D     │ ─────────────────▶ │                       │
        │  QoS/QoE      │                     └───────────┬───────────┘
        └──────────────┘                                  │
        ┌──────────────┐                                  │
        │  Module B     │  (SMS, indépendant de F)          │
        └──────┬───────┘                                  │
               │                                           │
               ▼                                           ▼
        ┌────────────────────────────────────────────────────────┐
        │                Module E — FastAPI (JWT, SQLite)          │
        │  /auth  /catalog  /codecs  /sms  /location  /qos  /optimize │
        └────────────────────────────────────────────────────────┘
```

Modules A–D exposent chacun une fonction `*_fitness(chromosome)` pure et stable ; Module F les
consomme sans connaître leur implémentation interne ; Module E expose le tout via HTTP. Construits
dans cet ordre (A→B→C→D→F→E) précisément parce que F dépend des contrats A/C/D, et que les
endpoints `/optimize/*` de E orchestrent F.

Choix de stack, en bref :
- **FastAPI** — validation Pydantic native, documentation Swagger générée automatiquement (L5),
  support natif de tâches de fond (`BackgroundTasks`) sans infrastructure de file externe.
- **SQLite** — contrainte « un seul ordinateur standard », zéro serveur à administrer.
- **DEAP / pymoo / scipy / pyswarm** — un par famille d'algorithme demandée (AG / NSGA-II-MOEA-D /
  DE / PSO), plutôt qu'une réimplémentation générique.
- **SlowAPI** — rate limiting compatible ASGI, décorateur par route.

---

## 3. Module A — Codecs audio & qualité perceptive

### 3.1 Pipeline

Signal de référence synthétisé (gTTS, repli pyttsx3), **8 kHz mono** — bande téléphonique
narrowband, cohérente avec GSM-FR et avec le mode narrowband du PESQ ITU-T P.862. Trois
dégradations modélisées à la main (pas de codec réel) :

| Codec | Débit cible | Modèle de dégradation |
|---|---|---|
| AAC | 16 kbps | Passe-bas + pré-écho sur transitoires |
| GSM-FR | 13 kbps | Passe-bande + bruit gaussien large-bande |
| Opus | 24 kbps | Quasi-transparent + distorsion harmonique cubique légère |

Trois métriques objectives (`metrics.py`) : SNR, LSD, et un **PESQ-NB simplifié** (corrélation
spectrale + enveloppe temporelle, pas de librairie PESQ sous licence).

### 3.2 Résultats

**Métriques objectives** (tonalité de test 8 kHz, seed=42) :

| Codec | SNR (dB) ↑ | LSD ↓ | PESQ-NB simplifié ↑ |
|---|---|---|---|
| AAC | 25.8 | 23.1 | 4.47 |
| GSM | 13.8 | 30.9 | 3.93 |
| Opus | 35.1 | 13.8 | 4.50 |

**MUSHRA simulé** (panel de 20 auditeurs par groupe, bootstrap 95 %, calibré sur les moyennes du
rapport fourni) :

| Codec | English — moy. [IC 95 %] | Native — moy. [IC 95 %] |
|---|---|---|
| Opus | 55.0 [49.1, 60.2] | 62.0 [58.1, 65.8] |
| GSM | 47.3 [41.2, 53.1] | 49.4 [41.5, 57.4] |
| AAC | 34.4 [26.4, 42.0] | 33.3 [27.3, 39.5] |

L'ordre Opus > GSM > AAC du rapport fourni est reproduit dans les deux groupes.

**WER réel (vraie API Whisper, texte de parole synthétisée, 2 exécutions identiques)** :

| Codec | SNR (dB) | PESQ-NB (sim) | WER réel | Transcription |
|---|---|---|---|---|
| AAC | 21.4 | 4.50 | **0.000** | correcte |
| GSM | 4.3 | 4.20 | **0.000** | correcte |
| Opus | 35.0 | 4.50 | **1.000** | dérive en français, hallucination |

**Optimisation (Pb1, Module F, AG DEAP)** : sur `codec_fitness`, un optimum intérieur réel apparaît
autour de **16–24 kbps** (au-delà, le gain PESQ ne compense plus la pénalité de bitrate) —
meilleur individu trouvé : `codec_fitness = 2.489`. Courbe de convergence :
`results/module_f/pb1_codec_convergence.png`.

*[Figure 1 — à insérer : `plot_mushra_comparison` + `plot_correlation_matrix` (module_a.visualize),
générées par `synth_audio → codecs → metrics → mushra_sim → visualize` sur la parole TTS réelle]*

### 3.3 Discussion — Q1 : PESQ vs. WER, complémentarité des métriques

Le rapport fourni donne un PESQ NB quasi identique pour Opus (1.52) et GSM (1.53) ; notre PESQ-NB
simplifié montre le même phénomène de rapprochement objectif (tableau §3.2, AAC/GSM proches, écart
plus resserré qu'avec MUSHRA). Le WER réel, lui, sépare radicalement les trois codecs — et dans le
sens inverse de ce qu'un score SNR/PESQ laisserait attendre : Opus (meilleur SNR, meilleur PESQ)
échoue totalement à l'intelligibilité (WER=1.0), AAC et GSM (moins bons en métriques objectives)
transcrivent parfaitement. Un artefact fin, quasi invisible à une métrique de corrélation globale,
suffit à faire dérailler un système de reconnaissance vocale. Ceci est développé plus en détail à
l'oral.

---

## 4. Module B — SMS hybride

### 4.1 Pipeline

Architecture GSM simulée (`entities.py` : `Hlr`/`Vlr`/`Msc`/`Smsc`, état pur) + dynamique temporelle
séparée (`network_sim.py` : délai exponentiel, perte, retransmission avec backoff, fenêtre de
réessai 72 h). Encodage/décodage PDU complet (`pdu.py`, 3GPP TS 23.040 — alphabet GSM 7 bits,
empaquetage 7→8 bits, BCD).

**Exemple PDU vérifié à la main** (`SubmitPdu(destination="+33612345678", text="Hi")`) :

```
00 01 00 0B 91 3316325476F8 00 00 02 C834
```

### 4.2 Résultats

**QoS sous charge** (`simulate_overload`, file M/D/1 à capacité bornée) : taux de perte et délai
moyen croissants avec le taux d'arrivée — courbe produite par `sweep_overload`.
*[Figure 2 — à insérer : courbe perte/délai vs. charge]*

**Envois réels sur compte Twilio trial** :

| Voie | Résultat |
|---|---|
| Verify (OTP) | `status=approved`, aller-retour envoi→réception humaine→vérification réussi |
| Messages (SMS libre, via nom de template) | `accept_latency_s=2.09`, `delivery_latency_s=5.39`, `final_status=delivered` |

### 4.3 Discussion — Q2 : simulation vs. Twilio réel, causes structurelles

Trois causes structurelles mesurées, absentes du simulateur :

1. **Politique de contenu du compte trial** — `messages.create(body=<texte libre>)` est rejeté
   (erreur 60409) ; seul le *nom* d'un template prédéfini passe. `entities.Smsc` n'a aucune notion
   de contenu autorisé.
2. **Asymétrie d'accès API** — `GET /Messages/{Sid}` renvoie 403 alors que `GET /Messages` (liste)
   fonctionne pour le même SID, y compris juste après envoi (indexation Twilio pas encore à jour).
3. **Composition de la latence réelle** — `accept_latency_s` (acceptation par l'API) et
   `delivery_latency_s` (jusqu'au statut terminal) sont deux temps distincts liés à
   l'infrastructure Twilio (file interne, opérateur destinataire), que le modèle exponentiel simulé
   à un seul paramètre ne décompose pas.

Piste d'intégration : calibrer la moyenne du délai exponentiel simulé sur `delivery_latency_s`
mesuré, et ajouter un état « contenu non autorisé » à `Smsc.send` pour modéliser (1).

---

## 5. Module C — Géolocalisation LBS

### 5.1 Pipeline

Données réelles : dump OpenCelliD offline (`docs/208.csv`, 392 420 lignes), zone Aveyron (6 815
BTS dans la bbox retenue), projection équirectangulaire locale. Trois méthodes de positionnement +
une géolocalisation IP :

| Méthode | Principe | Fichier |
|---|---|---|
| Cell-ID | Centroïde de cellule Voronoï (BTS la plus proche) | `cell_id.py` |
| TOA | Trilatération SLSQP sur pseudo-délais bruités | `toa.py` |
| Wi-Fi fingerprinting | k-NN sur grille RSSI simulée (zone locale 2 km) | `wifi_fp.py` |
| IP | `ipinfo.io` (réel) | `ipinfo_client.py` |

### 5.2 Résultats — tableau comparatif de précision

Sur les mêmes 500 BTS Aveyron échantillonnées (`seed=42`), 100 positions simulées :

| Méthode | Erreur médiane |
|---|---|
| Cell-ID | ≈ 3,4 km |
| TOA | ≈ 19 m |
| Wi-Fi (bruit nul) | ≈ 25,7 m |
| Wi-Fi (4 / 8 / 16 dB de bruit) | 488 m / 992 m / 1116 m |
| IP (ipinfo.io, réel) | échelle ville (Toulouse localisée correctement) |

**Validations réelles** : `ipinfo.io` a localisé l'IP publique de la machine à Toulouse
(43.6043, 1.4437) ; Overpass a renvoyé 10 POI réels autour du centre de Rodez (86–238 m). Carte
Folium démonstrative bout-en-bout : `results/module_c_demo.html` (position estimée + réelle + 8
POI + cercle d'incertitude).

*[Figure 3 — capture de `results/module_c_demo.html`]*

**Placement BTS (Pb2, Module F)** : `bts_coverage_fitness` retourne un triplet Pareto
(couverture, interférence, coût) — voir §8.2.

### 5.3 Discussion — usages et limites

L'écart de précision Cell-ID/TOA (3,4 km vs. 19 m) illustre le compromis coût/précision d'un vrai
déploiement LBS opérateur : Cell-ID est gratuit (déjà disponible via l'attachement réseau) mais
grossier ; TOA demande une synchronisation temporelle fine sur plusieurs antennes. La géoloc IP,
précise à l'échelle ville au mieux, limite ses usages légitimes à la personnalisation grossière ou
la détection de fraude — pas le suivi individuel.

---

## 6. Module D — QoS/QoE

### 6.1 Pipeline

Modèle E ITU-T G.107 complet (`model_e.py` : `Id` par formule standard, `Ie,eff` ajusté à la perte
de paquets, conversion `R→MOS` cubique standard), mesures RTT/gigue/perte **réelles** via requêtes
STUN RFC 5389 construites à la main (`stun_probe.py`, `socket` stdlib, aucune librairie tierce),
dashboard temps réel Rich (`dashboard.py`) avec export CSV en continu.

**Vérifié à la main** : `r_factor("opus", delay=0, loss=0) = 96.2` ; `r_to_mos(93.2) ≈ 4.409`.

### 6.2 Résultats

**Mesure STUN réelle** (20 aller-retours UDP vers `stun.l.google.com:19302`) :

| RTT moyen | Gigue | Perte |
|---|---|---|
| 17,51 ms | 3,57 ms | 0,00 % |

**MOS prédit à partir de ces conditions réelles** : `opus = 4.46 > gsm = 4.13 > aac = 3.94` —
ordre conforme à la table `Ie` du sujet.

**Courbes MOS = f(délai)** et **MOS = f(perte)**, par codec, seuils R annotés (R<60 insatisfaisant,
60–80 acceptable, >80 bon) : `correlation.py`.
*[Figure 4 — à insérer : `plot_mos_vs_delay` / `plot_mos_vs_loss`]*

**Corrélation globale** : `correlation.plot_correlation_matrix` réutilise directement
`module_a.visualize.plot_correlation_matrix` pour fusionner PESQ/WER/MUSHRA/MOS dans une même
heatmap.

---

## 7. Module E — API REST

FastAPI, authentification JWT OAuth2 à scopes (`admin`/`operator`/`user`, access 30 min / refresh
7 j), catalogue de 20 services en SQLite, rate limiting SlowAPI (60 req/min par token, IP en
repli), Swagger UI sur `/docs`.

| Domaine | Endpoints | Notes |
|---|---|---|
| Auth | `POST /auth/token`, `POST /auth/register` | register hors sujet littéral, ajouté sur décision utilisateur |
| Catalogue | `GET /services`, `GET /services/{id}` | lecture SQLite |
| Codecs | `POST /codecs/evaluate` | wrapper `module_a.fitness` |
| SMS | `POST /sms/send`, `GET /sms/status/{sid}` | wrapper `module_b.twilio_client` |
| Location | `POST /location/estimate` | 4 méthodes (cell_id/toa/wifi/ip) |
| QoS | `GET /qos/session`, `GET /qos/session/live` (SSE) | wrapper `module_d` |
| Optimize | `POST /optimize/run`, `GET /optimize/{job_id}` | `BackgroundTasks`, orchestre les 4 familles d'algos de Module F |

*[Figure 5 — capture Swagger UI `/docs`]*

**Tests** : 60 tests, 98 % de couverture sur `module_e` (branches non couvertes par conception :
`use_real_stun=true`, réseau réel ; chemins d'erreur `.env` cassé). Suite complète du dépôt : **511
tests, 97 % de couverture globale** (compte final, après les deux algorithmes bonus de Module F —
§8.8).

---

## 8. Module F — Algorithmique évolutionnaire

### 8.1 AG from-scratch et validation

AG en Python pur (sélection par tournoi k=3, croisement SBX η=20, mutation gaussienne adaptative à
décroissance exponentielle, élitisme 10 %), validé sur Rastrigin/Rosenbrock (2D/10D) contre
recherche aléatoire et hill climbing, **à budget d'évaluations identique**.

![Convergence Rastrigin 2D](results/module_f/benchmark_rastrigin_2d.png)

Axe X en évaluations de fitness (budget identique pour les 3 courbes, corrigé le
2026-09-08 — voir REPORT.md, `plot_convergence` journalisait auparavant l'AG par génération et les
deux baselines par évaluation sur le même axe, sous-représentant l'AG d'un facteur `pop_size`).
Sur ce run (2D, budget 1200 évaluations, seed=0), l'AG plafonne à `Rastrigin(x) ≈ 4.5` quand
random_search et hill climbing descendent sous 2.0 — résultat gardé tel quel plutôt que lissé, et
repris en §8.6 pour la discussion No Free Lunch.

Balayage d'hyperparamètres (population × croisement × mutation) → 3 heatmaps 2D (une par taille de
population), score final moyenné sur 10 exécutions.

### 8.2 Pb1 — configuration codec (AG DEAP)

Contre `module_a.fitness.codec_fitness`, comparé à une recherche aléatoire et une grille exhaustive
(300 évaluations). Meilleur individu : `codec_fitness = 2.489`, optimum intérieur ≈ 16–24 kbps
(§3.2).

![Convergence Pb1](results/module_f/pb1_codec_convergence.png)

Axe X également corrigé en évaluations de fitness (même bug d'unité que §8.1, retrouvé ici après
coup). Ici, contrairement à Rastrigin, l'AG (DEAP) est quasi optimal dès sa population initiale et
reste au-dessus ou à l'égal de random_search sur tout le budget partagé (300 évaluations) —
random_search ne rattrape le plateau qu'après ≈25 évaluations. Repris en §8.6.

### 8.3 Pb2 — placement BTS (NSGA-II vs. MOEA/D)

3 objectifs (couverture, interférence, coût) contre `module_c.fitness.bts_coverage_fitness`. Point
de compromis choisi par distance euclidienne au point idéal normalisé (knee point).

| Algorithme | Front final | Évaluations |
|---|---|---|
| NSGA-II | 44 points non dominés | 2500 |
| MOEA/D | 91 points non dominés | 2275 |

![Front de Pareto Pb2, N=10](results/module_f/pb2_pareto_n10.png)
![Hypervolume Pb2](results/module_f/pb2_hypervolume.png)

Deux problèmes de comparabilité corrigés le 2026-09-08 (détail dans REPORT.md) : le premier (axe X
générations vs. évaluations, §8.1/§8.2) ne s'appliquait pas ici — NSGA-II et MOEA/D journalisent
déjà tous deux un point d'hypervolume par génération pymoo. Le second en est un nouveau : chaque
appel de `hypervolume_history` calculait par défaut son point de référence à partir du seul run
passé en argument, donc NSGA-II et MOEA/D étaient chacun notés sur une échelle différente — deux
courbes visuellement comparées mais pas rigoureusement comparables. Corrigé via
`shared_reference_point`, qui calcule un point de référence unique couvrant le pire point des deux
runs. La figure d'hypervolume ci-dessus vient de ce run corrigé (NSGA-II `pop_size=100`, MOEA/D
`n_partitions=12`, 25 générations, seed=0, tableau ci-dessus) ; le nuage de points Pareto reste
celui d'un run antérieur non reproductible (1000/700 évaluations, paramètres non enregistrés) —
gardé néanmoins pour sa valeur illustrative, puisqu'il montre déjà qualitativement le même
phénomène (front MOEA/D nettement plus resserré que celui de NSGA-II).

Carte Folium du meilleur compromis : `results/module_f/pb2_bts_map.html`.

### 8.4 Pb3 — allocation QoS multi-utilisateurs (DE vs. PSO)

10 profils utilisateurs (VoIP/SMS/streaming), contrainte dure de capacité totale (70 % de la
demande cumulée), contre `module_d.fitness.qos_fitness` (scalaire pondéré). Comparaison à budget
fixé (1500 évaluations, moyenné sur 5 runs).

![DE vs PSO, budget fixé](results/module_f/pb3_de_vs_pso.png)

### 8.5 Tableau comparatif

| Problème | Algorithmes comparés | Axe de comparaison primaire |
|---|---|---|
| Pb1 (codec) | AG DEAP / recherche aléatoire / grille exhaustive / CMA-ES / ABC (§8.8) | évaluations de fitness égales |
| Pb2 (BTS) | NSGA-II / MOEA/D | hypervolume à évaluations égales |
| Pb3 (QoS) | DE (scipy) / PSO (pyswarm) | fitness à budget d'évaluations fixé |

### 8.6 Discussion — Q3 : No Free Lunch appliqué aux télécoms

Les courbes d'hypervolume (Figure Pb2 ci-dessus) et de convergence DE/PSO (Figure Pb3) permettent
de comparer les algorithmes *à budget identique*, condition nécessaire pour toute affirmation de
type NFL : sans elle, un algorithme peut sembler supérieur simplement parce qu'il a eu plus
d'évaluations — les Figures §8.1 et §8.2 en sont un rappel direct, toutes deux corrigées après une
même confusion d'unité d'axe entre points-par-génération et points-par-évaluation (cf. REPORT.md).
Une fois l'axe corrigé, les deux AG (from-scratch sur Rastrigin, DEAP sur Pb1) donnent des verdicts
opposés à budget d'évaluations réellement égal : sur Rastrigin 2D, l'AG plafonne *au-dessus* de
random_search et hill climbing — l'algorithme le plus sophistiqué n'y est pas le meilleur ; sur Pb1
(codec), l'AG DEAP est quasi optimal dès sa population initiale et reste devant random_search sur
tout le budget partagé. Même code d'AG-baseline, même correctif de figure, deux verdicts opposés
selon la structure du problème — une illustration directe du NFL avant même d'atteindre Pb2/Pb3.
Sur Pb2, une fois les deux algorithmes notés sur un point de référence d'hypervolume partagé (§8.3,
correctif distinct de celui d'axe X ci-dessus), NSGA-II progresse nettement sur les 25 générations
(hypervolume ×1,4) tandis que MOEA/D régresse fortement sur le même repère (÷14) — à ces
hyperparamètres (`n_neighbors=15`, `prob_neighbor_mating=0.7`, 12 partitions) et sur ce budget,
MOEA/D ne l'emporte pas ici, contrairement à l'intuition qu'une méthode par décomposition serait
mécaniquement compétitive sur un problème à 3 objectifs. Sa taille de population reste cela dit
fixée par le nombre de directions de référence (pas un paramètre libre comme pour NSGA-II), ce qui
structure différemment l'exploration du front selon sa géométrie réelle — un autre hyperparamétrage
(plus de partitions, plus de générations, un voisinage différent) pourrait renverser ce verdict, non
testé ici. Sur Pb3, DE et PSO sont deux métaheuristiques
sans gradient adaptées au même espace mixte discret/continu — le graphique ci-dessus permet de lire
lequel converge le plus vite *sur ce problème précis*, sans généraliser au-delà. Aucun des deux
algorithmes ne domine structurellement l'autre sur toute la classe des problèmes télécom : la
lecture détaillée des courbes, et le pourquoi de l'écart observé, sont développés à l'oral.

### 8.7 Discussion — Q4 : choix d'un point sur le front de Pareto QoS (Mod-D × Mod-F)

`select_best_compromise` (Pb2) normalise chaque objectif sur [0,1] et choisit le point le plus
proche du point idéal — un critère de distance à l'utopie, directement transposable au front
(MOS moyen, bande totale) de Pb3. Une alternative légitime est une pondération lexicographique :
satisfaire d'abord la contrainte `min_mos` de chaque profil VoIP (priorité absolue), puis maximiser
le MOS moyen résiduel sous la bande restante — un choix plus proche d'un SLA opérateur réel que la
distance euclidienne pure. Le compromis entre ces deux critères, et ce qu'il implique pour un
opérateur, est un point à trancher à l'oral plutôt qu'ici.

### 8.8 Complément — ABC et CMA-ES, deux algorithmes évolutionnaires bonus

**Motivation** : le sujet propose un bonus (+5 pts max) pour l'implémentation d'un algorithme
évolutionnaire supplémentaire (CMA-ES, SPEA2 ou MOEA/D — ce dernier déjà couvert par le Pb2, §8.3).
Au-delà du bonus, l'objectif était de démontrer une compréhension des mécanismes évolutionnaires
qui ne se limite pas à savoir appeler DEAP/pymoo/scipy sur un problème donné : deux algorithmes
**délibérément contrastés** ont été ajoutés, chacun implémenté from scratch en Python/NumPy pur
(même démarche que l'AG from-scratch de §8.1, aucune nouvelle dépendance), pour tester la
compréhension des opérateurs eux-mêmes plutôt que celle d'une API de librairie :

- **ABC (Artificial Bee Colony, Karaboga 2005)** — **aucun croisement, aucune mutation au sens
  AG**. Son mouvement d'exploration est une perturbation par différence de voisin sur une seule
  dimension (abeilles employées/spectatrices), sa diversité vient d'abeilles éclaireuses qui
  abandonnent purement une source de nourriture stagnante pour un tirage aléatoire frais. Choisi
  pour vérifier concrètement ce qu'une métaheuristique à population peut accomplir *sans* les deux
  opérateurs GA classiques.
- **CMA-ES (Covariance Matrix Adaptation Evolution Strategy, formulation standard de Hansen)** —
  le choix **piloté par l'élite** : sa mise à jour de moyenne/covariance à chaque génération est une
  recombinaison pondérée des seuls top-μ individus élites. L'élitisme n'y est pas une garde-fou
  ajoutée après coup (comme la fraction élite copiée telle quelle en §8.1) mais le mécanisme complet
  qui pilote à la fois le déplacement de la recherche et l'adaptation de sa propre distribution de
  mutation (la matrice de covariance elle-même est apprise à partir des individus élites qui ont
  réussi) — contraste direct avec la décroissance exponentielle de sigma codée en dur en §8.1.

**Décisions de conception tranchées explicitement avant l'implémentation** (détail complet dans
`REPORT.md`) : CMA-ES est nativement non contraint (échantillonnage gaussien pouvant sortir des
bornes) — les points recadrés dans les bornes déclenchent une **resynchronisation** des vecteurs
internes de l'algorithme (`_sample_and_clip`), pour que son modèle interne ne dérive jamais de ce
qui a réellement été évalué, plutôt qu'un recadrage « pour la forme » qui aurait laissé son état
interne référencer un point jamais vraiment testé — ce choix compte d'autant plus sur un gène étroit
comme `plc_level ∈ [0,1]` du Pb1, où le recadrage est fréquent. Arrêt à budget fixe uniquement (pas
de critère de convergence anticipée), pour rester comparable aux autres algorithmes du Module F.

**Validation Rastrigin 2D** (seed=0, budget ≈2500 évaluations, mêmes conditions que §8.1) :

| Algorithme | Meilleur `Rastrigin(x)` | Évaluations |
|---|---|---|
| ABC | 3,4 × 10⁻⁷ | 2525 |
| CMA-ES | 0,349 | 2500 |

Les deux convergent nettement mieux que l'AG from-scratch sur ce même budget (§8.1, `≈4,5`) — à
prendre avec prudence : ce n'est pas une preuve de supériorité générale (No Free Lunch, §8.6), plus
un signe que Rastrigin 2D est un terrain favorable à une recherche guidée par covariance (CMA-ES)
ou par perturbation locale à échappement (ABC) qu'à la mutation à taux fixe de l'AG maison.

**Câblés comme solveurs bonus du Pb1** (`cma_es_codec`/`abc_codec`, non mandatés — Pb1 n'impose que
« AG via DEAP », les deux nouveaux algorithmes n'entrent pas en conflit avec cette exigence) :

![Pb1 codec : AG vs random search vs CMA-ES vs ABC](results/module_f/pb1_codec_convergence_bonus_algos.png)

Budget partagé de 300-325 évaluations (`seed=0`) : les quatre convergent vers le même optimum
(`codec_fitness ≈ 2,489`), l'AG DEAP et CMA-ES quasi optimaux dès les premières évaluations
(population initiale déjà proche du plateau), ABC partant plus bas mais rattrapant le groupe en
moins de 25 évaluations. Sur ce problème à 4 gènes, peu dimensionné, les quatre stratégies de
recherche se valent largement — un résultat cohérent avec le fait que Pb1 est le problème le plus
simple des trois (comparé à Pb2, 3 objectifs, ou Pb3, 20 gènes mixtes discret/continu).

---

## 9. Conclusion et perspectives

Les six modules sont fonctionnellement complets, testés (511 tests, 97 % de couverture) et validés
contre au moins une vraie API/mesure chaque fois qu'un module en dépend (Twilio, OpenCelliD +
Nominatim/Overpass + ipinfo.io, STUN, Whisper). Le fil rouge sim-vs-réel s'est vérifié empiriquement
à trois reprises indépendantes (Module A : Opus WER=1.0 malgré le meilleur SNR/PESQ ; Module B :
restrictions de compte trial absentes du simulateur ; Module D : ordre MOS prédit conforme à une
vraie mesure réseau) — sans que ces écarts aient été construits a priori.

**Limites connues, documentées mais non corrigées** : taux de perte de paquets en Module A calibré
empiriquement (1 %) pour laisser le signal du bitrate observable sans le noyer ; profils
utilisateurs QoS (Module D) et coût BTS proportionnel à la distance à l'infrastructure existante
(Module C, faute de données routières) sont des proxys documentés, pas des valeurs mesurées.

**Pistes non explorées** : PESQ complet (licence non disponible pour ce projet), remplacement de la
géométrie 5G NR pour Module C/D, SPEA2 ou une variante active de CMA-ES (poids négatifs) en
complément de NSGA-II/MOEA-D/CMA-ES (§8.8), apprentissage par renforcement pour l'allocation QoS
dynamique (Pb3) en remplacement d'une optimisation statique par génération.

---

## 10. Annexes

### 10.1 Extraits de code commentés

**Empaquetage 7 bits GSM (Module B, `pdu.py`)** — vérifié à la main pour `"Hi"` :

```python
value = 0
for i, septet in enumerate(septets):
    value |= septet << (7 * i)
# LSB-first : identique à un déballage manuel bit à bit, plus court
return value.to_bytes((len(septets) * 7 + 7) // 8, "little")
```

**Point de référence de l'hypervolume (Module F, `pb2_bts.py`)** — marge additive, pas
multiplicative (un nadir négatif × 1.1 le rendrait *meilleur*, pas pire) :

```python
ref_point = nadir + 0.1 * np.abs(nadir)
```

**Découpage Sutherland-Hodgman d'une cellule Voronoï non bornée (Module C, `cell_id.py`)** — 4
points fictifs hors bounding box forcent toutes les cellules réelles à devenir finies avant
découpage à la bbox réelle :

```python
phantom = far_points_outside(bbox, margin=10 * bbox_diagonal)
vor = Voronoi(np.vstack([real_points, phantom]))
```

### 10.2 Fronts de Pareto complets et cartes Folium

- `results/module_f/pb2_bts_map.html` — meilleur compromis Pb2 sur carte interactive.
- `results/module_c_demo.html` — position estimée + réelle + POI + incertitude.
- Front de Pareto Pb2 complet (N=10, NSGA-II vs. MOEA/D) : voir Figure §8.3.
