# mobi-sim

Projet étudiant (MSc Télécommunications) : MobiSim, une plateforme hybride de services de
communication mobile combinant simulation logicielle pure, appels à de vraies APIs gratuites, et
optimisation par algorithmes évolutionnaires, sur six modules (codecs audio, SMS, géolocalisation,
QoS, algorithmique évolutionnaire, API REST).

Ce guide explique comment faire tourner le projet en partant d'un clone/fork tout neuf, sans rien
supposer de déjà configuré.

## Prérequis

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (gestionnaire de dépendances/environnement)

## 1. Installer les dépendances

```bash
git clone <url-de-votre-fork>
cd mobi-sim
uv sync
```

`uv sync` crée un environnement virtuel (`.venv/`) et installe toutes les dépendances épinglées
(`uv.lock`), y compris les libs d'algorithmique évolutionnaire (DEAP, pymoo, pyswarm) et l'API
(FastAPI, SQLAlchemy, python-jose, etc.).

## 2. Configurer les clés API (`.env`)

```bash
cp .env.example .env
```

Puis remplissez `.env` avec vos propres clés — toutes gratuites, sans carte bancaire :

| Variable | Où l'obtenir | Utilisée par |
|---|---|---|
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER` | Compte trial sur [twilio.com/try-twilio](https://twilio.com/try-twilio) | Module B (SMS réel) |
| `TWILIO_TEST_TO_NUMBER` | Votre propre numéro mobile, vérifié dans la console Twilio (trial oblige) | `module_b/manual_twilio_check.py` |
| `TWILIO_VERIFY_SERVICE_SID` | Créé une fois via `client.verify.v2.services.create(...)` dans la console Twilio | `module_b/manual_twilio_verify_check.py` |
| `OPENAI_API_KEY` | Crédit d'essai sur [platform.openai.com](https://platform.openai.com/signup) | Module A (transcription Whisper, ≈0,006 $/min) |
| `OPENCELLID_API_KEY` | Inscription gratuite sur [opencellid.org](https://opencellid.org/register) | Module C (uniquement si vous utilisez l'API live plutôt que le dump offline, voir §3) |
| `IPINFO_TOKEN` | Inscription gratuite sur [ipinfo.io](https://ipinfo.io/signup) (50k req/mois) | Module C (géoloc IP) |
| `JWT_SECRET_KEY`, `JWT_ALGORITHM` | À générer vous-même (ex. `openssl rand -hex 32`) ; `HS256` par défaut | Module E (JWT) |

Aucune de ces clés n'est nécessaire pour lancer la suite de tests dans son ensemble (tout est
mocké dans `pytest`) — elles ne servent qu'aux scripts de validation « réelle » (§4) et à l'usage
courant de l'API (§5).

## 3. Données réelles requises — dump OpenCelliD (Module C, Module E, Module F)

Le dossier `docs/` est volontairement absent d'un clone tout neuf (gitignoré : PDF du sujet et
dumps de données, jamais commités). Or `module_c/opencellid_loader.py` attend un extrait BTS
OpenCelliD à `docs/208.csv` (France, MCC 208), utilisé par :

- `module_c.fitness._get_default_terrain()` (terrain de démonstration par défaut),
- les endpoints `/location/*` de Module E (`module_e/tests/test_location.py` en dépend aussi),
- la démo de placement BTS de Module F (`pb2_bts.py`).

Sans ce fichier, `uv run pytest module_e` échoue et les endpoints `/location/*` (méthodes
`cell_id`/`toa`/`wifi`) plantent. **Les tests unitaires de `module_c` et `module_f` eux-mêmes n'en
ont pas besoin** (ils tournent sur un petit terrain synthétique généré à la volée).

Pour l'obtenir :

1. Créez un compte gratuit sur [opencellid.org](https://opencellid.org/register).
2. Téléchargez l'export CSV par pays (page *Downloads*), MCC **208** (France) — ou un autre pays de
   votre choix, à condition d'ajuster `mcc`/`AVEYRON_BBOX` dans `module_c/opencellid_loader.py`.
3. Décompressez-le et placez-le à `docs/208.csv` (créez le dossier `docs/` s'il n'existe pas) :

```bash
mkdir -p docs
mv ~/Downloads/208.csv docs/208.csv
```

## 4. Lancer les tests

```bash
uv run pytest                                   # les 461 tests, tous modules
uv run pytest module_a -v                       # un seul module
uv run pytest --cov --cov-report=html --cov-report=term-missing   # couverture (objectif ≥75%, rapport dans htmlcov/)
```

Chaque module (`module_a` à `module_f`) a sa propre suite (`test_module_*.py`, ou `tests/` pour
Module E). Tout ce qui touche une API externe y est mocké — la suite complète tourne hors ligne
(à l'exception de `module_e`/`module_f` qui lisent `docs/208.csv`, cf. §3).

## 5. Valider les intégrations réelles (scripts manuels)

Chaque module qui appelle une vraie API a un script `manual_*_check.py`, volontairement **exclu**
de `pytest` (il dépense un vrai quota/crédit ou envoie un vrai SMS) — à lancer à la main :

| Commande | Ce qu'elle fait | Clés `.env` requises |
|---|---|---|
| `uv run python -m module_a.manual_whisper_check` | Synthétise un signal, le dégrade (AAC/GSM/Opus), transcrit chaque version via la vraie API Whisper | `OPENAI_API_KEY` |
| `uv run python -m module_b.manual_twilio_check` | Envoie un vrai SMS via l'API Messages Twilio et mesure la latence de livraison | `TWILIO_*`, `TWILIO_TEST_TO_NUMBER` |
| `uv run python -m module_b.manual_twilio_verify_check` | Démarre un OTP réel (sans argument), puis le vérifie (`... <code>`) | `TWILIO_*`, `TWILIO_VERIFY_SERVICE_SID` |
| `uv run python -m module_c.manual_ipinfo_check` | Géolocalise l'IP publique de la machine via ipinfo.io | `IPINFO_TOKEN` |
| `uv run python -m module_c.manual_nominatim_check` | Requête Overpass réelle (POI autour d'un point) | aucune (API publique) |
| `uv run python -m module_d.manual_stun_check` | 20 vraies mesures RTT/gigue/perte contre `stun.l.google.com:19302` | aucune (API publique) |

## 6. Lancer l'API (Module E)

```bash
uv run uvicorn module_e.main:app --reload
```

Swagger UI : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). La base SQLite
(`module_e.db`, gitignorée) est créée et peuplée automatiquement (20 services de démo) au premier
démarrage.

Flux d'authentification OAuth2 à scopes (`admin`/`operator`/`user`) — créer un compte puis obtenir
un token :

```bash
curl -X POST http://127.0.0.1:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "demo", "password": "un-mot-de-passe-solide", "scope": "user"}'

curl -X POST http://127.0.0.1:8000/auth/token \
  -d "username=demo&password=un-mot-de-passe-solide"
```

Le `access_token` renvoyé se passe en `Authorization: Bearer <token>` sur toutes les autres routes
(`/catalog`, `/codecs`, `/sms`, `/location`, `/qos`, `/optimize`).

## 7. Générer des figures / résultats

`module_x/visualize.py` (et `module_f/visualize.py`) n'exposent que des fonctions pures qui
retournent une `Figure`/une `Map` déjà construite — aucune n'écrit de fichier toute seule ni n'a de
point d'entrée `__main__`. À appeler depuis un script ou un REPL, puis sauvegarder via
`save_figure`/`save_map_html` (écrit dans `results/`, gitignoré) :

```python
from module_a import synth_audio, codecs, metrics, visualize

signal, sr = synth_audio.generate_reference_signal()
degraded = codecs.simulate_codec(signal, sr, "opus", seed=42)
fig = visualize.plot_metric_bar({"opus": metrics.snr_db(signal, degraded)}, "SNR (dB)", "SNR par codec")
visualize.save_figure(fig, "snr_opus.png")
```

## Structure du projet

| Module | Domaine | Contrat exposé à Module F |
|---|---|---|
| `module_a/` | Codecs audio & qualité perceptive (AAC/GSM/Opus, PESQ simplifié, WER via Whisper, MUSHRA) | `codec_fitness(chromosome)` |
| `module_b/` | SMS hybride (SMSC/HLR/VLR simulés + Twilio réel) | `routing_fitness(chromosome)` |
| `module_c/` | Géolocalisation LBS (Cell-ID, TOA, Wi-Fi fingerprinting, IP) sur données OpenCelliD réelles | `bts_coverage_fitness(chromosome)` |
| `module_d/` | QoS/QoE (modèle E ITU-T G.107, mesures STUN réelles) | `qos_fitness(chromosome)` |
| `module_e/` | Passerelle API REST FastAPI (JWT, catalogue SQLite, rate limiting, `/optimize/*`) | — (consomme A–D et F) |
| `module_f/` | Algorithmique évolutionnaire (AG/DEAP, NSGA-II/MOEA-D via pymoo, DE/scipy, PSO/pyswarm) appliquée aux fitness de A/C/D | — |

Le détail des choix d'architecture et des résultats de test est dans `REPORT.md` (journal de
construction, en français, par fichier).
