# MobiSim — Rapport de construction

Journal continu des choix d'architecture, des justifications et des résultats de tests, mis à jour
au fur et à mesure de l'implémentation. Voir [docs/SUJET.md](docs/SUJET.md) pour le sujet complet
et [CLAUDE.md](CLAUDE.md) pour le résumé de l'architecture cible. Ce fichier est le compagnon
« pourquoi » du code : le code reste sans commentaires là où le raisonnement est non-évident ;
ce raisonnement vit ici à la place.

---

## Module A — Codecs audio & qualité perceptive

### `synth_audio.py`

**Choix** : le signal de parole de référence est synthétisé (gTTS en ligne, repli pyttsx3 hors
ligne), jamais enregistré — aucune dépendance microphone, entièrement reproductible, généré en
mémoire (`io.BytesIO`, aucune écriture disque). Normalisé à **8 kHz mono**, crête normalisée à 1.0.

**Pourquoi 8 kHz** : c'est la fréquence d'échantillonnage de la téléphonie bande étroite
(narrowband), pas une valeur par défaut arbitraire.
- Le Nyquist (4 kHz) couvre exactement la bande vocale téléphonique classique 300–3400 Hz ; GSM
  Full Rate est un vrai codec narrowband à 8 kHz, donc le simuler à 8 kHz est correct, pas
  approximatif.
- Le mode narrowband de l'ITU-T P.862 (PESQ) est défini à 8 kHz — garde notre PESQ simplifié
  comparable en esprit au standard.
- Les scores MUSHRA de référence du rapport fourni (Opus 57.7/61.4, GSM 48.0/51.4, AAC 34.9/36.2)
  proviennent d'une expérience en contexte téléphonie narrowband (AAC@16kbps, GSM-FR) — rester à
  8 kHz garde notre pipeline dans le même régime que les chiffres auxquels on se corrèle.
- Moins coûteux : tableaux plus petits, filtrage plus rapide, transcription Whisper moins chère
  (facturée à la minute).

Conséquence : Opus est volontairement contraint dans une boîte narrowband très en dessous de ce
qu'il sait réellement faire (wideband/HD) — c'est intentionnel, c'est ce qui rend la comparaison
AAC/GSM/Opus équitable (apples-to-apples).

### `codecs.py`

**Choix** : trois simulateurs de dégradation (`simulate_aac`, `simulate_gsm`, `simulate_opus`),
chacun `(signal, sr, target_snr_db=..., seed=...) -> signal_dégradé`, plus un dispatcher
`simulate_codec(signal, sr, codec, **kwargs)`. Aucune librairie de codec propriétaire — les
artefacts sont modélisés à la main selon la signature perceptive connue de chaque vrai codec :

| Codec | Débit | Modèle |
|---|---|---|
| AAC | 16 kbps | Passe-bas (Butterworth, coupure ~3.8 kHz) + lissage **pré-écho** sur les transitoires détectés |
| GSM-FR | 13 kbps | Passe-bande (300–3400 Hz) + bruit gaussien additif au SNR cible |
| Opus | 24 kbps | Aucune limitation de bande (quasi-transparent) + légère distorsion harmonique cubique + bruit faible niveau |

**Pourquoi du bruit additif, pas seulement du filtrage** : le bruit modélise le *bruit de
quantification*, ce qu'un vrai codec avec pertes introduit réellement en quantifiant plus
grossièrement pour atteindre un débit cible. Un filtre fixe seul ne donne qu'une différence binaire
« limité en bande ou non » ; calibrer le bruit sur un `target_snr_db` donne un curseur de sévérité
continu et ajustable — nécessaire plus tard pour que le Pb1 du `module_f` (AG de configuration
codec) dispose d'un paysage de fitness lisse à optimiser plutôt que d'une fonction en marches
d'escalier.

**Pourquoi le pré-écho, AAC uniquement** : explicitement demandé par le sujet (« filtre passe-bas +
artefacts de pré-écho via transitoires gaussiens »). Le pré-écho est un artefact réel et documenté
des codecs par transformée en blocs (AAC, MP3) : le bruit de quantification est calculé par bloc
d'analyse, donc un transitoire net partageant un bloc avec du contenu calme fait fuir du bruit
*en arrière* dans le temps, audible juste avant l'attaque réelle. `_add_pre_echo` détecte les
transitoires via des sauts d'énergie court-terme et injecte une rampe de bruit décroissante avant
chacun d'eux. GSM (vocodeur, pas de ce mode de défaillance par transformée en blocs) et Opus
(transparent à ce débit) n'en reçoivent pas.

**Note de conception — enjeu narratif** : l'énigme du rapport fourni est que le PESQ-NB note Opus
(1.52) et GSM (1.53) de façon quasi identique alors que MUSHRA et WER les séparent nettement.
Modéliser la dégradation GSM comme du bruit large-bande plat (que pénalise proportionnellement une
métrique de corrélation spectrale) vs. la dégradation AAC comme un pré-écho localisé dans le temps
(que remarque à peine une corrélation globale sur tout le signal, mais que capte une oreille
humaine ou une transcription ASR) est ce qui devrait permettre à `metrics.py` et `whisper_eval.py`
de reproduire ce même écart plus tard, plutôt que d'avoir à le simuler artificiellement dans
`visualize.py`.

**Résultats de test** (tonalité de test synthétique, 440+1200+3000 Hz + transitoires artificiels
injectés, 8 kHz, seed=42 — pas de la vraie parole, juste un test de fumée vérifiant que le pipeline
tourne et produit des valeurs cohérentes) :

| Codec | crête | NaN ? |
|---|---|---|
| AAC | 0.794 (crête alignée sur la référence) | Non |
| GSM | 0.794 | Non |
| Opus | 0.794 | Non |

Dispatcher (`simulate_codec`) vérifié contre les appels directs ; un nom de codec inconnu lève une
`ValueError` listant les options valides.

Remarque de lint connue : le paramètre `sr` de `simulate_opus` n'est pas utilisé (Opus n'applique
aucun filtrage) — conservé pour l'uniformité de signature entre les entrées de `SIMULATORS`, afin
que le dispatcher puisse appeler les trois de façon identique.

### `metrics.py`

**Choix** : trois métriques objectives, toutes `(reference, degraded[, sr]) -> float` :
- `snr_db` — SNR dans le domaine temporel.
- `log_spectral_distortion` — LSD moyennée par trame entre spectrogrammes d'amplitude
  (`scipy.signal.stft`).
- `pesq_nb_simplified` — **pas** l'ITU-T P.862, aucune librairie PESQ propriétaire (conformément au
  sujet). Construit comme
  `spectral_weight · corrélation_spectrale + (1 − spectral_weight) · corrélation_enveloppe_temporelle`
  (les deux via `scipy.stats.pearsonr`), remis à l'échelle de `[-1, 1]` vers la plage MOS-like PESQ
  conventionnelle `[1.0, 4.5]`. `spectral_weight = 0.6` par défaut.
- `all_metrics` regroupe les trois dans un dict pour la commodité.

**Résultats de test** (même tonalité synthétique + transitoires artificiels que ci-dessus, 8 kHz,
seed=42) :

| Codec | SNR (dB) ↑ | LSD ↓ | PESQ-NB (sim) ↑ |
|---|---|---|---|
| AAC | 25.8 | 23.1 | 4.47 |
| GSM | 13.8 | 30.9 | 3.93 |
| Opus | 35.1 | 13.8 | 4.50 |

Aucun NaN, aucun crash sur les trois codecs.

**Observation ouverte à revérifier avec de la vraie parole** : sur cette tonalité synthétique, le
PESQ simplifié classe AAC *au-dessus* de GSM — l'inverse de l'ordre MUSHRA du rapport fourni (AAC
noté le plus mauvais globalement, 34.9/36.2). Ce n'est pas traité comme un bug : une métrique de
corrélation naïve pénalise à peine le lissage pré-écho localisé de l'AAC, de la même façon que le
vrai PESQ-NB est rapporté comme sous-pénalisant certains artefacts par rapport aux jugements
humains/MUSHRA et WER (le sujet demande explicitement d'analyser cet écart dans le rapport, §Q1).
À revérifier une fois exécuté sur de la vraie parole TTS (vrais transitoires, pas une tonalité
pure) plutôt que supposé valide — noté ici pour ne pas l'oublier avant la matrice de corrélation
de `visualize.py`.

### `whisper_eval.py`

**Choix** : trois fonctions publiques —
- `transcribe(signal, sample_rate, client=None)` — encode le signal en WAV en mémoire (réutilise
  `signal_to_wav_bytes` de `synth_audio.py`) et appelle
  `client.audio.transcriptions.create(model="whisper-1", file=(...))`. `client` est injectable
  (facilite les tests sans clé API réelle et le partage d'un seul client sur plusieurs appels).
- `word_error_rate(reference_text, hypothesis_text)` — WER via `jiwer.wer`.
- `evaluate_intelligibility` / `evaluate_codecs_intelligibility` — enchaînent transcription + WER
  pour un signal, ou pour un dict `{codec: signal_dégradé}` en réutilisant un seul client.

**Pourquoi normaliser le texte avant le calcul du WER** : `jiwer.wer` brut compare mot à mot sans
normalisation — hors Whisper renvoie systématiquement un texte avec majuscules et ponctuation
(ex. « The quick brown fox. ») alors que notre texte de référence n'en a pas forcément la même
forme exacte. Sans normalisation, une transcription parfaite au niveau intelligibilité serait quand
même comptée en erreur à cause de la casse/ponctuation — ce n'est pas ce qu'on veut mesurer (on
veut l'intelligibilité du contenu, pas le style de formatage de Whisper). `_WER_NORMALIZE`
(`jiwer.Compose` : minuscules, suppression ponctuation, espaces multiples, strip) est appliqué aux
deux côtés avant comparaison.

**Test réalisé** (sans clé API — un client Whisper factice est injecté pour vérifier le câblage
sans dépenser de crédit) :
- `word_error_rate` : texte identique casse/ponctuation différente → `0.0` ; une vraie substitution
  de mot (« fox » → « fax ») → `0.25`, comme attendu.
- `transcribe` avec un client factice retournant `" The quick brown fox. "` → texte bien strippé,
  format `(filename, bytes, content_type)` du fichier envoyé validé (WAV en mémoire, >44 octets
  d'en-tête).
- `evaluate_codecs_intelligibility` sur un dict `{aac, gsm, opus}` → bien un dict de résultats par
  codec, chacun avec `transcription` et `wer` à `0.0` (texte de référence identique après
  normalisation).

Testé une première fois sans clé API (client factice, cf. ci-dessus), puis contre la vraie API
Whisper via `manual_whisper_check.py` — voir la section dédiée ci-dessous pour le résultat.

### `test_module_a.py`

**Décision** : plutôt que de repousser les tests à la fin du module (l'ordre du sujet §3 les liste
en dernier), on les écrit dès que 4 fichiers existent, conformément à la nouvelle section « Git
handling »/pratique d'ingénieur — détecter les régressions tôt plutôt qu'après avoir accumulé du
code non testé sur `mushra_sim.py`/`visualize.py`/`fitness.py`.

**Choix** : tests unitaires purs, aucun accès réseau ni clé API requis pour lancer la suite.
- `synth_audio.py` : `_synth_via_gtts`/`_synth_via_pyttsx3` sont monkeypatchées (jamais de vrai
  appel réseau/moteur TTS) ; `_resample`, `_wav_bytes_to_mono_float` (mono et stéréo) et
  `signal_to_wav_bytes` sont testées directement sur des tableaux synthétiques.
- `codecs.py` : forme/dtype/finitude de sortie, crête alignée sur la référence, reproductibilité
  par seed, dispatcher insensible à la casse, codec inconnu → `ValueError`.
- `metrics.py` : cas limites (signal identique → SNR infini, PESQ au maximum ; silence et signal
  très court → pas de NaN).
- `whisper_eval.py` : un client OpenAI factice (classe Python simple, pas de mock réseau) vérifie
  le format du fichier envoyé (`(nom, bytes, content-type)`, en-tête WAV) et le câblage
  transcription→WER ; `_client()` testé avec une clé factice (aucun appel réseau déclenché par le
  simple constructeur `OpenAI(api_key=...)`).

**Bugs trouvés et corrigés en écrivant les tests** (aucun des deux ne s'était manifesté sur la
tonalité de test de 1 seconde utilisée jusqu'ici — seulement sur des signaux volontairement très
courts) :

1. **`codecs._add_pre_echo`** : la fenêtre de lissage d'énergie (20 ms → 160 échantillons à 8 kHz)
   pouvait dépasser la longueur du signal ; `np.convolve(..., mode="same")` renvoyait alors un
   tableau *plus long* que le signal, produisant des indices de transitoire hors bornes et un
   crash dans `simulate_aac`. Corrigé en bornant la fenêtre à `len(x)`.
2. **`metrics._magnitude_spectrogram`** : `noverlap` était calculé à partir de la fenêtre de 32 ms
   demandée, mais `scipy.signal.stft` réduit silencieusement `nperseg` à la longueur du signal
   pour les entrées courtes sans toucher à `noverlap` — d'où `noverlap >= nperseg` et une
   `ValueError`. Corrigé en bornant les deux valeurs nous-mêmes de façon cohérente.

**Nettoyage associé** : `_butter_filter` avait deux branches (passe-haut seul, passe-tout) qu'aucun
appelant n'utilise (AAC ne passe jamais que `high`, GSM passe toujours `low` et `high`) — simplifié
plutôt que testé, ce code mort n'avait pas lieu d'être maintenu.

**Résultat** : 39 tests, tous verts. Couverture `module_a` : **94 %** (objectif CLAUDE.md : ≥75 %).
Seules zones non couvertes, délibérément : les corps réels de `_synth_via_gtts`/`_synth_via_pyttsx3`
(appels réseau/moteur TTS véritables, hors périmètre des tests unitaires) et une branche de garde
dans `_add_pre_echo` (`idx == 0`) quasiment inatteignable en pratique (le premier élément de
`d_energy` vaut toujours 0 par construction).

### `mushra_sim.py`

**Contexte** : le rapport de référence ne publie que les scores MUSHRA *moyens* par groupe
d'auditeurs (English speakers, Native speakers) et par codec — pas les notes individuelles brutes.
Pour pouvoir calculer un intervalle de confiance ou une corrélation avec PESQ/WER, il faut simuler
un panel plausible plutôt que travailler sur trois nombres isolés.

**Choix** :
- `simulate_panel_ratings(mean, std=15.0, n_listeners=20, seed=None)` — tire les notes d'un panel
  via une loi normale centrée sur la moyenne rapportée, **bornée à [0, 100]** (l'échelle MUSHRA est
  bornée, contrairement à la loi normale — sans ce clip, un tirage pourrait produire une note
  invalide comme 105 ou -3).
- `simulate_mushra_panel(means=REPORTED_MEANS, ...)` — génère les 6 panels (3 codecs × 2 groupes)
  d'un coup, chacun sur son propre flux RNG dérivé de `seed` pour rester reproductible sans
  partager le même bruit d'échantillonnage entre codecs.
- `bootstrap_ci(ratings, n_bootstrap=2000, ci=0.95, seed=None)` — IC 95% par **bootstrap** (tirage
  avec remise des notes simulées, calcul de la moyenne à chaque tirage, percentile 2.5/97.5),
  conformément au sujet — pas un IC gaussien classique (`mean ± 1.96·SE`), qui supposerait une
  distribution non bornée et serait moins fidèle avec un échantillon de panel de petite taille.
- `summarize_mushra(...)` — combine les deux pour produire `{codec: {groupe: {mean, ci_low,
  ci_high}}}`, prêt pour `visualize.py`.

**Choix assumé, à documenter dans le rapport final** : l'écart-type inter-auditeur (`DEFAULT_STD =
15.0`) et la taille de panel (`DEFAULT_N_LISTENERS = 20`) sont des hypothèses plausibles (échelle
MUSHRA 0-100, panels typiques de l'ordre de 10-30 auditeurs), pas des valeurs tirées du rapport
fourni puisque celui-ci ne donne que les moyennes. Repère utile : docs/SUJET §Q1 demande déjà
d'analyser l'écart PESQ/WER vs MUSHRA — le choix de std impactera la largeur des IC affichés,
donc à mentionner explicitement comme hypothèse de simulation dans le rapport final, pas comme un
fait mesuré.

**Hors périmètre, volontairement** : pas de référence cachée ni d'ancrage passe-bas (méthodologie
MUSHRA complète, norme ITU-R BS.1534) — le sujet ne demande de reproduire que les 3 vrais codecs
du rapport fourni, pas le protocole MUSHRA complet.

**Résultats de test** (`simulate_mushra_panel(seed=0)` puis `summarize_mushra(seed=0)`, 20
auditeurs/groupe) :

| Codec | Groupe | Moyenne simulée | IC 95% (bootstrap) |
|---|---|---|---|
| Opus | English | 55.0 | [49.1, 60.2] |
| Opus | Native | 62.0 | [58.1, 65.8] |
| GSM | English | 47.3 | [41.2, 53.1] |
| GSM | Native | 49.4 | [41.5, 57.4] |
| AAC | English | 34.4 | [26.4, 42.0] |
| AAC | Native | 33.3 | [27.3, 39.5] |

L'ordre Opus > GSM > AAC du rapport est bien préservé dans les deux groupes malgré le bruit de
simulation. 47 tests au total (8 nouveaux pour ce fichier), tous verts ; couverture `module_a` :
**95 %**, `mushra_sim.py` à 100 %.

### `visualize.py`

**Choix** : uniquement des fonctions qui construisent et **retournent** une `Figure` matplotlib à
partir de données déjà calculées (par `metrics.py`, `whisper_eval.py`, `mushra_sim.py`) — aucun
I/O, aucun `plt.show()`. Charge à l'appelant (script d'orchestration à venir, ou un futur endpoint
`module_e`) de l'afficher, l'embarquer, ou l'enregistrer. Ce découplage est ce qui rend le module
testable en headless (backend `Agg` dans les tests) sans jamais faire apparaître de fenêtre.

- `plot_mushra_comparison(summary)` — barres groupées (par codec × groupe English/Native), barres
  d'erreur = IC 95% bootstrap de `mushra_sim.summarize_mushra`.
- `plot_metric_bar(values, ylabel, title)` — un histogramme générique réutilisé pour PESQ et pour
  WER plutôt que deux fonctions quasi identiques.
- `mushra_grand_mean(summary)` — réduit les deux groupes MUSHRA à une seule moyenne par codec, pour
  pouvoir aligner MUSHRA avec PESQ/WER (scalaires) dans une même matrice de corrélation.
- `build_correlation_table(pesq_by_codec, wer_by_codec, mushra_summary)` — assemble
  `{codec: {pesq, wer, mushra}}`.
- `plot_correlation_matrix(metrics_by_label)` — heatmap de corrélation de Pearson
  (`np.corrcoef`), annotée. Volontairement générique sur la clé (« label ») : avec seulement 3
  codecs aujourd'hui, la matrice a 3 points par variable — statistiquement faible (df=1), c'est
  justement l'un des points à discuter dans le rapport final (§Q1 du sujet) ; mais la même fonction
  pourra plus tard prendre en entrée de nombreuses évaluations du chromosome codec du Module F
  (Pb1), avec beaucoup plus de points, sans rien changer au code.
- `save_figure(fig, filename, output_dir="results")` — écrit un PNG dans `results/` (répertoire
  gitignoré, cf. `.gitignore` : sortie régénérable, pas versionnée).

**Point de robustesse testé** : si une métrique est constante entre les codecs (variance nulle,
ex. un WER identique partout), `np.corrcoef` renvoie `NaN` pour les corrélations impliquant cette
métrique — géré en affichant `"n/a"` dans la case plutôt que de planter ou d'afficher `nan`.

**Résultats de test** : 55 tests au total (8 nouveaux pour ce fichier), tous verts ; couverture
`module_a` : **96 %**, `visualize.py` à 100%. Un avertissement matplotlib
(`set_xticklabels() should only be used with...`) est apparu pendant le développement dans
`plot_metric_bar` (labels posés sans `set_xticks()` préalable) — corrigé en fixant les ticks
numériques avant de poser les labels, comme le fait déjà `plot_mushra_comparison`.

### `fitness.py`

**Contexte** : dernier fichier du Module A. Expose le contrat `codec_fitness(chromosome)` que le
Module F (Pb1, AG via DEAP) doit pouvoir appeler tel quel. Chromosome imposé par le sujet (§3
Module F) : `[bitrate ∈ {8,12,16,24,32} kbps, frame_size ∈ {10,20,30,40} ms, plc_level ∈ [0,1],
codec ∈ {0=AAC, 1=GSM, 2=Opus}]`, fitness `f(x) = w1·PESQ_sim(x) + w2·(1−WER(x)) −
w3·(bitrate/bitrate_max)`, poids configurables, contrainte `bitrate ≤ B_max` gérée par pénalité.

**Décision structurante — signature à un seul argument** : le sujet écrit littéralement
`codec_fitness(chromosome)`, pas `codec_fitness(chromosome, reference_signal, ...)`. Comme un AG
peut appeler cette fonction des milliers de fois par run, régénérer le signal de référence (appel
réseau gTTS) à chaque évaluation serait à la fois ruineux en temps et rendrait la fitness non
reproductible d'un run à l'autre. Solution : un cache module-level (`_get_reference_signal`)
génère le signal une seule fois par fréquence d'échantillonnage et le réutilise ensuite. La
fonction plus riche `codec_fitness_components(chromosome, reference_signal, **overrides)` reste
disponible pour les tests/le debug (elle prend le signal en paramètre explicite, pas de cache).
Conformément à CLAUDE.md (« garder la signature stable une fois que Module F en dépend »), cette
signature est actée maintenant, avant que Module F n'existe, pour ne pas avoir à la casser plus
tard.

**Décision — pas de vrai WER dans la boucle de l'AG** : la formule du sujet inclut `WER(x)`, mais
appeler la vraie API Whisper (whisper_eval.py) à chaque évaluation d'individu est incompatible
avec un AG (des milliers d'appels payants + latence réseau, cf. tableau des API externes de
CLAUDE.md). `estimate_wer_proxy(pesq_nb)` fournit à la place une estimation gratuite, déterministe
et heuristique (pas mesurée) : `wer ≈ (1 − pesq_normalisé)²`, décroissante, bornée à `[0,1]`,
calibrée uniquement sur les valeurs extrêmes de PESQ (0 à PESQ_MAX, 1 à PESQ_MIN). Un vrai
`wer_fn` (fermeture appelant `whisper_eval.evaluate_intelligibility`) reste injectable pour une
passe de validation ponctuelle (ex. sur les meilleurs individus finaux de l'AG) — c'est exactement
le fil rouge « sim vs. réel » que CLAUDE.md demande de garder vivant dans tout le projet.

**Décision — traduire bitrate/frame_size/plc_level en paramètres que codecs.py comprend** :
`codecs.py` ne connaît que `target_snr_db`/`cutoff_hz`, pas un « bitrate » à proprement parler.
- `bitrate` → interpolation linéaire vers un `target_snr_db` (10 dB à 8 kbps, 45 dB à 32 kbps,
  bornes choisies arbitrairement mais documentées ici, indépendantes du codec choisi — plus de
  bits égale toujours moins de bruit de quantification, quel que soit le codec).
- `frame_size` + `plc_level` → aucune notion de perte de paquets n'existe ailleurs dans le Module
  A (le modèle de dégradation est par échantillon, pas par paquet) ; `_apply_packet_loss` simule
  donc un modèle minimal : un taux de perte fixe (`packet_loss_rate`, pas un gène du chromosome)
  fait disparaître des trames de longueur `frame_size_ms`, remplacées par `plc_level` fois la
  dernière trame valide (dissimulation par répétition atténuée — `plc_level=1` répète la dernière
  trame à pleine puissance, `plc_level=0` laisse un silence brut).

**Constat empirique important, à documenter dans le rapport final** : en isolant l'effet du
bitrate (`packet_loss_rate=0`), le PESQ simplifié suit bien la SNR cible et la fitness par défaut
présente un **optimum intérieur réel** autour de 16-24 kbps (le gain de PESQ au-delà ne compense
plus la pénalité de bitrate) — exactement le genre de compromis qu'un AG doit pouvoir découvrir.
Mais avec le taux de perte de paquets initialement choisi (3 %), la perte d'une seule trame de
20 ms suffisait à faire chuter le PESQ autant que toute la plage de SNR testée (10 à 45 dB),
noyant complètement le signal du bitrate dans le bruit d'échantillonnage de l'AG (quelle trame
tombe, par hasard, dépend du seed bien plus que du bitrate choisi). `DEFAULT_PACKET_LOSS_RATE` a
donc été réduit à **1 %**, qui reste réaliste (cf. littérature G.107 : 1 % correspond à un
« bon réseau ») tout en laissant le signal du bitrate généralement observable — mais pas
systématiquement : sur 5 seeds testés à la main, 3 montrent l'optimum intérieur attendu, 2 restent
dominés par une perte de trame malchanceuse. Cette variance résiduelle n'est **pas corrigée
davantage ici** : c'est un vrai comportement stochastique de la simulation, pas un bug, et son
traitement (moyenner la fitness sur plusieurs seeds par individu, par exemple) relève du choix de
conception de l'AG lui-même dans le Module F, pas du contrat `fitness.py`. À mentionner
explicitement dans le rapport final comme limite connue et compromis assumé.

**Résultats de test** : 78 tests au total (31 nouveaux pour ce fichier), tous verts ; couverture
`module_a` : **97 %**, `fitness.py` à 100%.

### `manual_whisper_check.py`

**Contexte** : `test_module_a.py` mock volontairement le client OpenAI (cf. sa section ci-dessus) —
la suite pytest ne dépense donc jamais de crédit API réel. Il manquait encore une vérification
contre la vraie API Whisper, notée comme dette dans la section `whisper_eval.py` ci-dessus. Ce
script n'est pas un fichier du découpage `docs/SUJET.md` §3 : c'est un utilitaire manuel, à lancer
à la main (`uv run python -m module_a.manual_whisper_check`), pas via pytest — il dépense du crédit
réel à chaque exécution.

**Choix** : réutilise `synth_audio`/`codecs`/`metrics`/`whisper_eval` tels quels (aucune logique
dupliquée) sur la phrase de référence par défaut (quelques secondes d'audio, coût négligeable),
transcrit la référence propre puis chacun des trois signaux dégradés via un vrai client OpenAI, et
affiche métriques locales + WER + transcription côte à côte.

**Résultat obtenu (clé API réelle, deux exécutions identiques — `whisper-1` décode en greedy à
température 0, donc reproductible sur un même audio, pas du bruit d'échantillonnage)** :

| Codec | SNR (dB) | PESQ-NB (sim) | WER (réel) | Transcription |
|---|---|---|---|---|
| AAC | 21.4 | 4.50 | **0.000** | correcte, identique à la référence |
| GSM | 4.3 | 4.20 | **0.000** | correcte, identique à la référence |
| Opus | 35.0 | 4.50 | **1.000** | dérive en français + « lazy dog » → « AZ-Dog » |

**Constat important, à documenter dans le rapport final (§Q1)** : c'est l'**inverse** de l'hypothèse
de conception notée dans `codecs.py` ci-dessus (où l'écart PESQ/WER attendu venait du pré-écho AAC
sous-pénalisé par une métrique de corrélation globale). Ici, sur de la vraie parole synthétisée et
la vraie API Whisper, c'est **Opus** — le codec avec le meilleur SNR et le meilleur PESQ simplifié
des trois — qui échoue totalement à l'intelligibilité réelle, pendant qu'AAC et GSM (métriques
objectives moins bonnes) transcrivent parfaitement. Hypothèse non vérifiée plus avant ici : la
distorsion harmonique cubique légère d'Opus (`_add_harmonic_distortion`), bien qu'à peine visible
sur SNR/PESQ (corrélations globales insensibles à ce type d'artefact fin), semble suffire à faire
dériver Whisper hors distribution (changement de langue détectée, hallucination partielle) — un
exemple concret et mesuré de l'écart métrique-objective vs. intelligibilité réelle que le sujet
demande d'analyser, obtenu ici sans avoir eu à le construire artificiellement.

---

## Module B — SMS hybride (SMSC/HLR/VLR/MSC simulés + Twilio réel)

### `entities.py`

**Choix** : quatre classes d'état pur — `Hlr`, `Vlr`, `Msc`, `Smsc` — reflétant l'architecture GSM
réelle, sans aucune logique de délai/perte/retransmission (ça, c'est le rôle de `network_sim.py` à
venir ; même séparation que `codecs.py`/`fitness.py` en Module A) :
- `Hlr.register_ms(msisdn, imsi)` / `query_hlr(msisdn)` — base d'abonnés permanente.
- `Vlr.attach(msisdn, location_area)` / `is_attached` / `location_of` / `detach` — présence
  courante sur le réseau, distincte de l'abonnement HLR.
- `Msc.route(msisdn)` — un abonné n'est joignable que s'il est **à la fois** connu du HLR **et**
  actuellement attaché au VLR ; modélise la vraie distinction GSM entre « a un abonnement » et
  « le téléphone est actuellement allumé et enregistré ».
- `Smsc.send(...)` met en file (`MessageStatus.QUEUED`, id auto-incrémenté) ; `Smsc.deliver(message)`
  fait une tentative de livraison synchrone via `Msc.route`, met à jour le statut
  (`DELIVERED`/`UNDELIVERABLE`) et retourne un bool. Une seule tentative, sans notion de temps —
  `network_sim.py` empilera délai/perte/retransmission par-dessus.

**Résultats de test** : 11 tests, tous verts (registration HLR, cycle attach/detach VLR, les 4
combinaisons de routabilité MSC, et les deux chemins de `Smsc.deliver`).

### `pdu.py`

**Contexte important** : le sujet demande de « valider en encodant/décodant les exemples
hexadécimaux du cours » — mais aucun exemple hexadécimal n'existe nulle part dans ce dépôt
(`docs/SUJET.md` ne contient que la phrase, pas de PDU réel ; le cours source cité en §6.3 n'est
pas fourni). Plutôt que d'inventer une chaîne hex et de prétendre qu'elle vient « du cours » (ce qui
serait trompeur et invérifiable), la validation s'appuie sur des tests d'aller-retour
(`decode(encode(x)) == x`) sur de nombreux cas, plus un exemple calculé et vérifié à la main
ci-dessous pour la partie la plus piégeuse (l'empaquetage 7 bits).

**Choix — portée volontairement limitée** : alphabet GSM 7 bits par défaut uniquement (table
d'extension non supportée — `€`, `[`, `{`, etc. lèvent `ValueError` plutôt que d'être mal encodés
silencieusement), pas d'en-tête UDH / SMS concaténé, format de validité relative uniquement pour
TP-VP (pas absolu/enhanced), fuseau horaire toujours UTC+0 dans TP-SCTS (simulation locale, pas de
vrai décalage horaire à modéliser).

**Alphabet GSM 7 bits (3GPP TS 23.038)** : table explicite indexée par octet (dict `{index: char}`)
plutôt qu'une supposition « c'est presque de l'ASCII » — les zones 0x20-0x3F/0x41-0x5A/0x61-0x7A
coïncident bien avec l'ASCII, mais pas le reste (ex. index 0x00 = `@`, pas NUL ; lettres
accentuées, signe monnaie, lettres grecques utilisées en physique). Un test dédié
(`test_at_sign_is_index_zero_not_ascii_nul`) vérifie spécifiquement ce piège.

**Empaquetage 7 bits — vérifié à la main** : les septets sont concaténés en un flux de bits
(LSB en premier), puis découpés en octets de 8 bits (LSB en premier aussi) — implémenté via un
entier Python (`value |= septet << (7*i)` puis `.to_bytes(..., "little")`), pas une boucle de bits
manuelle, mais mathématiquement identique. Vérifié à la main pour `"Hi"` (H=0x48, i=0x69) :
- 0x48 en 7 bits (LSB→MSB) : `0,0,0,1,0,0,1`
- 0x69 en 7 bits (LSB→MSB) : `1,0,0,1,0,1,1`
- Flux concaténé (14 bits) + 2 bits de bourrage à 0 : `00010011 00101100`
- Premier octet `00010011` (lu LSB→MSB) = 8+64+128 = **0xC8**
- Second octet `00101100` (lu LSB→MSB) = 4+16+32 = **0x34**

Le code produit bien `pack_septets([0x48, 0x69]) == bytes([0xC8, 0x34])` — testé explicitement
(`test_two_septets_pack_as_hand_computed`, avec aussi le cas `"A","B"` plus simple en commentaire
dans le code). Comme le PDU transporte `TP-UDL` (le nombre exact de septets) séparément, le
décodage n'a pas besoin de la technique de désambiguïsation des bits de bourrage que mentionnent
certains tutoriels GSM7 pour un nombre de septets ≡ 7 (mod 8) — on dépaquette simplement les
`TP-UDL` premiers septets et le reste (bits de bourrage) est ignoré.

**Bug trouvé en écrivant les tests** : `_encode_bcd_pair` (champs de `TP-SCTS`) inversait l'ordre du
swap semi-octet par rapport à `_encode_bcd_digits` (adresses) — `(tens<<4)|units` au lieu de
`(units<<4)|tens`. Invisible sur `_decode_bcd_pair` seul (test isolé aurait pu passer par
coïncidence), détecté par le test d'aller-retour `TestScts::test_round_trip` qui échouait avec
`ValueError: month must be in 1..12` (les nibbles inversés donnaient un mois à deux chiffres
invalide). Corrigé pour utiliser la même convention de swap que les adresses.

**Deux conventions de longueur différentes, gardées séparées exprès** : `TP-DA`/`TP-OA` compte des
*chiffres décimaux* dans son octet de longueur, alors que l'info SMSC compte des *octets*
(type-adresse + BCD) — piège classique documenté. `encode_address_field`/`decode_address_field` et
`encode_smsc_field`/`decode_smsc_field` sont deux paires de fonctions distinctes plutôt qu'une
seule paramétrée, précisément pour ne pas risquer de mélanger les deux conventions.

**Exemple complet vérifié** (`SubmitPdu(destination="+33612345678", text="Hi")`, pas de SMSC ni de
durée de validité) :

```
00 01 00 0B 91 3316325476F8 00 00 02 C834
```

| Octets | Champ | Valeur |
|---|---|---|
| `00` | info SMSC | absente (utiliser le SMSC par défaut) |
| `01` | premier octet | TP-MTI=01 (SUBMIT), TP-VPF=00 (pas de VP) |
| `00` | TP-MR | référence 0 |
| `0B 91 3316325476F8` | TP-DA | 11 chiffres, international (0x91), `33612345678` en BCD swappé + bourrage `F` |
| `00` | TP-PID | normal |
| `00` | TP-DCS | alphabet GSM 7 bits par défaut |
| `02` | TP-UDL | 2 septets |
| `C8 34` | TP-UD | `"Hi"` empaqueté — valeur vérifiée à la main ci-dessus |

**Résultats de test** : 32 nouveaux tests (43 au total pour `module_b`), tous verts. Couverture
`pdu.py` : 99 % à l'origine ; seule `gsm7_decode`'s branche d'erreur (`KeyError→ValueError` sur un
septet invalide) restait non couverte — comblée ensuite par
`test_decode_unsupported_septet_raises` (voir plus bas, « Complément — couverture de
`gsm7_decode` »), portant `pdu.py` à 100 %. `entities.py` reste à 100 %.

### `network_sim.py`

**Choix** : toute la logique de délai/perte/retransmission vit ici, pas dans `entities.py`
(`Smsc.deliver` reste une vérification synchrone de routabilité, sans notion de temps) :
- `simulate_delivery(smsc, message, rng, ...)` — une tentative tire un délai (`rng.exponential
  (mean_delay_s)`, plus réaliste qu'une loi normale pour un délai réseau/file d'attente, toujours
  positif), puis échoue soit parce que `Msc.route` est faux (destinataire injoignable — téléphone
  éteint, réévalué à chaque tentative), soit par perte transitoire indépendante
  (`loss_probability`, réseau congestionné même si joignable). Backoff exponentiel
  (`backoff_factor`) entre tentatives ; le message est déclaré définitivement indélivrable une fois
  le temps cumulé au-delà de `max_retry_window_s` — **72h par défaut**, conforme au sujet.
- `simulate_batch_delivery(n_messages, reachable_probability, ...)` — simule une population de
  téléphones dont une fraction seulement est attachée au VLR (certains éteints), agrège taux de
  livraison, délai moyen, nombre moyen de tentatives.
- `simulate_overload(arrival_rate, duration_s, throughput_msgs_per_s, queue_capacity, ...)` — file
  à un seul serveur en temps discret : arrivées de Poisson par pas de `1/throughput_msgs_per_s` s
  (un message traité par pas), rejet immédiat si la file dépasse `queue_capacity`. Mesure
  directement ce que demande le sujet : taux de perte et délai moyen sous charge soutenue
  (« centaines de SMS/s »).
- `sweep_overload(arrival_rates, ...)` — le point précédent répété sur plusieurs charges, pour
  produire directement les données des courbes QoS-vs-charge du rapport (une seed dérivée par
  point, pour rester reproductible sans partager le même bruit d'échantillonnage entre points —
  même idée que `mushra_sim.simulate_mushra_panel` en Module A).

**Pourquoi Poisson + serveur à débit fixe plutôt qu'un modèle plus élaboré** : suffisant pour
démontrer la dégradation QoS demandée (perte + délai croissants avec la charge) sans complexité
supplémentaire non justifiée par le sujet — cohérent avec l'esprit « simulation, pas modélisation
réseau certifiée » déjà adopté pour `codecs.py` en Module A.

**Résultats de test** : 12 nouveaux tests (55 au total), tous verts, y compris un test de
conservation explicite (`n_delivered + n_dropped + n_still_queued == n_arrived`) et des tests de
reproductibilité par seed pour les trois fonctions stochastiques. Couverture `network_sim.py` :
**100 %** (`module_b` global : 99 %).

### `twilio_client.py`

**Choix** : trois fonctions publiques, sans aucune dépendance FastAPI/Flask — délibérément, la
route `POST /twilio/webhook` appartient au Module E (pas encore construit, cf. `docs/SUJET.md` §3
MOD-E et §5 phase 6), pas au Module B :
- `send_sms(to, body, from_=None, client=None)` — enveloppe `client.messages.create(...)` du SDK
  `twilio-python` ; `from_` retombe sur `TWILIO_PHONE_NUMBER` si non fourni.
- `get_delivery_status(sid, client=None)` — enveloppe `client.messages(sid).fetch()`, l'alternative
  « polling » que le sujet propose au webhook.
- `handle_status_webhook(payload: dict) -> SmsStatus` — fonction pure qui normalise les champs du
  callback Twilio (`MessageSid`, `MessageStatus`, `ErrorCode`) en `SmsStatus`, prête à être appelée
  depuis la future route FastAPI du Module E sans que ce fichier-ci ne sache ce qu'est FastAPI.

`client` est injectable partout, même schéma que `whisper_eval.transcribe` en Module A. Ajout de
`twilio` aux dépendances via `uv add twilio`.

**`manual_twilio_check.py`** : script manuel non exécuté par pytest (envoie un vrai SMS via le
trial Twilio), même schéma que `module_a/manual_whisper_check.py` — voir sa docstring pour le
raisonnement. Nécessite `TWILIO_TEST_TO_NUMBER` (numéro destinataire vérifié de l'étudiant),
**absent de la liste `docs/SUJET.md` §6.2** (qui ne liste que `TWILIO_PHONE_NUMBER`, l'expéditeur
trial) — ajouté à `.env.example` car un envoi réel a structurellement besoin de distinguer
expéditeur et destinataire, omission du sujet plutôt qu'un choix de conception.

**Tests** : un `FakeTwilioClient`/`FakeMessagesResource`/`FakeMessageContext` reproduisant la forme
exacte du SDK réel (`client.messages.create(...)` et `client.messages(sid).fetch()` — `messages`
est à la fois appelable et pourvu de `.create`, comme le vrai SDK), même approche que le
`FakeClient` OpenAI du Module A plutôt qu'une bibliothèque de mock. Pas de test d'intégration
réseau — `manual_twilio_check.py` couvre ça, à la demande seulement.

**Résultats de test** : 8 nouveaux tests (63 au total), tous verts. Couverture `twilio_client.py` :
**100 %** ; `manual_twilio_check.py` à 0 % (jamais exécuté par pytest, comme
`manual_whisper_check.py` en Module A — n'affecte pas le seuil global du projet, 95 % tous modules
confondus).

### Complément — envoi réel bloqué par la politique de template trial + contournement via Verify API

**Constat en exécutant `manual_twilio_check.py`** : Twilio a durci sa politique SMS trial depuis
l'écriture de `twilio_client.py` — `messages.create(body=...)` avec un texte libre échoue désormais
systématiquement sur un compte trial avec `HTTP 400 : Invalid template name. Trial accounts can
only use predefined SMS templates.` (erreur Twilio 60409). Les comptes trial ne peuvent plus
envoyer que l'un d'une dizaine de templates de contenu prédéfinis (`sms_2fa`,
`sms_appointment_reminders`, `sms_order_confirmation`, `sms_delivery_updates`,
`sms_customer_support`, `sms_marketing_promotions`, `sms_event_notifications`,
`sms_account_alerts`, `sms_feedback_surveys`, `sms_internal_alerts`). C'est une contrainte réelle
absente de la simulation (`entities.Smsc` n'a aucune notion de « contenu autorisé ») : un premier
exemple concret et mesuré d'écart sim/réel pour §8 Q2, découvert sans avoir eu à le construire
artificiellement, même esprit que la dérive Whisper trouvée en Module A. **Correction ultérieure** :
contrairement à ce que documentait ici une première version de ce paragraphe, le corps du message
n'a pas besoin de reproduire un texte approuvé invisible hors Console — envoyer littéralement le
*nom* du template comme `body` (ex. `body="sms_appointment_reminders"`) est accepté par l'API et
Twilio y substitue le contenu réel à la livraison. Vérifié en le testant directement (voir
`manual_twilio_check.py` plus bas) plutôt que supposé depuis la documentation, qui ne précisait pas
ce détail.

**Contournement retenu — Twilio Verify plutôt qu'upgrade payant** : `docs/SUJET.md` §3 MOD-B cite
déjà « Optionnel : Twilio Verify... pour un OTP SMS ». Le corps du SMS envoyé par Verify est généré
par Twilio lui-même (code OTP), jamais du texte fourni par l'appelant — il échappe donc
structurellement à la restriction de template (qui ne s'applique qu'à `messages.create`). Deux
fonctions ajoutées à `twilio_client.py`, même schéma d'injection de `client` que le reste du
fichier :
- `start_verification(to, channel="sms", client=None) -> VerificationResult` — enveloppe
  `client.verify.v2.services(service_sid).verifications.create(...)`.
- `check_verification(to, code, client=None) -> VerificationCheckResult` — enveloppe
  `client.verify.v2.services(service_sid).verification_checks.create(...)`.

Nécessite un Verify Service (ressource Twilio distincte du compte, créée une seule fois via
`client.verify.v2.services.create(friendly_name=...)`, gratuite) — son SID va dans
`TWILIO_VERIFY_SERVICE_SID`, ajouté à `.env`/`.env.example` (même raisonnement d'omission du sujet
que `TWILIO_TEST_TO_NUMBER`).

**`manual_twilio_verify_check.py`** : nouveau script manuel, même statut hors-pytest que
`manual_twilio_check.py`. Sans argument, démarre une vérification (envoie l'OTP réel) ; appelé avec
un code en argument, le vérifie — scindé en deux invocations plutôt qu'un `input()` bloquant, pour
rester utilisable depuis un terminal piloté par un agent aussi bien qu'à la main.

**Tests** : fakes Verify (`FakeVerification`, `FakeVerificationsResource`,
`FakeVerificationCheck`, `FakeVerificationChecksResource`, `FakeVerifyService`,
`FakeServicesResource`, `FakeVerifyV2`, `FakeVerify`) composés dans `FakeTwilioClient.verify.v2`,
même principe de forme-exacte-du-SDK que les fakes Messages. 5 nouveaux tests (68 au total),
tous verts. Couverture `twilio_client.py` : toujours **100 %**.

**Résultat de l'envoi réel** : `manual_twilio_verify_check.py` exécuté sans argument le
2026-09-01 — un vrai SMS OTP envoyé par l'API Verify à `TWILIO_TEST_TO_NUMBER`
(`sid=VEadbbe2122442b62843f6f85a2831fbf3`, `status=pending` à l'acceptation). Reçu sur le
téléphone réel de l'étudiant, le code a été renvoyé au script (second appel avec l'argument
`<code>`), qui a retourné `status=approved`, `valid=True` auprès de l'API Verify réelle —
aller-retour complet (envoi → réception humaine → vérification) réussi sur le compte trial,
là où l'envoi SMS libre via `messages.create` reste bloqué (cf. ci-dessus). Contrairement à
`compare.measure_real_delivery` (spécifique à l'API Messages), le script Verify ne chronomètre
pas de latence — l'objectif ici était de prouver qu'un envoi/réception réel fonctionne de bout en
bout sur ce compte, pas de peupler `compare.py`, dont les métriques (`accept_latency_s`,
`delivery_latency_s`) resteraient de toute façon non représentatives d'un flux Verify. Le SMS-vs-
Verify est lui-même une donnée pour §8 Q2 : le trial Twilio autorise l'authentification (OTP,
alertes système) mais pas la messagerie de contenu libre, une segmentation anti-spam qui n'existe
pas dans le simulateur.

### Complément — `messages.create` débloqué, et un deuxième bug trial découvert au passage

**Le SMS libre fonctionne en fait** : une fois `body` remplacé par un nom de template valide (cf.
correction ci-dessus), `manual_twilio_check.py` a pu à nouveau utiliser l'API Messages plutôt que
Verify. `_TRIAL_TEMPLATE_BODY = "sms_appointment_reminders"` remplace le texte libre d'origine dans
le script.

**Deuxième contrainte trial trouvée en le faisant fonctionner** : `get_delivery_status`
(`client.messages(sid).fetch()`) renvoie systématiquement `HTTP 403 Forbidden` sur ce compte trial
— alors que le même message est parfaitement visible via `client.messages.list()`. Confirmé en
comparant les deux appels sur le même SID. `get_delivery_status` retente donc désormais via
`list()` (recherche du SID dans les 50 messages les plus récents) quand le fetch individuel est
refusé, et ne relève l'exception d'origine que si le SID reste introuvable ou si l'erreur n'est pas
un 403.

**Effet de bord découvert en re-testant `compare.measure_real_delivery`** : juste après l'envoi, un
message tout juste créé peut échouer *aussi* sur `list()` pendant quelques secondes (indexation
Twilio pas encore à jour) — `get_delivery_status` relève alors légitimement l'exception (SID
introuvable nulle part), mais la boucle de polling de `measure_real_delivery` ne l'attrapait pas et
plantait dès la première tentative. Corrigé : la boucle traite désormais un `TwilioRestException`
403 comme un statut « pas encore délivré » ordinaire (elle réessaie après `sleep`), et ne relève que
les erreurs non-403.

**Tests** : `TestGetDeliveryStatus` gagne 3 cas (repli sur `list()`, ré-lève une erreur non-403,
ré-lève un 403 introuvable même via `list()`) via un `FakeMessagesResource` étendu
(`fetch_error_status`, `list()`, `hidden_from_list`). `TestMeasureRealDelivery` gagne 2 cas (retente
à travers un 403 transitoire puis livre, ré-lève une erreur non-403 pendant le polling). 5 nouveaux
tests (73 au total pour `module_b`), tous verts. Couverture `twilio_client.py` et `compare.py` :
**100 %**.

**Résultat de l'envoi réel (API Messages)** : `manual_twilio_check.py` exécuté avec succès —
`accept_latency_s=2.09`, `delivery_latency_s=5.39`, `final_status=delivered`. Deux tentatives
précédentes ont d'abord buté sur de l'instabilité d'infrastructure Twilio elle-même (un `HTTP 500`
puis un `HTTP 502` CloudFront sur `list()`, sans rapport avec le compte trial) — non reproduites
en tests, `manual_twilio_check.py` reste un script de smoke-test ponctuel, pas un client robustifié
contre toute panne d'infrastructure tierce. `compare.py` a maintenant un vrai `RealDeliverySample`
à comparer aux stats simulées de `network_sim.simulate_batch_delivery` pour §8 Q2.

### `compare.py`

**Choix** : trois fonctions, aucune ne code en dur les causes de l'écart sim/réel — c'est la
matière du rapport final (§8 Q2 du sujet : « ≥3 causes structurelles »), pas du code :
- `measure_real_delivery(to, body, ...)` — envoie un vrai SMS et chronomètre deux latences
  distinctes : `accept_latency_s` (temps d'acceptation par l'API Twilio) et `delivery_latency_s`
  (temps jusqu'au statut terminal `delivered`/`failed`/`undelivered`, par polling). `sleep`/`now`
  sont injectables (comme `client`) précisément pour ne jamais vraiment attendre en test.
- `summarize_real_samples(samples)` — agrège plusieurs mesures réelles (taux de livraison, latence
  moyenne d'acceptation, latence moyenne de livraison **sur les seuls échantillons l'ayant
  atteinte** — un timeout ne doit pas silencieusement tirer la moyenne vers le bas).
- `build_comparison_table(simulated, real)` — assemble stats simulées (`network_sim
  .BatchDeliveryStats`) et stats réelles (`RealStats`) côte à côte, même esprit que
  `visualize.build_correlation_table` en Module A.

**Refactoring associé** : `manual_twilio_check.py` (Module B, commit précédent) simplifié pour
réutiliser `measure_real_delivery` plutôt que dupliquer sa propre boucle de polling — la logique
de mesure temporelle n'existe plus qu'à un seul endroit.

**Résultats de test** : 5 nouveaux tests (78 au total), tous verts, y compris un test d'horloge
factice (`sleep()` avance le temps simulé et fait progresser le statut du message factice —
aucune vraie attente) couvrant à la fois le cas « livré après un poll » et le cas « timeout sans
statut terminal ». Couverture `compare.py` : **100 %**.

### `fitness.py`

**Contexte — écart avec le sujet à noter explicitement** : `docs/SUJET.md` §3 MOD-B demande
`routing_fitness(chromosome) → latence simulée, utilisée par le Module F`, mais contrairement au
Pb1 de Module A (config codec, explicitement câblé dans §3 MOD-F), **aucun problème du Module F
(§3 MOD-F : Pb1 codec/A, Pb2 placement BTS/C, Pb3 QoS/D) n'utilise `routing_fitness`**. Cette
fonction est donc construite au même niveau d'exigence de stabilité de contrat que `codec_fitness`
(CLAUDE.md : « garder la signature stable une fois que Module F en dépend »), mais sans qu'aucun
livrable noté du Module F ne l'appelle concrètement — écart du sujet documenté ici plutôt que
laissé implicite ou maquillé.

**Chromosome — design propre à ce module, faute de définition dans le sujet** :
`[retry_backoff_s, backoff_factor, max_retry_window_hours]`, la politique de retransmission que
`network_sim.simulate_delivery` applique. Compromis qu'un AG doit pouvoir découvrir : un backoff
court réduit le délai des messages qui finissent par réussir mais multiplie les tentatives (charge
SMSC) ; une fenêtre de réessai courte libère les ressources plus vite mais abandonne plus tôt des
destinataires joignables-mais-lents, réduisant le taux de livraison. `decode_chromosome` reprend
exactement le style tolérant de `module_a.fitness.decode_chromosome` (valeurs discrètes ramenées au
choix valide le plus proche, gène continu (`backoff_factor`) borné par clip).

**`routing_fitness_components`** : simule un lot (`network_sim.simulate_batch_delivery`) et calcule
`fitness = w1·taux_livraison − w2·(délai_moyen/normalisation) − w3·(tentatives_moyennes/normalisation)`,
plus une pénalité de contrainte optionnelle (`min_delivery_rate`) — même schéma que la pénalité de
plafond de bitrate en Module A.

**Différence notable avec `codec_fitness`, à documenter** : pas de cache module-level équivalent à
`_get_reference_signal`. `codec_fitness` en avait besoin car générer le signal de référence est un
appel réseau (gTTS) coûteux à répéter des milliers de fois ; ici, `network_sim.simulate_batch_delivery`
est du Python/NumPy pur et bon marché — `routing_fitness` est donc un simple wrapper direct sur
`routing_fitness_components`, sans état partagé entre appels. Même défaut `seed=None` que
`codec_fitness` par cohérence : la reproductibilité entre appels reste la responsabilité de
l'appelant (Module F), pas fixée ici.

**Résultats de test** : 7 nouveaux tests (85 au total pour `module_b`), tous verts. Couverture
`fitness.py` : **100 %**. Couverture globale `module_b` à ce stade : **96 %** (seuls
`pdu.gsm7_decode`'s branche d'erreur et les deux scripts manuels `manual_twilio_check.py`/
`manual_twilio_verify_check.py`, jamais exercés par pytest par conception, restaient non
couverts).

### Complément — couverture de `gsm7_decode`

**Choix** : dernier écart de couverture identifié dans `module_b` — `gsm7_decode` (pdu.py) lève
`ValueError` sur un septet hors table (`KeyError` intercepté), mais seule la branche symétrique de
`gsm7_encode` était testée (`test_unsupported_character_raises`). Ajout de
`test_decode_unsupported_septet_raises`, qui vérifie que décoder `[0x1B]` (le code d'échappement de
la table d'extension, explicitement absent de `_GSM7_CHARS`) lève bien `ValueError`.

**Résultats de test** : 1 nouveau test (86 au total pour `module_b`), tous verts. Couverture
`pdu.py` : **100 %**. Couverture globale `module_b` : **96 %**, `manual_twilio_check.py` et
`manual_twilio_verify_check.py` restant à 0 % par conception (scripts manuels, jamais exercés par
pytest) — ce sont désormais les deux seuls fichiers non couverts du module.

---

## Module C — Géolocalisation LBS (OpenCelliD + Nominatim + cartes Folium)

### `opencellid_loader.py`

**Choix** : le sujet recommande explicitement le mode offline (dump CSV complet) plutôt que l'API
live d'OpenCelliD (quota 1000 req/jour) pour éliminer toute contrainte de quota — l'utilisateur a
déjà téléchargé le dump France (MCC=208) dans `docs/208.csv` (392 420 lignes, sans en-tête). Deux
fonctions seulement :
- `load_opencellid_csv(csv_path, mcc=208, use_cache=True)` — charge le CSV en DataFrame pandas
  avec des colonnes nommées explicitement (`COLUMNS`), filtre optionnel par MCC. Mise en cache
  mémoire par `(csv_path, mcc)`, même principe que `_reference_cache` de `module_a/fitness.py`
  (pas un nouveau format de cache disque type parquet) — évite de reparser 392k lignes à chaque
  appel dans un même run. Chaque appel retourne une copie (`.copy()`) pour qu'une mutation côté
  appelant ne corrompe pas le cache.
- `sample_bts(df, bbox, n=1000, seed=None)` — filtre par bounding box puis échantillonne sans
  remise jusqu'à `n` lignes ; si la zone contient moins de `n` BTS, retourne tout ce qu'il y a
  plutôt que de lever une exception (une démo construite sur ce qui existe vaut mieux qu'un crash
  au démarrage).

**Piège documenté explicitement** : le schéma public OpenCelliD liste `lon` avant `lat` — bug
classique si on suppose l'ordre inverse. `COLUMNS` le nomme sans ambiguïté et un test dédié
(`test_lon_before_lat_column_order`) vérifie l'ordre sur des données connues.

**Région cible — Rodez / Aveyron (département 12)** : choisi avec l'utilisateur plutôt que
Nice/Paris (qui se trouvent être les premières lignes du fichier). Vérifié que la zone
`lon∈[1.5,3.5], lat∈[43.8,44.9]` contient réellement 6 815 BTS dans le dump — largement au-dessus
de la cible du sujet (500-2000 BTS), donc `sample_bts` échantillonne depuis une population réelle
non triviale plutôt que de racler le minimum.

**Tests** : CSV synthétique à 5 lignes (`tmp_path`, pas le vrai `docs/208.csv` — trop volumineux
pour des tests unitaires rapides), couvrant filtrage MCC, ordre lon/lat, indépendance des copies
issues du cache, filtrage bbox, plafonnement sans exception, et reproductibilité par seed. Vérifié
séparément (hors suite pytest) que `load_opencellid_csv()` + `sample_bts(n=1000, seed=42)` sur le
vrai `docs/208.csv` retourne bien 1000 BTS réelles en Aveyron.

**Résultats de test** : 7 tests, tous verts. Couverture `opencellid_loader.py` : **100 %**.

### `terrain_sim.py`

**Choix** : tout ce qui vient ensuite (Voronoï, trilatération TOA, grille Wi-Fi) raisonne
naturellement en mètres euclidiens, pas en degrés (lon, lat) — projection équirectangulaire
centrée sur le centroïde des BTS échantillonnées plutôt qu'une projection géodésique complète
(UTM via `pyproj`, absent de `docs/SUJET.md` §6.1) : la zone d'étude (Aveyron, ~100 km) est assez
petite pour que l'erreur d'approximation reste largement sous 0,1 %, sans dépendance
supplémentaire.
- `lonlat_to_xy(lon, lat, lon0, lat0)` / `xy_to_lonlat(x, y, lon0, lat0)` — paire de fonctions
  réciproques, vectorisées (acceptent scalaires ou tableaux numpy). La seconde est nécessaire
  parce que Folium (à venir, `map_viz.py`) affiche en (lon, lat), pas en mètres.
- `Terrain` (dataclass gelée) : le DataFrame BTS d'origine enrichi des colonnes `x`/`y`, plus
  `lon0`/`lat0` (centre de projection). `positions_xy` et `bbox_xy` exposent respectivement un
  tableau `(N, 2)` et une bounding box en mètres — la forme exacte qu'attendront
  `scipy.spatial.Voronoi`, `scipy.optimize.minimize` et `sklearn.neighbors` dans les fichiers
  suivants.
- `build_terrain(bts_df)` centre la projection sur le centroïde des BTS *échantillonnées* (pas un
  point de référence global fixe) — l'origine du repère local reste proche de (0, 0) quelle que
  soit la région choisie.

**Tests** : aller-retour lon/lat → x/y → lon/lat, vérification qu'un degré de latitude fait bien
~111,2 km (formule de référence, indépendante de l'implémentation), centrage sur le centroïde,
cohérence de `bbox_xy` avec le min/max réel, conservation des colonnes d'origine du DataFrame.

**Résultats de test** : 8 nouveaux tests (15 au total pour `module_c`), tous verts. Couverture
`terrain_sim.py` : **100 %**.

### `cell_id.py`

**Choix — positionnement Cell-ID par centroïde Voronoï** (spec ligne 190-191) : le point piégeux
est que `scipy.spatial.Voronoi` laisse les cellules de bord non bornées (sommets « à l'infini »),
sans centroïde défini. Solution standard plutôt que d'ajouter une dépendance (`shapely`) : ajouter
quatre points fictifs loin en dehors de la bounding box avant de construire le diagramme, ce qui
force toutes les cellules des BTS réelles à devenir finies, puis découper (Sutherland-Hodgman,
numpy pur) chaque cellule à la bounding box réelle avant d'en calculer le centroïde (formule du
lacet, pondérée par l'aire).
- `voronoi_cell_centroids(terrain)` — un centroïde par BTS réelle, calculé une seule fois (pas par
  requête : la « erreur médiane sur 100 positions simulées » du sujet réutilise le même diagramme).
- `estimate_position(true_xy, terrain, centroids=None)` — estimation Cell-ID = centroïde de la
  cellule Voronoï de la BTS réelle la plus proche de `true_xy` (accepte des centroïdes
  précalculés pour éviter de reconstruire le diagramme à chaque appel).
- `evaluate_accuracy(terrain, n_positions=100, seed=None)` — tire `n_positions` positions
  aléatoires dans la bounding box du terrain, mesure l'erreur pour chacune, retourne l'erreur
  médiane (spec ligne 191) et le tableau brut.

**Vérifié à la main** : cas de test à 4 BTS aux coins d'un carré 10×10 — le diagramme Voronoï est
exactement les deux médiatrices `x=5`/`y=5`, donc la cellule (découpée) de la BTS en (0,0) est le
carré `[0,5]×[0,5]`, de centroïde `(2.5, 2.5)` — calculé à la main et confirmé par le code
(`test_matches_hand_computed_quadrant_centroids`), même démarche que le PDU vérifié à la main en
Module B.

**Résultat sur données réelles** (hors suite de tests, 500 BTS Aveyron échantillonnées avec
`seed=42`, 100 positions simulées avec `seed=1`) : erreur médiane Cell-ID ≈ **3,4 km** — ordre de
grandeur plausible pour un positionnement Cell-ID en zone rurale à faible densité de tours, à
comparer plus tard aux résultats TOA/Wi-Fi/IP dans le tableau comparatif du rapport final.

**Tests** : découpage Sutherland-Hodgman (polygone débordant, entièrement intérieur, entièrement
extérieur → vide), centroïde de polygone (carré unité, dégénéré à 2 points, dégénéré à 3 points
colinéaires — aire nulle), le cas carré vérifié à la main ci-dessus, estimation à partir de
centroïdes précalculés ou recalculés, reproductibilité par seed de `evaluate_accuracy`.

**Résultats de test** : 11 nouveaux tests (26 au total pour `module_c`), tous verts. Couverture
`cell_id.py` : **100 %**.

### `toa.py`

**Choix — trilatération TOA** (spec ligne 192-193) : le sujet précise le modèle de délai
(`distance / vitesse lumière + bruit gaussien`) et le solveur (`scipy.optimize.minimize`, SLSQP)
mais pas la fonction objectif elle-même — choix naturel : minimiser la somme des carrés des
résidus entre distances mesurées (dérivées des pseudo-délais) et distances hypothétiques aux BTS
ancres.
- `_nearest_anchors(true_xy, terrain, k)` — sélectionne les `k` BTS réelles les plus proches
  (défaut `k=4`, plafonné au nombre de BTS disponibles).
- `simulate_pseudoranges(true_xy, anchors_xy, timing_noise_std_s, rng)` — délai = distance / c,
  bruit gaussien sur le délai (défaut σ=50 ns, soit ~15 m de bruit de portée à 1σ — ordre de
  grandeur plausible pour un positionnement type TOA/GPS), reconverti en distance ; délais négatifs
  écrêtés à 0 (non physiques).
- `trilaterate(anchors_xy, measured_distances, initial_guess=None)` — SLSQP minimisant
  `Σ(‖x − ancre_i‖ − distance_mesurée_i)²`, point de départ = centroïde des ancres par défaut.
- `estimate_position` / `evaluate_accuracy` — mêmes contrats que `cell_id.py` (même signature
  `evaluate_accuracy(terrain, n_positions=100, seed=None) -> {"median_error_m", "errors_m"}`) pour
  rester directement comparables dans le tableau récapitulatif du rapport final. Visualiser les
  cercles de portée reste le rôle de `map_viz.py`, pas de ce fichier.

**Vérifié à la main** : 4 ancres aux coins d'un carré 100×100, bruit nul — `trilaterate` retrouve
exactement la position vraie `(40, 60)` à `1e-3` près (`test_recovers_position_exactly_with_zero_noise`).

**Résultat sur données réelles** (mêmes 500 BTS Aveyron et mêmes seeds que Cell-ID ci-dessus) :
erreur médiane TOA ≈ **19 m**, contre ≈ 3,4 km pour Cell-ID — l'écart de précision attendu entre
une estimation "cellule la plus proche" et une vraie trilatération, exactement ce que le tableau
comparatif du rapport final (§8) doit mettre en évidence.

**Tests** : sélection des k plus proches ancres (plafonnée au nombre de BTS disponibles), bruit nul
→ distance exacte, bruit non nul perturbe la mesure, trilatération exacte à bruit nul (à la main),
point de départ explicite, précision raisonnable avec bruit réaliste, forme et reproductibilité de
`evaluate_accuracy`.

**Résultats de test** : 9 nouveaux tests (35 au total pour `module_c`), tous verts. Couverture
`toa.py` : **100 %**.

### `wifi_fp.py`

**Choix — empreintes réutilisant les vraies BTS** : le sujet ne définit nulle part un jeu de
« points d'accès Wi-Fi » séparé — comme Cell-ID et TOA, ce fichier réutilise les positions réelles
des BTS échantillonnées comme sources du signal simulé plutôt que d'inventer un dataset
supplémentaire. RSSI simulé par un modèle de perte de parcours log-distance standard (référence à
1 m + bruit gaussien optionnel), pas un modèle physique calibré — même esprit que la dégradation
SNR de `module_a/codecs.py`.

**Piège trouvé en testant sur les vraies données — portée du fingerprinting** : le fingerprinting
Wi-Fi est par nature une technique à échelle locale (un bâtiment, un campus, un quartier —
quelques centaines de mètres à quelques km), contrairement à Cell-ID/TOA qui opèrent sur tout le
réseau macro. Une première version calquait la grille 100 m sur la bounding box *entière* du
terrain — sur l'échantillon Aveyron réel (~159 km × 122 km), ça fait **~1,9 million de cellules**,
un calcul qui ne termine pas en temps raisonnable. Corrigé : `build_fingerprint_grid` et
`evaluate_accuracy` prennent désormais un `zone_center_xy`/`zone_size_m` (2 km par défaut, centré
sur le centroïde des BTS) délimitant une zone locale, indépendante de l'étendue macro du terrain —
recentré sur ce qu'un fingerprinting Wi-Fi réel couvrirait effectivement.
- `simulate_rssi(position_xy, bts_positions_xy, noise_std_db=0.0, rng=None)` — RSSI par BTS.
- `build_fingerprint_grid(terrain, zone_center_xy=None, zone_size_m=2000.0, cell_size_m=100.0)` —
  grille de référence sans bruit sur la zone locale (spec ligne 194 : grille 100 m × 100 m).
- `fit_knn(fingerprints, k=3)` / `estimate_position(query, centroids, model)` — k-NN
  (`sklearn.neighbors.NearestNeighbors`) sur l'espace des empreintes RSSI ; estimation = moyenne
  des k centroïdes de grille les plus proches en distance RSSI.
- `evaluate_accuracy(...)` / `sweep_noise(terrain, noise_levels_db, ...)` — la seconde répète la
  première à plusieurs niveaux de bruit pour produire la courbe « précision en fonction du bruit »
  demandée par le sujet (spec ligne 195), une seed dérivée par point (même principe que
  `network_sim.sweep_overload` en Module B) pour rester reproductible sans partager le bruit
  d'échantillonnage entre points.

**Résultat sur données réelles** (500 BTS Aveyron, zone locale par défaut 2 km, 100 positions,
`seed=1`) : erreur médiane 25,7 m à bruit nul (récupération quasi exacte sur la grille), 488 m à
4 dB, 992 m à 8 dB, 1116 m à 16 dB — dégradation nette et monotone avec le bruit, cohérente avec ce
que le sujet demande de montrer.

**Tests** : RSSI décroissant avec la distance, déterminisme à bruit nul, perturbation par le bruit,
taille de grille correcte pour une zone/cellule données, garde-fou zone-plus-petite-que-la-cellule
(retombe sur une seule cellule plutôt qu'une grille vide), non-explosion de la grille pour un
terrain largement plus grand que la zone par défaut, récupération exacte d'un centroïde de grille à
bruit nul, forme/reproductibilité de `evaluate_accuracy`, dégradation avec le bruit, un résultat par
niveau de bruit dans `sweep_noise` et sa reproductibilité.

**Résultats de test** : 12 nouveaux tests (47 au total pour `module_c`), tous verts. Couverture
`wifi_fp.py` : **100 %**.

### `ipinfo_client.py`

**Choix** : une seule fonction, `locate_ip(ip=None, token=None, session=None, timeout_s=5.0)` —
enveloppe `GET https://ipinfo.io/{ip}/json` (ou `.../json` sans IP pour la propre IP publique de
l'appelant). `session` injectable comme `client` dans `twilio_client.py` — sans injection, utilise
directement le module `requests` (son `requests.get` a la même signature qu'une session, donc pas
besoin de créer une session par défaut). ipinfo.io renvoie les coordonnées comme une seule chaîne
`"lat,lon"` (`_parse_loc`), et omet parfois le champ `loc` entièrement (IP dont la géoloc est
inconnue) — `lat`/`lon` valent alors `None` plutôt que de lever une exception.

**Discussion granularité/usages légitimes** (spec ligne 197, à développer dans le rapport final) :
la géoloc IP est précise au mieux à l'échelle de la ville, parfois seulement de la région/pays —
largement insuffisant pour localiser une personne précisément, ce qui limite les usages légitimes à
la personnalisation grossière de contenu, la détection de fraude, ou les analytics agrégées, pas le
tracking individuel.

**Tests** : `FakeIpinfoSession`/`FakeIpinfoResponse` reproduisant la forme de `requests.Session`
(`.get(url, params, timeout)` → objet avec `.raise_for_status()`/`.json()`), même principe que
`FakeTwilioClient` en Module B. Réponse complète, IP propre par défaut, IP explicite dans l'URL,
champ `loc` absent, jeton manquant lève une erreur, jeton explicite prioritaire sur l'environnement,
statut HTTP d'erreur propage `requests.HTTPError`.

**`manual_ipinfo_check.py`** : script manuel non exécuté par pytest, même schéma que
`manual_twilio_check.py`/`manual_nominatim_check.py`.

**Résultat de l'exécution réelle (2026-09-02)**, une fois `IPINFO_TOKEN` renseigné avec un vrai
jeton (compte gratuit ipinfo.io) : géolocalisation de l'IP publique de la machine résolue en
`Toulouse, Occitanie, FR` (`lat=43.6043, lon=1.4437`) — cohérent avec la localisation réelle de
l'étudiant, à l'échelle de la ville comme attendu (cf. discussion granularité ci-dessus). Aucune
erreur, jeton lu depuis `.env` via `load_dotenv()` sans avoir besoin d'être passé explicitement.

**Résultats de test** : 7 nouveaux tests (54 au total pour `module_c`), tous verts. Couverture
`ipinfo_client.py` : **100 %** ; `manual_ipinfo_check.py` à 0 % par conception (jamais exercé par
pytest).

### `lbs_poi.py`

**Choix — Overpass plutôt que la recherche texte-libre Nominatim** : le sujet cite « Nominatim/
Overpass » ensemble (même donnée OpenStreetMap, gratuite, sans inscription) — la recherche
texte-libre de Nominatim ne se prête pas naturellement à « les N POI les plus proches, toutes
catégories confondues » ; le filtre `around:rayon,lat,lon` d'Overpass QL est le bon outil pour
cette forme de requête précise. Une seule fonction :
- `nearby_pois(lat, lon, radius_m=500, limit=10, session=None, timeout_s=25.0)` — construit une
  requête Overpass QL (`node(around:...)[amenity]`), sur-récupère (3× `limit`, l'ordre Overpass
  n'étant pas garanti trié par distance), calcule la vraie distance haversine à chaque résultat, et
  trie/tronque côté client.

**Conformité spec ligne 202** : `User-Agent` obligatoire (`USER_AGENT`, envoyé sur chaque appel) ;
limite 1 req/s — un seul appel par invocation ici n'a rien à limiter en soi, un appelant qui
boucle sur plusieurs positions est responsable d'espacer ses appels d'au moins 1 s.

**Tests** : `FakeOverpassSession`/`FakeOverpassResponse` (même principe que
`FakeIpinfoSession`) — requête contient bien lat/lon/rayon, en-tête `User-Agent` envoyé, tri par
distance croissante, troncature à `limit`, éléments sans coordonnées ignorés, tags manquants →
placeholder `"?"`, statut HTTP d'erreur propagé, plus un calcul haversine vérifié (1° de latitude
≈ 111,2 km, même référence que `terrain_sim.py`).

**`manual_nominatim_check.py`** — **exécuté avec succès** (pas de clé requise, contrairement à
ipinfo.io) : requête réelle Overpass autour du centre de Rodez (44.3506, 2.5731), 500 m — 10 POI
réels retournés, du bar « Les Colonnes » à 86 m au point d'eau « drinking_water » à 238 m,
mélange plausible de commerces, mobilier urbain et infrastructure de stationnement pour un centre
de petite ville française.

**Résultats de test** : 9 nouveaux tests (63 au total pour `module_c`), tous verts. Couverture
`lbs_poi.py` : **100 %**.

### `map_viz.py`

**Choix** : même discipline que `module_a/visualize.py` — les fonctions de construction ne font
aucune I/O et retournent un objet déjà peuplé (`folium.Map` au lieu d'un `Figure` matplotlib),
`save_map_html` est le seul point d'écriture explicite, dans le même dossier `results/` (gitignoré)
que le reste du projet. Les POI sont acceptés en duck-typing (`.lat`/`.lon`/`.name`/`.category`/
`.distance_m`) plutôt que via un import de `lbs_poi.Poi` — ce fichier n'a ainsi aucune dépendance
dure envers l'origine des données.
- `build_position_map(true_lat, true_lon, estimated_lat, estimated_lon, uncertainty_radius_m=None,
  pois=None, zoom_start=15)` — marqueur position estimée (toujours), position réelle (si fournie),
  cercle d'incertitude (si fourni), marqueurs POI cliquables (spec ligne 203-204).
- `add_toa_range_circles(m, anchors_latlon, radii_m)` — superpose les cercles de portée TOA sur une
  carte existante (spec ligne 193), retourne la même instance de carte pour chaînage.
- `save_map_html(m, filename, output_dir="results")` — carte HTML autonome, lisible hors connexion
  après génération (spec ligne 204).

**Vérification** : comme pour une carte, « est-ce que ça a l'air correct » ne se teste pas par
assertion — vérifié plutôt en comptant les objets `folium.Marker`/`folium.Circle` ajoutés à
`m._children`, plus une démonstration bout-en-bout sur données réelles (échantillon Aveyron
500 BTS, position estimée par TOA, 8 vrais POI Overpass autour de cette position, rayon 1 km) :
carte HTML de 15,7 Ko générée dans `results/module_c_demo.html`, confirmée structurellement (10
marqueurs Leaflet = 1 position estimée + 1 position réelle + 8 POI, 1 cercle = incertitude).

**Tests** : marqueur estimé toujours présent, marqueur réel ajouté/omis selon fourniture, cercle
d'incertitude ajouté/omis selon fourniture, marqueurs POI ajoutés (et absence de POI gérée
gracieusement), cercles de portée TOA (un par ancre, retour de la même instance pour chaînage),
création de fichier + création du dossier de sortie s'il manque.

**Résultats de test** : 12 nouveaux tests (75 au total pour `module_c`), tous verts. Couverture
`map_viz.py` : **100 %**.

### `fitness.py`

**Choix — contrat multi-objectif, différent de `codec_fitness`/`routing_fitness`** :
`bts_coverage_fitness(chromosome) -> (f1, f2, f3)` retourne directement le triplet d'objectifs
plutôt qu'un score scalaire — écart délibéré par rapport au contrat `*_fitness() -> float` des
modules A et B, parce que le Pb2 du Module F est explicitement multi-objectif Pareto (NSGA-II via
pymoo attend un tableau d'objectifs par individu, pas un seul nombre). Documenté explicitement en
tête de fichier pour que ce ne soit jamais lu comme un oubli.

**Chromosome** : `[x1, y1, ..., xN, yN]`, coordonnées normalisées `[0, 1]` (repliées/`clip`ées),
dénormalisées vers la bounding box réelle du terrain par `decode_chromosome(chromosome, terrain)`
(spec ligne 333).

**Les trois objectifs** (spec lignes 336-337) :
- **f1 = −couverture (%)** — `coverage_fraction` tire des points de test aléatoires dans la
  bounding box et mesure la fraction à moins de `COVERAGE_RADIUS_M` (3 km, cohérent avec l'erreur
  médiane Cell-ID ≈ 3,4 km mesurée plus haut sur les mêmes données) de n'importe quelle BTS
  (réelle existante + nouvelle candidate).
- **f2 = interférence moyenne** — `mean_interference` : chaque voisin (existant ou nouveau) à moins
  de `INTERFERENCE_RADIUS_M` (1 km) contribue `(rayon − distance) / rayon` (1 à distance nulle, 0 au
  bord), moyenné par nouvelle BTS puis sur toutes les nouvelles BTS.
- **f3 = coût** — `mean_cost_to_infrastructure` : distance moyenne de chaque nouvelle BTS à la BTS
  réelle existante la plus proche, le proxy route/infrastructure retenu avec l'utilisateur (le
  sujet demande un coût « proportionnel à la distance aux routes » sans jamais fournir de jeu de
  données routier — interrogé explicitement sur ce point, décision : les vraies BTS s'agrègent en
  pratique le long des routes/infrastructures, donc la distance à la BTS existante la plus proche
  approxime la distance à la route, sans appel réseau supplémentaire par évaluation — cette
  fonction sera appelée des milliers de fois par NSGA-II, même raisonnement que
  `codec_fitness`/`estimate_wer_proxy` en Module A).

**Terrain par défaut mis en cache** : `_get_default_terrain()` charge et échantillonne
`docs/208.csv` (Aveyron, 500 BTS, `seed=42`) une seule fois par processus — même schéma que
`_reference_cache` en Module A, testé de la même façon (monkeypatch de
`opencellid_loader.load_opencellid_csv`, vérification d'un seul appel à travers deux évaluations
de fitness, cf. `TestCodecFitness.test_reference_signal_is_generated_once_and_cached` en Module A).
Tous les autres tests passent un terrain synthétique explicite — aucun test ne dépend du vrai
`docs/208.csv`.

**Résultat sur données réelles** (500 BTS Aveyron, `seed=1`, 300 points de test) : une BTS placée au
centre du terrain donne `f1≈-36,3`, `f2≈0` (zone rurale, peu de voisins à moins d'1 km),
`f3≈4660 m` ; la même BTS placée dans un coin donne `f1≈-36,0` (couverture quasi inchangée,
dominée par les 500 BTS réelles existantes), `f3≈2336 m` (plus proche d'un cluster d'infrastructure
existant à cet endroit précis) — le placement affecte nettement le coût, peu la couverture globale
avec une seule BTS ajoutée à 500 existantes, cohérent avec l'intuition.

**Tests** : décodage aux bornes et avec clipping, longueur impaire lève une erreur, couverture avec
rayon énorme/minuscule, reproductibilité par seed ; interférence plus forte entre BTS proches
qu'éloignées, nulle sans voisin, nulle sur un tableau vide ; coût quasi nul si la nouvelle BTS
coïncide avec une existante, croissant avec la distance, nul sur un tableau vide ; breakdown
complet (clés, valeurs finies, `f1 = -coverage_pct`, reproductibilité), tuple `bts_coverage_fitness`
cohérent avec `bts_coverage_fitness_components`, mise en cache du terrain par défaut.

**Résultats de test** : 19 nouveaux tests (94 au total pour `module_c`), tous verts. Couverture
`fitness.py` : **100 %**.

---

## Module D — QoS/QoE (modèle E ITU-T G.107, mesures STUN, dashboard Rich)

### `model_e.py`

**Contexte** : le sujet donne le squelette du facteur R (`R = R0 − Is − Id − Ie + A`, `R0=93.2`,
`A=10`, table `Ie` par codec ≈ {GSM:20, AAC:25, Opus:7}) mais ne donne ni la formule de `Id`
(délai), ni celle de `Ie` effectif (perte de paquets), ni la conversion `R → MOS` — seulement
« conversion par formule ITU-T ». Plutôt qu'inventer ces formules, ce fichier utilise les formules
standard, publiées ITU-T G.107 (les mêmes que celles couramment citées par Cisco et al. pour le
E-model simplifié) :

- **`delay_impairment(delay_ms)`** — `Id = 0.024·d + 0.11·(d − 177.3)·H(d − 177.3)` (`H` = échelon
  de Heaviside), la formule standard : linéaire jusqu'à ~177 ms, puis une pente plus forte au-delà
  (seuil correspondant approximativement au délai à partir duquel un interlocuteur commence à
  percevoir une gêne conversationnelle, pas un chiffre choisi arbitrairement ici).
- **`effective_equipment_impairment(ie_base, loss_pct, bpl)`** — `Ie,eff = Ie + (95 − Ie)·Ppl/(Ppl/Bpl + 2)`,
  la formule standard ajustée à la perte de paquets. **C'est le point d'entrée choisi pour la perte
  de paquets dans le modèle** : la table `Ie` du sujet est plate (un seul chiffre par codec, sans
  terme de perte), et `Is` est documenté par le sujet comme spécifique au codage/écho, pas à la
  perte — `Ie,eff` est donc le seul levier restant, cohérent avec son usage standard ITU-T.
  `DEFAULT_BPL = 10.0` est une constante unique (pas de valeur par codec, faute de base) —
  hypothèse documentée, même esprit que le `DEFAULT_PACKET_LOSS_RATE` unique du Module A.
- **`r_factor(codec, delay_ms, loss_pct, is_impairment=0.0, bpl=DEFAULT_BPL)`** — combine le tout,
  `R` borné à `[0, 100]`. `Is` vaut `0.0` par défaut (aucun trajet d'écho n'est modélisé ailleurs
  dans le projet) — hypothèse documentée, injectable pour un futur modèle d'écho.
- **`r_to_mos(r)`** — mapping cubique standard ITU-T G.107 : `MOS=1` pour `R≤0`, `MOS=4.5` pour
  `R≥100`, sinon `MOS = 1 + 0.035·R + R·(R−60)·(100−R)·7×10⁻⁶`.
- **`mos_from_conditions(codec, delay_ms, loss_pct, ...)`** — enchaîne les deux, utilisé par tout
  le reste du module.

**Vérifié à la main** : `r_factor("opus", delay_ms=0, loss_pct=0)` = `93.2 − 0 − 0 − 7 + 10 = 96.2`
(Ie,eff = Ie à perte nulle) ; `effective_equipment_impairment(20, 5, bpl=10)` =
`20 + 75·5/2.5 = 170` ; `r_to_mos(93.2)` ≈ `4.409286`.

**Résultats de test** : 24 tests, tous verts. Couverture `model_e.py` : **100 %**.

### `stun_probe.py`

**Choix — STUN via `socket` stdlib, pas de librairie tierce** (spec ligne 234-239, conforme à
`docs/SUJET.md` §1.2 : `stun.l.google.com:19302`, libre d'accès, sans inscription) : construction
manuelle du paquet RFC 5389 (en-tête 20 octets : type `0x0001` Binding Request, longueur `0x0000`,
cookie magique `0x2112A442`, ID de transaction 96 bits aléatoire via `os.urandom(12)`) et parsing
manuel de la réponse (`0x0101` Binding Success Response, cookie + ID de transaction assortis).
`measure_one_rtt` chronomètre l'aller-retour avec `time.perf_counter()` ; `sock` injectable, même
principe que `session`/`client` en Module C/B.

**Choix — taux de perte mesuré, pas injecté** : le sujet demande un « taux de perte simulé sur 20
mesures ». Plutôt que d'ajouter un paramètre de probabilité de perte synthétique arbitraire (un
degré de liberté supplémentaire sans base réelle), le taux de perte de `measure_rtt_jitter` est
directement la fraction des 20 requêtes UDP réelles qui expirent ou échouent à parser — un vrai
aller-retour UDP vers un vrai serveur produit déjà de la vraie perte dans des conditions réelles.
Un taux de perte totalement contrôlable pour la courbe MOS=f(perte) exigée par le sujet est géré
séparément, directement par `correlation.py`/`session_sim.py` (appel direct à `model_e` avec un
`loss_pct` balayé arbitrairement), sans dépendre de ce que `stun_probe` mesure réellement à
l'instant T.

**Tests** : `FakeStunSocket` (même principe que `FakeIpinfoSession`/`FakeOverpassSession` en
Module C) — écho d'une réponse Binding Success valide reconstruite à partir de l'ID de transaction
du dernier paquet envoyé, avec possibilité de simuler un timeout à des indices d'appel choisis.
Couvre : en-tête du paquet de requête, ID de transaction aléatoire à chaque appel, validation de
réponse (assortie / ID différent / mauvais type / trop courte), RTT positif sur réponse valide,
`None` sur timeout et sur réponse invalide (garbage), comptage correct de la perte sur des pertes
partielles/totales, non-fermeture d'un socket injecté vs. fermeture d'un socket créé en interne
(vérifié en monkeypatchant `socket.socket` pour éviter tout vrai trafic réseau dans la suite
pytest).

**Résultats de test** : 14 nouveaux tests (38 au total pour `module_d`), tous verts. Couverture
`stun_probe.py` : **100 %**.

### `session_sim.py`

**Choix — congestion modélisée par déficit de bande passante** (spec ligne 233 : « injection
métriques → modèle E ») : `simulate_session(codec, bandwidth_kbps, required_bandwidth_kbps,
rtt_ms, jitter_ms, seed)` calcule un ratio de déficit
`max(0, (requis − alloué) / requis)`, puis en dérive à la fois une perte de paquets croissante et
un délai de mise en file croissant — un lien congestionné dégrade d'abord le délai (les tampons se
remplissent) avant de perdre des paquets (les tampons débordent), comportement qualitatif réel
plutôt qu'un choix arbitraire. `delay_ms = rtt/2 + gigue + délai_de_congestion` (estimation
aller-simple + tampon de gigue + congestion) ; un bruit gaussien seedé est ajouté à la perte pour
plus de réalisme, le tout borné/clippé.

**Hypothèses documentées, sans base numérique dans le sujet** : `MAX_CONGESTION_LOSS_PCT = 30.0`
(perte ajoutée à un déficit de bande de 100 %) et `MAX_CONGESTION_DELAY_MS = 150.0` (délai de
congestion ajouté au même déficit) — constantes uniques choisies pour donner une dégradation
perceptible mais pas caricaturale, même esprit que `DEFAULT_PACKET_LOSS_RATE` en Module A.

**Résultats de test** : 8 nouveaux tests (46 au total pour `module_d`), tous verts, dont
reproductibilité par seed, cohérence de `r_factor`/`mos` avec un appel direct à `model_e`,
déficit croissant → perte croissante → MOS décroissant, cas limite `required_bandwidth_kbps=0`
(pas de déficit). Couverture `session_sim.py` : **100 %**.

### `dashboard.py`

**Choix** : `run_dashboard(n_iterations=20, refresh_s=0.5, csv_path=..., sock=None, codec="opus",
sleep=time.sleep, console=None)` — `rich.live.Live` réaffiche une `Table` (RTT/Gigue/Perte/MOS/
Codec, exactement les champs du sujet ligne 238-239) à chaque itération, une vraie mesure STUN par
ligne (`stun_probe.measure_one_rtt`), gigue/perte calculées en cumulatif sur les échantillons vus
jusqu'ici (même logique que `stun_probe.measure_rtt_jitter`, mais mise à jour au fil de l'eau
plutôt qu'en un seul bloc final). `sleep`/`console` injectables, même schéma que
`compare.measure_real_delivery` en Module B (`sleep`/`now` injectables pour ne jamais vraiment
attendre en test) — ici `console` en plus, pour rediriger le rendu Rich vers un buffer mémoire en
test plutôt que polluer la sortie de la suite pytest.

**Export CSV automatique, pas différé** (spec ligne 239) : le fichier est ouvert une seule fois en
écriture, une ligne est écrite et `flush`ée à chaque itération plutôt qu'accumulée en mémoire puis
écrite à la fin — une exécution interrompue en cours de route laisse quand même les données
partielles sur disque. Dossier de sortie créé s'il manque, même garde-fou que `map_viz.save_map_html`
en Module C.

**Tests** : `FakeStunSocket` du fichier `stun_probe.py` réutilisée telle quelle (aucune nouvelle
fausse socket nécessaire). Couvre : colonnes/valeurs rendues (rendu Rich capturé dans un buffer
mémoire, `nan` jamais affiché tel quel), une ligne CSV par itération avec les bons en-têtes,
création du dossier de sortie manquant, non-fermeture d'un socket injecté vs. fermeture d'un
socket créé en interne, perte partielle reflétée dans les lignes suivantes, `sleep` appelé
`n_iterations − 1` fois (jamais après la dernière ligne).

**Résultats de test** : 9 nouveaux tests (55 au total pour `module_d`), tous verts. Couverture
`dashboard.py` : **100 %**. Dépendance ajoutée : `rich` (`uv add rich`), seule nouvelle dépendance
du Module D.

### `correlation.py`

**Choix — réutilisation directe de `module_a.visualize.plot_correlation_matrix`, pas de
réimplémentation** (spec ligne 241-243) : cette fonction est déjà générique sur n'importe quel
mapping `{label: {métrique: valeur}}` (documenté explicitement dans son propre docstring en
Module A) — `module_d.correlation.plot_correlation_matrix` est un simple alias vers la fonction
du Module A plutôt qu'une copie. Seule imperfection assumée : le titre de la heatmap reste
« Corrélation PESQ × MUSHRA × WER » même une fois la clé `"mos"` ajoutée — défaut cosmétique
connu, pas jugé suffisant pour justifier une modification du Module A (déjà complet et testé)
depuis une commande du Module D.
- `build_correlation_table(pesq_by_codec, wer_by_codec, mushra_summary, mos_by_codec)` — appelle
  `module_a.visualize.build_correlation_table` (fusion PESQ/WER/MUSHRA existante) puis y ajoute la
  clé `"mos"` par codec.

**Choix — seuils R annotés en MOS, pas en R** (spec ligne 231-232 : « Seuils annotés : R < 60
(insatisfaisant), 60-80 (acceptable), > 80 (bon) ») : l'axe des ordonnées des deux courbes est en
MOS, pas en R — les seuils sont donc convertis via `model_e.r_to_mos(60)`/`r_to_mos(80)` avant
d'être tracés en lignes horizontales, plutôt que de changer l'axe pour du R brut (le MOS est ce
que le sujet demande de tracer explicitement, « Tracer MOS = f(délai)... »).
- `plot_mos_vs_delay(codecs, delay_range_ms, loss_pct=0.0)` / `plot_mos_vs_loss(codecs,
  loss_range_pct, delay_ms=0.0)` — une courbe par codec (AAC/GSM/Opus), seuils R annotés,
  bornes MOS `[1.0, 4.5]` fixes sur l'axe Y pour rester comparables entre les deux figures.

**Résultats de test** : 6 nouveaux tests (61 au total pour `module_d`), tous verts, dont un test
d'identité (`plot_correlation_matrix is module_a.visualize.plot_correlation_matrix`, pas une
copie) et une décroissance du MOS avec le délai/la perte pour chaque courbe. Couverture
`correlation.py` : **100 %**.

### `fitness.py`

**Contexte — dernier fichier substantiel du Module D**. Expose le contrat `qos_fitness(chromosome)`
pour le Pb3 du Module F (DE via `scipy.optimize.differential_evolution` vs. PSO via `pyswarm`,
spec lignes 342-350) : 10 utilisateurs simultanés (profils VoIP/SMS/streaming, chacun avec une QoS
minimale requise), chromosome `[codec_u1, bw_u1, ..., codec_u10, bw_u10]` (20 gènes), contrainte
dure de capacité réseau totale partagée.

**Écart avec le sujet, résolu comme en Module A** : le sujet dit `qos_fitness(chromosome) → MOS
moyen` (un scalaire nu, spec ligne 243) alors que les objectifs du Pb3 sont formulés comme une
paire `(−MOS moyen, bande totale)` (spec ligne 349). Résolu de la même façon que l'écart analogue
du Pb1 en Module A (le terme WER) : un seul score scalaire pondéré, avec pénalité de contrainte
dure pour tout dépassement de capacité totale, poids configurables. Le contrat reste un `float` nu
(pas un tuple), puisque DE/PSO sont des solveurs mono-objectif — contrairement au Pb2 (`module_c
.fitness.bts_coverage_fitness`, qui retourne bien un tuple, pour NSGA-II).

**Profils utilisateurs — hypothèse documentée, faute de valeurs numériques dans le sujet** : le
sujet nomme les types de profil (VoIP/SMS/streaming) et dit « chacun avec QoS requise (MOS
minimum) » sans jamais donner de chiffres. `DEFAULT_USER_PROFILES` (3 VoIP à 32 kbps/MOS min 3.5,
3 SMS à 1 kbps/MOS min 3.0, 4 streaming à 64 kbps/MOS min 3.8) sont des valeurs plausibles choisies
et documentées, pas mesurées. `DEFAULT_TOTAL_CAPACITY_KBPS` est fixée à 70 % de la demande totale
si tout le monde recevait exactement sa bande requise — volontairement en-dessous du besoin total,
sinon le problème d'allocation n'a pas de vrai compromis à résoudre.

**Aucun vrai appel STUN dans la boucle de fitness** : DE/PSO peuvent évaluer cette fonction des
milliers de fois — ouvrir une socket UDP à chaque évaluation serait lent et rendrait la fitness
non reproductible, même raisonnement que le proxy WER en Module A. `rtt_ms`/`jitter_ms` par défaut
sont des constantes fixes documentées (`DEFAULT_BASELINE_RTT_MS=40`, `DEFAULT_BASELINE_JITTER_MS=5`,
valeurs plausibles pour un réseau mobile), pas une mesure réelle mise en cache — **simplification
par rapport au plan initial** : contrairement au signal de référence TTS du Module A ou au terrain
BTS du Module C, il n'y a ici rien de coûteux à mettre en cache, donc pas de cache module-level. Un
appelant voulant la vraie mesure sur le meilleur individu final peut simplement passer les
`rtt_ms`/`jitter_ms` réels issus de `stun_probe.measure_rtt_jitter()` en paramètres — plus simple
qu'une fonction `rtt_fn` injectable séparée, sans rien perdre du besoin (validation ponctuelle
sim-vs-réel sur les meilleurs individus).

**Tests** : décodage (longueur invalide lève une erreur, snapping modulo du gène codec, bande
passante clippée aux deux bornes, profils préservés dans l'ordre), breakdown complet (clés,
valeurs finies), absence de violation de capacité en-dessous du seuil vs. violation positive
au-dessus (valeur exacte vérifiée), absence de violation de MOS minimum quand généreusement
approvisionné (seeded, seuils MOS min abaissés pour rester robuste au bruit de
`session_sim.simulate_session`) vs. violation positive quand affamé, reproductibilité par seed,
formule pondérée vérifiée à la main, cohérence `qos_fitness`/`qos_fitness_components`.

**Résultats de test** : 19 nouveaux tests (80 au total pour `module_d`), tous verts. Couverture
`fitness.py` : **100 %**. Suite complète du dépôt (338 tests, tous modules) toujours verte, **96 %**
de couverture globale — aucune régression introduite dans les modules A–C.

## En attente / pas encore implémenté

Module A est complet (tous les fichiers de `docs/SUJET.md` §3 sont implémentés et testés).
Module B est complet, code, tests et validation réelle : tous les fichiers de `docs/SUJET.md` §3
implémentés et testés (96 % de couverture — 100 % hors les deux scripts manuels
`manual_twilio_check.py`/`manual_twilio_verify_check.py`, non exercés par pytest par conception),
avec deux envois réels réussis sur le compte trial Twilio — `manual_twilio_verify_check.py` (OTP
via l'API Verify, `status=approved`) et `manual_twilio_check.py` (SMS via l'API Messages,
`accept_latency_s=2.09`, `delivery_latency_s=5.39`, `final_status=delivered`) — tous deux détaillés
dans les sections « Complément » de `twilio_client.py` ci-dessus. `compare.py` dispose maintenant
d'un vrai `RealDeliverySample` pour la comparaison sim/réel du rapport final (§8 Q2), qui pourra
aussi s'appuyer sur les deux contraintes trial découvertes au passage (politique de template SMS,
`GET /Messages/{Sid}` 403 alors que `GET /Messages` fonctionne) comme causes structurelles de
l'écart. Rien ne reste en attente pour Module B. Module C est complet côté code et tests : tous les
fichiers de `docs/SUJET.md` §3 implémentés et testés (94 tests, 96 % de couverture — 100 % hors les
deux scripts manuels `manual_ipinfo_check.py`/`manual_nominatim_check.py`, non exercés par pytest
par conception), et validés contre les deux vraies API : `manual_nominatim_check.py` (10 POI réels
près de Rodez, voir la section `lbs_poi.py` ci-dessus) et `manual_ipinfo_check.py` (IP publique de
la machine résolue en Toulouse, Occitanie, FR, voir la section `ipinfo_client.py` ci-dessus). Rien
ne reste en attente pour Module C côté code/tests/validation réelle ; reste seulement, non
bloquant : tableau comparatif Cell-ID/TOA/Wi-Fi/IP et carte Folium démonstrative
(`results/module_c_demo.html`) à intégrer au rapport final (§8, Module C 2-3 p.). Module D en
cours : `model_e.py`, `stun_probe.py`, `session_sim.py`, `dashboard.py`, `correlation.py`,
`fitness.py` faits — code et tests complets. Reste seulement `manual_stun_check.py` (validation
réelle) avant que Module D soit entièrement clos. Modules E–F, non commencés.
