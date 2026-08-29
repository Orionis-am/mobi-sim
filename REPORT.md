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

## En attente / pas encore implémenté

Module A est complet (tous les fichiers de `docs/SUJET.md` §3 sont implémentés et testés).
Modules B–F, non commencés.
