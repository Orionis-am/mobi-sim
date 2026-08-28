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

Pas encore testé contre la vraie API Whisper (nécessite `OPENAI_API_KEY`, coût réel) — à faire une
fois `fitness.py`/tests d'intégration en place, avec des clips courts pour limiter la dépense
(~0,006 USD/min, cf. CLAUDE.md).

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

---

## En attente / pas encore implémenté

- `module_a/mushra_sim.py` — simulation panel MUSHRA (IC 95% par bootstrap).
- `module_a/visualize.py` — graphiques MUSHRA/PESQ/WER + matrice de corrélation PESQ×MUSHRA×WER.
- `module_a/fitness.py` — contrat `codec_fitness(chromosome)` pour le Module F.
- `module_a/test_module_a.py` — tests unitaires (objectif ≥75% de couverture, cf. CLAUDE.md).
- Modules B–F, non commencés.
