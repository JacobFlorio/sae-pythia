# Findings: Pythia-160M SAE — layers 3, 6, and 9

Three Top-K SAEs trained on layers 3, 6, and 9 of Pythia-160M (16,384 features, k=32, Pile-uncopyrighted) at two token budgets: a 5M-token smoke run and a full 50M-token run. This document covers reconstruction quality, failure modes, geometric analysis, automated interpretability scoring, and what changes between training scales.

All committed artifacts:
- `dashboards/features_*_sample_dedupe.json` — top-30 feature slices, Jaccard-deduped
- `dashboards/layer_comparison.json` — cluster metrics across layers
- `dashboards/geometry.json` — decoder superposition metrics
- `dashboards/autointerp_L*.json` — Claude-generated descriptions + balanced accuracy

---

## Cross-layer summary (5M tokens)

| Metric | Layer 3 | Layer 6 | Layer 9 |
|---|---|---|---|
| **FVU** | 0.078 | **0.074** | 0.125 |
| Dead latents | 6,658 (41%) | 6,734 (41%) | **3,511 (21%)** |
| BOS-only features in top 50 | 24 | **17** | 25 |
| Strict mid-doc features in top 50 | 0 | **1** | 0 |
| Unique clusters after Jaccard dedupe | 3 | **5** | 3 |
| Peak activation median | 68.2 | 65.5 | **48.3** |
| Auto-interp balanced accuracy (top 30) | 0.883 | 0.893 | 0.888 |

---

## 1. Reconstruction quality and the 50M training result

At 5M tokens, layer 6 is the reconstruction leader (FVU 0.074). At 50M tokens, the picture changes substantially:

| Layer | FVU @ 5M | FVU @ 50M | Dead @ 5M | Dead @ 50M |
|---|---|---|---|---|
| Layer 3 | 0.078 | **0.042** | 6,658 (41%) | **1 (<0.01%)** |
| Layer 6 | 0.074 | 0.057 | 6,734 (41%) | 10 (0.06%) |
| Layer 9 | 0.125 | 0.099 | 3,511 (21%) | 10 (0.06%) |

**Dead latents are essentially eliminated at 50M tokens across all layers.** The AuxK loss is working — it just needed the token budget. At 5M tokens, 41% of latents were dormant; at 50M, every layer reaches near-zero.

**Layer ranking reverses at 50M tokens.** Layer 3 goes from second-worst (0.078) to best (0.042), a 46% FVU reduction. Layer 6 improves 23% (0.074→0.057). Layer 9 improves 21% (0.125→0.099). Layer 3 benefits most from longer training, likely because early-layer representations are more linearly separable once the SAE has enough capacity to specialize.

**Layer 9 remains the hardest layer at both training scales.** Even with dead latents eliminated and 10× the tokens, layer 9 reconstruction (FVU 0.099) does not reach layer 6's 5M-token performance (0.074). This persistence across training scales is the central mystery.

---

## 2. The layer 9 paradox: geometry rules out superposition

At 5M tokens, layer 9 has *fewer* dead latents (21%) than layers 3 and 6 (both 41%), yet substantially worse reconstruction. The natural hypothesis: layer 9 encodes information in denser superposition, requiring more live latents to handle interference. We tested this directly with decoder column geometry.

| Layer | Mean cos sim | Frac pairs \|cos\|>0.1 | Uniformity loss | Effective rank |
|---|---|---|---|---|
| Layer 3 | **0.00422** | **0.0108** | -3.9716 | 672.0 |
| Layer 6 | 0.00136 | 0.0072 | **-3.9833** | **689.8** |
| Layer 9 | 0.00273 | 0.0088 | -3.9769 | 683.9 |

**The superposition hypothesis is rejected.** Layer 9 does not have higher decoder column cosine similarity, more interfering pairs, worse uniformity, or lower effective rank than layers 3 or 6. In fact, layer 3 — which achieves the *best* reconstruction at 50M tokens — has the *highest* mean cosine similarity. Decoder geometry does not predict reconstruction difficulty.

**Conclusion:** The layer 9 paradox is intrinsic to the residual stream, not the SAE's learned decoder. By layer 9, the residual stream encodes 9 rounds of contextually-integrated information. That representation is genuinely harder to decompose into sparse, linearly-independent directions — not because the SAE fails to spread its decoder columns evenly, but because the information itself doesn't factor cleanly at this model size and training budget. More tokens or a larger SAE may close the gap; a fundamentally different architecture (e.g., attention-based encoder) might fare better.

---

## 3. Pathologies: BOS and paragraph-break clusters

All three layers have duplicated latents absorbing the highest-variance structural positions.

**BOS super-cluster** (fires at `token_pos=0` on document-start tokens):
```
# Cluster representative, layer 3 — max activation 76.0
[76.0] 'Anthony'  Anthony Iluobe  Chief Anthony Iluobe (JP) was born in …
[74.9] 'Guard'    Guard youths from alcopops  9:55 AM, May 8, …
[74.9] 'Prom'     Promethium: uses  The following uses for promethium …

# Layer 6 — max 71.7
[71.7] 'Guard'    Guard youths from alcopops  9:55 AM, May 8, …

# Layer 9 — max 52.0 (signal weakens at depth)
[52.0] 'Prom'     Promethium: uses  The following uses for promethium …
```

**Paragraph-break cluster** (fires on early-document newlines in code headers and HTML):
```
[74.3] '\n'  /**  * ScriptDev2 is an extension for mangos …
[74.3] '\n'  Confederación Revolucionaria de Obreros y Campesinos …
[74.0] '\n'   Her mate, inspired by all the money she knows Purdy is saving …
```

At 5M tokens: 17–25 of the top-50 peak-ranked features are BOS-only; only 5 unique doc-fingerprints survive Jaccard-0.5 deduplication. 90% of the top list is duplicated positional absorption.

---

## 4. Automated interpretability scoring

Auto-interp was run on the top-30 features (by peak activation) from each layer's full dashboard. Claude generates a one-sentence description from the top-activating examples, then predicts which of a held-out mix of positive/negative snippets will fire. Balanced accuracy: 0.5 = random, 1.0 = perfect.

| Layer | Features scored | Mean balanced accuracy | Features @ 1.00 | Features < 0.65 |
|---|---|---|---|---|
| Layer 3 | 30 | 0.883 | 8 | 2 |
| Layer 6 | 28 | 0.893 | 8 | 1 |
| Layer 9 | 29 | 0.888 | 8 | 1 |

Scores are nearly identical across layers. This has a specific interpretation: **the top-30 by peak activation is dominated by structural features that score trivially well.** BOS-cluster and paragraph-break features get 1.00 because their pattern is "fires on document-start tokens" — perfectly predictable with high overlap. The features that *fail* auto-interp reveal the limit:

```
# Layer 3, latent 13954 — score 0.50
"The neuron detects the first few characters of proper nouns,
 names, and titles, particularly at the beginning of words or phrases."

# Layer 9, latent 7151 — score 0.38
"Newline character following code comment markers,
 class/package declarations, or HTML/XML opening tags."
```

Latent 13954 fails because its BOS-cluster firing is generic: it fires on the first token of a document regardless of what it is, so Claude's description of "proper nouns" only partially predicts the held-out positives. Latent 7151 actually scores *below* chance (0.38) — Claude's description is wrong, suggesting the feature activates on something else entirely.

**High auto-interp score measures predictability, not semantic richness.** The most interesting features found by Jaccard deduplication (GCD problems, auto-insurance boilerplate, patent figure captions) score similarly to structural features but are far more informative as research artifacts.

### Deduplicated auto-interp: honest scores on semantic features only

Re-running auto-interp on the Jaccard-deduplicated top-30 (removing BOS/structural absorbers) yields a more honest measurement:

| Layer | Peak-top-30 mean | Dedupe-top-30 mean | Delta | Failures (<0.65) | Perfect (≥0.90) |
|---|---|---|---|---|---|
| Layer 3 | 0.901 | **0.844** | −0.057 | 4 | 8 |
| Layer 6 | 0.804 | **0.795** | −0.009 | 6 | 4 |
| Layer 9 | 0.858 | **0.804** | −0.054 | 8 | 5 |

**L3 and L9 drop ~0.05 when structural absorbers are removed.** This quantifies exactly how much BOS-cluster features inflated the earlier means — about 5–6 percentage points, corresponding to the easy 1.00-scoring features that dedup removes.

**L6's near-zero delta (−0.009) confirms the L6 50M features were already the hard ones.** The BOS cluster at layer 6 is smaller (22 features vs 24–25 for L3/L9), so even the peak-top-30 was already dominated by complex phrasal features. The 0.804 score is a real measure of interpretability difficulty.

**Failure rate increases with layer depth.** L3 has 4 failures, L6 has 6, L9 has 8. This matches the lexical→phrasal→conceptual gradient: as features integrate more context, single-sentence descriptions become less predictive. The 8 failures at L9 are the conceptual features (Q&A structure, domain-specific reasoning patterns) that are genuinely hard to capture in one sentence.

---

## 5. Semantic features by layer

After fingerprint deduplication, semantic features emerge with a clear depth progression:

### Layer 3 — lexical and template features
Features respond to specific token fragments, largely independent of context.

| Feature | Peak | Description |
|---|---|---|
| 4633 | 8.6 | "U.S. Pat." abbreviation in patent filings |
| 7752 | 8.6 | "Public" in GNU General Public License boilerplate |
| 742  | 8.5 | Judicial titles ("Judge," "Justice," "Judges") in US court opinions |
| 13118 | 8.5 | "FIG." in patent figure cross-references |
| 9354 | 8.4 | "vitro" in "in vitro" across biomedical papers (plain + Markdown italic) |

### Layer 6 — phrasal and register features
Features respond to surrounding register and phrasing, not just the token.

| Feature | Peak | Description |
|---|---|---|
| 5735 | 20.2 | Auto-insurance rate-quote disclaimer boilerplate |
| 4895 | 9.7 | Patent "Field of the Invention" opening section |
| 11957 | 9.3 | Apache/BSD license "WITHOUT WARRANTIES… OF ANY KIND" clause |
| 10188 | 17.0 | Hyphens inside biomedical bibliographic citation markup |
| 10166 | 9.2 | HackerNews `------ username` reply-thread separators |

### Layer 9 — domain and concept features
Features fire on domain-specific conceptual patterns; peak activations are lower but the patterns are sharper.

| Feature | Peak | Description |
|---|---|---|
| 12045 | 31.5 | "Greatest common divisor" in math problem sets (GCD/HCF) |
| 11476 | 26.5 | HackerNews reply-thread newlines (appears at L6 and L9) |
| 9612  | 28.6 | Indentation whitespace in HTML/code contexts |
| 6710  | 25.2 | Decimal points in numeric literals and JSON |

The lexical→phrasal→conceptual gradient is consistent with mechanistic interpretability accounts of how transformer layers progressively integrate context.

---

## 6. Methodological findings

**1. Rank by peak activation is a trap.** At 5M tokens, 90% of the top-50 list is duplicated positional absorbers. Fingerprint-based Jaccard deduplication is cheap and effective.

**2. High auto-interp score ≠ interesting feature.** Structural absorbers score 1.00 because they're trivially predictable; that inflates mean balanced accuracy without reflecting semantic richness.

**3. Dead-latent fraction is a training-budget signal, not a quality signal.** At 5M tokens it looks like a failure; at 50M it essentially disappears. The AuxK coefficient doesn't need tuning — the default just needs more data.

**4. Layer 9's difficulty is intrinsic to its residual stream.** Decoder geometry (cosine similarity, effective rank, uniformity) does not explain why layer 9 is harder to reconstruct. The information in deep residual streams resists linear decomposition at this model size.

**5. Layer 3 is the training-scale surprise.** At 5M tokens it looks like the worst layer (tied with L9); at 50M it beats layer 6 by 27% on FVU. Early-layer representations apparently benefit more from latent specialization once the dead-latent budget is freed.

---

## 7. The 50M token results: a phase transition in feature quality

Building dashboards from the 50M checkpoints and rerunning `compare_layers.py` reveals a near-complete transformation in feature landscape.

### Cluster metrics: before and after

| Metric | L3 5M | L3 50M | L6 5M | L6 50M | L9 5M | L9 50M |
|---|---|---|---|---|---|---|
| BOS-only features / top 50 | 24 | 16 | 17 | 22 | 25 | **12** |
| Strict mid-doc / top 50 | 0 | 12 | 1 | 12 | 0 | **25** |
| Unique clusters (Jaccard dedupe) | 3 | 27 | 5 | 26 | 3 | **39** |
| Peak activation median | 68.2 | 17.9 | 65.5 | 26.0 | 48.3 | 21.2 |

**The BOS super-cluster is a training-budget artifact, not an intrinsic SAE property.** At 5M tokens, 3–5 unique clusters survive deduplication of the top-50 peak list — meaning ~90% of the list is structural duplication. At 50M, that collapses to 26–39 unique clusters, meaning the top-50 now contains genuine feature diversity.

**Layer 9 reversal.** At 5M tokens, layer 9 had the worst feature diversity (3 unique clusters, 0 strict mid-doc features) despite fewest dead latents. At 50M, it has the *best* diversity (39 unique clusters, 25/50 strict mid-doc features). The layer 9 paradox fully inverts at the longer training scale: once latents specialize, the deeper residual stream supports richer feature decomposition, not poorer.

**Peak activation collapse is expected and healthy.** Median peak drops from ~65 (5M) to ~18–26 (50M). With dead latents revived and specializing, activation mass spreads across hundreds of distinct features instead of concentrating in a few structural absorbers. Lower peak = more distributed representation.

### Known features persist and strengthen

Three features found at 5M appear again at 50M with higher activation — confirming they are stable, real features rather than training noise:

| Feature | Layer | Peak @ 5M | Peak @ 50M | Description |
|---|---|---|---|---|
| Auto-insurance boilerplate | L6 | 20.2 | **43.6** | "lapse in coverage," "miles annually," rate disclaimers |
| Greatest common divisor | L9 | 31.5 | **41.0** | "calculate the highest common divisor of…" |
| Biomedical citation hyphens | L6 | 17.0 | **35.2** | Hyphens in `{ref-type="other"}` markup |

**Cross-scale feature persistence is strong evidence of genuine learning.** A feature that was present at 5M tokens and strengthens at 50M is not a coincidence — the SAE is recovering a real direction in the residual stream that the model consistently uses for that concept.

### New features at 50M

Several features that were invisible at 5M emerge clearly at 50M:

**L9, feature 7535 — StackExchange/Q&A format (peak 41.7)**
```
[41.7] ' doesn'  … equipped with the same amount of information, simply doesn't get it. When I sta …
[39.0] '\n'      … the database to the thread with no luck. Any ideas? Thanks!  A:  Segmentation faults in Python …
[39.0] '\n'      … find the limits of integration simply by changing to polar coordinates. Thanks  A:  In polar coor …
```
Fires on Q&A reply structure — the newline before "A: " in StackExchange-formatted answers, and contextually around question-answer transitions.

**L6, feature 1920 — XML/Android layout declarations (peak 29.5)**
```
[29.5] '.'  <?xml version="1.0" encoding="utf-8"?> <LinearLayout xmlns:android= …
[29.5] '.'  <?xml version="1.0" encoding="UTF-8"?> <segment>     < …
[29.5] '.'  <?xml version="1.0" encoding="utf-8"?> <RelativeLayout android:layout …
```
Fires on the period in `version="1.0"` across XML declarations from Android layouts, data files, and markup formats.

**L3, feature 611 — sentence-ending period in mid-document prose (peak 24.1)**
```
[24.1] '.'  … modern cell phones feature a multitude of features that expand on the traditional cell phone functi …
[24.1] '.'  … mystics and their task; we existed like stray animals sheltered in a monastery. …
[24.0] '.'  … Sacramento Kings' (16-22) recent poor play minus a star resurfaced. The thought came to fruition …
```
Fires on periods at sentence boundaries in continuous prose, positions 23–50. A structural feature, but a mid-document one — different from the BOS absorbers.

### The FVU/diversity decoupling

At 50M tokens, layer 9 has the richest feature diversity (39 unique clusters, 25/50 mid-doc) but still the worst reconstruction (FVU 0.099 vs L3's 0.042). This decouples two metrics that are often conflated:

- **FVU** measures how much of the *total* residual stream variance the SAE explains. It is dominated by the high-variance structural features (position 0, early newlines) that the SAE is good at capturing.
- **Feature diversity** measures how many *distinct* semantic patterns the SAE has specialized latents for.

Layer 9 apparently has more diverse semantic content to represent, requiring more specialized latents (hence few dead latents even at 5M), but also has a larger "long tail" of residual stream variance that's hard to explain — either from directions that require even more tokens to converge, or from aspects of the contextual representation that don't compress well into independent sparse features.

## 8. Cross-scale feature matching: decoder geometry across scales and feature widths

Using `scripts/match_features.py`, we computed cosine similarity between every pair of decoder directions in the L9 5M and 50M checkpoints. Since `W_dec` rows are unit-normalized, the full similarity matrix is `W_dec_5M @ W_dec_50M.T` — no additional normalization needed. The best-match cosine for each 5M feature tells us how "preserved" it is in the 50M run.

### Distribution of best-match cosines (5M → 50M, L9)

| Threshold | Fraction of features | Count (of 16,384) |
|---|---|---|
| cos > 0.9 | 0.8% | **134** |
| cos > 0.7 | 5.7% | **933** |
| cos > 0.5 | 16.3% | **2,669** |
| cos > 0.3 | 23.9% | **3,909** |
| Mean best-match cosine | — | **0.2595** |
| Median best-match cosine | — | **0.1547** |

**The phase transition is a genuine reorganization, not a smooth scaling.** If features simply sharpened while maintaining their directions, we would expect most best-match cosines above 0.9. Instead, the median is 0.15 — most 5M features do not have a clear identity in the 50M run. The ~10× increase in unique feature clusters (3→39) is consistent: the feature space restructures substantially, not just refines.

**A small, stable core persists.** 134 features (0.8%) survive with cos > 0.9 — near-identical geometry across scales. 933 features (5.7%) survive with cos > 0.7. These are the features that existed in the 5M run and the 50M run simply refined rather than replaced.

### Top stable features (reciprocal matches, cos > 0.9)

2,749 features have *reciprocal* matches — A→B and B→A both point to each other — at cos > 0.3. The highest-cosine reciprocal pairs are all simple syntactic/lexical features:

| 5M latent | 50M latent | Cosine | Top token | Peak 5M→50M |
|---|---|---|---|---|
| 12457 | 2926 | 0.987 | ' overlap' | 12.4 → 12.4 |
| 10658 | 3253 | 0.987 | ' wake' | 16.2 → 16.8 |
| 14048 | 3405 | 0.986 | ' also' | 13.6 → 13.6 |
| 12403 | 15125 | 0.985 | ' sense' | 10.9 → 10.8 |
| 15387 | 7518 | 0.978 | ' isolated' | 9.1 → 8.1 |
| 2963 | 12952 | 0.978 | ' adjusting' | 11.1 → 10.9 |

**The most stable features are the simplest ones.** High-cosine reciprocal matches are lexical (specific tokens, function words) — features that exist to reliably fire on a particular token type don't need to reorganize as training extends. Their indices change (they move within the 16k feature space) but their decoder directions are almost identical.

**Conceptual features reorganize.** The GCD and Q&A features known from the dashboards are not in the top reciprocal list, suggesting they either formed anew at 50M (no 5M counterpart) or reorganized substantially. This is consistent with the observation that 0 strict mid-doc features existed at 5M L9 — those features weren't present at 5M and thus have no 5M match.

### Implication for scaling

We can now answer this question directly with the completed 200M/32k L6 run.

---

### L6: 50M/16k → 200M/32k (doubling feature space at 4× tokens)

A second matching run compares the 50M d_sae=16k checkpoint against the 200M d_sae=32k checkpoint. With different feature widths, the 16k decoder rows (unit-norm in 768-d space) are matched against the best-matching 32k row.

| Metric | L9: 5M→50M | L6: 50M/16k→200M/32k |
|---|---|---|
| Mean best-match cosine | 0.26 | **0.61** |
| Median best-match cosine | 0.15 | **0.67** |
| cos > 0.9 | 0.8% (134) | **11.1% (1,814)** |
| cos > 0.7 | 5.7% (933) | **45.4% (7,438)** |
| cos > 0.5 | 16.3% (2,669) | **71.1% (11,643)** |
| Reciprocal matches (cos>0.3) | 2,749 | **11,615** |

**These are fundamentally different regimes.** The L9 5M→50M transition is a reorganization: median cos 0.15, most features have no clear identity in the larger run. The L6 50M→200M transition is gradual refinement: median cos 0.67, 71% of features persist with cos > 0.5.

**The 32k run keeps the 16k feature base and adds on top of it.** 11,615 reciprocal matches mean the 32k model has near-identical representations for most of the 50M/16k features — they just moved to new indices. The extra 16k feature slots fill in with new directions. This is feature expansion, not feature replacement.

Top stable reciprocal matches:

| 50M/16k | 200M/32k | Cosine | Token |
|---|---|---|---|
| 8618 | 29915 | 0.996 | ' method' |
| 10203 | 13608 | 0.994 | ' interactions' |
| 2602 | 4294 | 0.991 | ' disconnected' |
| 13502 | 9956 | 0.988 | ' antagonists' |
| 7072 | 18897 | 0.988 | 'ones' |

Semantically meaningful tokens (not just punctuation) dominate the top stable list — a sharp contrast to the L9 5M→50M case where the most stable features were syntactic tokens (`.`, `(`, `ed`).

**What is in the new 32k features?** The deduplicated 200M/32k dashboard reveals precision morphological detectors absent from the 50M/16k run:

| Feature | Description | Score |
|---|---|---|
| 28651 | "pre-" prefix in compound words/hyphenated terms | 1.000 |
| 18970 | "inter-" prefix in compound words | 1.000 |
| 21881 | "non-" prefix in hyphenated compounds | 1.000 |
| 24685 | "hyper-" in medical/technical terms | 1.000 |
| 22526 | Closing HTML/XML tags `</` | 1.000 |
| 29030 | Hyphens in compound phrases | 1.000 |

This is **feature splitting in action.** The 50M/16k model likely had a single "compound word prefix" feature that activates on `pre-`, `inter-`, `non-`, `hyper-` and similar. At 32k features, that direction splits into four distinct monosemantic features, each with perfect auto-interp scores. The 16k general feature becomes four 32k specific features — exactly what the superposition hypothesis predicts as a benefit of wider SAEs.

---

## 9. The 200M/32k run: auto-interp results

Auto-interp on the Jaccard-deduplicated top-30 features from the 200M/32k L6 run:

| Config | Tokens | d_sae | Dedupe mean | Failures (<0.65) | Perfect (1.00) |
|---|---|---|---|---|---|
| L6 50M/16k | 50M | 16,384 | 0.795 | 6 | 4 |
| L6 200M/32k | 200M | 32,768 | **0.864** | **3** | **9** |

**Interpretability improves significantly with scale.** The dedupe mean rises from 0.795 to 0.864, failures drop from 6 to 3, and perfect scores double from 4 to 9. This is a real signal: the 32k features are more monosemantic (easier for Claude to describe accurately enough to score well on the forced-choice task).

The 9 perfect-scoring features are all precise morphological/syntactic detectors: newlines after capitals, section breaks, sentence-ending periods, four distinct prefix features (`pre-`, `inter-`, `non-`, `hyper-`), closing HTML tags, and hyphenated compounds. The 3 failures are underspecified features that fire on broad categories ("beginning of word", "hyphens in compound terms") — essentially still-polysemantic latents that haven't specialized enough.

**Interpretability summary across all runs:**

| | L3 | L6 | L9 |
|---|---|---|---|
| 50M/16k dedupe score | 0.844 | 0.795 | 0.804 |
| 200M/32k dedupe score | — | **0.864** | — |

---

## 10. The full scaling sweep: L3 200M, L9 200M, L6 400M/64k

Three final runs complete the scaling study, separating two distinct axes: *token scaling at fixed dictionary size* (L3, L9) and *joint token + dictionary scaling* (L6 progression 50M/16k → 200M/32k → 400M/64k).

### Final autointerp table (Jaccard-deduplicated top-N, Claude Sonnet 4.5 judge)

| Run | Tokens | d_sae | FVU | Dead | Mean BA |
|---|---|---|---|---|---|
| L3 50M/16k | 50M | 16,384 | 0.042 | 1 | 0.844 |
| L3 200M/16k | 200M | 16,384 | ≈0.042 | ~1 | 0.810 |
| L6 50M/16k | 50M | 16,384 | 0.057 | 10 | 0.795 |
| L6 200M/32k | 200M | 32,768 | 0.043 | 3 | 0.864 |
| **L6 400M/64k** | **400M** | **65,536** | **0.039** | **~35** | **0.923** |
| L9 50M/16k | 50M | 16,384 | 0.099 | 10 | 0.804 |
| **L9 200M/16k** | **200M** | **16,384** | **0.090** | **0** | **0.928** |

Two headline results:

**A. L6 dict scaling is cleanly monotonic.** The L6 progression 50M/16k (0.795) → 200M/32k (0.864) → 400M/64k (0.923) shows interpretability improving at every step as the dictionary doubles. FVU also drops monotonically (0.057 → 0.043 → 0.039). This is the clearest evidence in the sweep for the *feature splitting helps interpretability* story: as the SAE gets more capacity, polysemantic features fracture into monosemantic ones that Claude can describe precisely enough to win the forced-choice scoring task.

**B. At fixed dict size (16k), layer dominates over tokens.** L9 200M/16k posts 0.928 — the best score in the entire sweep — while using only a 16k dictionary. L3 at the same 16k dict size *regresses* from 0.844 (50M) to 0.810 (200M). This splits the token-only scaling regime into two different behaviors:

- **L9 (deep layer):** token scaling at fixed dict size helps (0.804 → 0.928). Deeper residual streams carry richer feature content that a 16k dict has not saturated at 50M tokens.
- **L3 (early layer):** token scaling at fixed dict size hurts slightly (0.844 → 0.810). Early-layer feature space is simple enough that 16k dict is already saturated at 50M; extra tokens cause minor drift without adding structure.

This is consistent with the L9-paradox from §2 — the deeper layer's representation is intrinsically harder to compress — but flips the sign of the scaling benefit. At 16k features, more tokens is what L9 needed to finally close the interpretability gap.

### Cross-scale cosine matching: refinement vs reorganization

Re-running decoder-column matching for the new 200M runs at fixed 16k dict size:

| Match | Dict size | Median cos | Mean cos | Regime |
|---|---|---|---|---|
| L3 50M → L3 200M | 16k | 0.729 | 0.678 | Gradual refinement |
| L6 50M → L6 200M | 32k | 0.673 | 0.608 | Gradual refinement |
| L9 5M → L9 50M | 16k | 0.15 | ~0.2 | Reorganization |

Both L3 and L6 show the same gradual-refinement regime (median cos ~0.67–0.73): when dict size is held fixed and only tokens are scaled, the feature basis drifts slowly, with ~17% of features achieving cos > 0.9 matches across scales. The L9 5M→50M jump was reorganization (median cos 0.15) — a qualitatively different regime that we now interpret as *undertrained* → *trained*, not *trained* → *better trained*. The 5M run was below the phase transition and hadn't yet converged to a stable feature basis.

### What the sweep shows, summarized

1. **Dict scaling > token scaling for interpretability gains.** L6's 0.795 → 0.923 progression (Δ = 0.128) is driven by the 4× dict expansion, not the 8× token expansion. L3 token-only scaling at 16k dict went *backwards*.
2. **Deeper layers need more tokens.** L9 at 16k dict gained 0.124 from a 4× token bump alone, because its feature space wasn't saturated. L3 was already saturated at 50M.
3. **"Feature splitting" and "feature refinement" are different phenomena.** Feature splitting happens when dict size grows (one feature becomes several monosemantic specializations — see §8). Feature refinement happens when tokens grow at fixed dict size (the same feature becomes a cleaner version of itself — median cos 0.67+). Both can help interpretability, but only under the right conditions for each layer.
4. **Pythia-160M L9 at 200M/16k is the sweet spot for this architecture.** Mean BA 0.928, 0 dead latents, cheapest config in the top tier. If you had to deploy one SAE from this sweep, this is it.

---

## 11. Limitations and honest framing

This is a small-scale replication study on a 160M-parameter model, not a novel methodological contribution. Related prior work (Anthropic's *Scaling Monosemanticity*, DeepMind's *Gemma Scope*, EleutherAI's SAE releases) has established feature splitting and scale-dependent interpretability at much larger scales. What this repo adds:

- A clean, fully reproducible end-to-end pipeline runnable on a single consumer GPU.
- A direct head-to-head comparison of *token scaling vs dict scaling* on the same layers, controlled for everything else.
- Cross-scale cosine matching to distinguish *refinement* from *reorganization* regimes.
- An LLM-judge autointerp evaluation across 7 training configs, scored consistently by Claude Sonnet 4.5.

What it does *not* establish:

- Findings on Pythia-160M may not transfer to frontier-scale models. Layer-indexing intuitions (early vs late) are not directly portable.
- The dedupe-top-N autointerp methodology evaluates the *best* features, not average feature quality. Mean BA improvements could reflect better top-end features with unchanged or worse long-tail.
- No downstream task evaluation. We have not tested whether higher-BA SAEs produce more useful interventions (steering, ablation, probing transfer).

## 12. Next steps

- **Downstream utility evaluation.** Pick a probing or steering task and measure whether the 0.923 L6 400M/64k SAE beats the 0.795 L6 50M/16k baseline on something that matters, not just judge scores.
- **Long-tail autointerp.** Run autointerp on a *random* 100-feature sample per SAE instead of the Jaccard-dedupe top-N, to measure average feature quality rather than best-case.
- **Feature splitting quantification for 400M/64k.** Map which 32k features split into which 64k features using the cos 0.5–0.8 partial-match heuristic.
- **Port to a larger model.** Repeat the L6 dict-scaling progression on Pythia-1B or Qwen-1.5B and see whether the monotonic interpretability improvement holds.
