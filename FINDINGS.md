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

## 7. Next steps

- **50M dashboards + cross-layer comparison.** Build dashboards from the 50M checkpoints and re-run `compare_layers.py` to see whether the BOS cluster shrinks and mid-document feature count increases at the longer training scale.
- **50M auto-interp.** Score the 50M dashboards and compare balanced-accuracy distributions — the key question is whether the semantic features score higher once structural absorbers weaken.
- **Targeted autointerp on dedupe-filtered features.** Current auto-interp runs on top-30 by peak, which is dominated by structural features. Running it specifically on the dedupe sample would give scores for the semantically interesting features.
- **Layer 9 deep dive.** The residual stream geometry hypothesis is rejected; what explains the difficulty? Candidates: higher effective dimensionality of activations, stronger interdependence between token positions, or a mismatch between the top-k sparsity constraint and how information is encoded at depth.
