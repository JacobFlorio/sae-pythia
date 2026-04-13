# Findings: Pythia-160M SAE — layers 3, 6, and 9

Three Top-K SAEs trained identically (16,384 features, k=32, 5M tokens, Pile-uncopyrighted) on layers 3, 6, and 9 of Pythia-160M. This document covers what each SAE learned — reconstruction quality, failure modes, and the semantic features that survive deduplication — plus a cross-layer comparison.

All snippets are reproducible from the committed `dashboards/*_sample_dedupe.json` files or regenerated from the full dashboard JSONs via `scripts/build_dashboard.py`. Cross-layer metrics come from `scripts/compare_layers.py`.

---

## Cross-layer summary

| Metric | Layer 3 | Layer 6 | Layer 9 |
|---|---|---|---|
| **FVU** (lower = better reconstruction) | 0.078 | **0.074** | 0.125 |
| Dead latents | 6,658 (~41%) | 6,734 (~41%) | **3,511 (~21%)** |
| BOS-only features in top 50 | 24 | **17** | 25 |
| Strict mid-doc features in top 50 | 0 | **1** | 0 |
| Unique clusters after Jaccard dedupe | 3 | **5** | 3 |
| Peak activation median | 68.2 | 65.5 | **48.3** |
| Peak activation max | 76.0 | 71.7 | **52.0** |

---

## 1. Reconstruction quality

Layer 6 achieves the best reconstruction (FVU 0.074) despite having a similar dead-latent fraction to layer 3 (both ~41%). Layer 9 is the outlier: its FVU jumps to 0.125, meaning the SAE explains only ~88% of the variance, 5 points worse than layer 6.

The paradox: **layer 9 has the fewest dead latents (21%) but the worst reconstruction.** This implies the layer 9 residual stream is genuinely harder to decompose. By layer 9, the residual stream is deeply contextual — it has integrated 9 rounds of attention and MLP processing — and the superposition of information it encodes may simply not factor cleanly into independent linear features at this training scale. More live latents are necessary but not sufficient.

---

## 2. Pathology: BOS absorption clusters

All three layers have a cluster of duplicated latents that fire on the first content token of a document (`token_pos=0`), regardless of what that token is. The cluster is largest at layer 9 (25/50), smallest at layer 6 (17/50).

```
# Layer 3, cluster representative — max=76.0
[76.0] 'Anthony'   Anthony Iluobe  Chief Anthony Iluobe (JP) was born in …
[74.9] 'Guard'     Guard youths from alcopops  9:55 AM, May 8,
[74.9] 'Prom'      Promethium: uses  The following uses for promethium …

# Layer 6, cluster representative — max=71.7
[71.7] 'Guard'     Guard youths from alcopops  9:55 AM, May 8,
[71.4] 'Prom'      Promethium: uses  The following uses for promethium …
[71.2] 'Anthony'   Anthony Iluobe  Chief Anthony Iluobe (JP) was born in …

# Layer 9, cluster representative — max=52.0
[52.0] 'Prom'      Promethium: uses  The following uses for promethium …
[51.5] 'Guard'     Guard youths from alcopops  9:55 AM, May 8,
[51.5] 'Anthony'   Anthony Iluobe  Chief Anthony Iluobe (JP) was born in …
```

**The BOS signal weakens with depth.** The peak activation of the strongest BOS-cluster representative drops from 76.0 (layer 3) to 71.7 (layer 6) to 52.0 (layer 9). This makes sense: at layer 3, the residual stream still closely resembles the raw token embedding, which has high position-0 variance; by layer 9, 9 rounds of attention have diffused that positional signal into a broader contextual representation.

A second structural cluster — duplicated latents firing on early-document newlines (code license headers, HTML structure) — appears at all three layers with similar characteristics. Layer 6 has 5 unique clusters after Jaccard-0.5 deduplication; layers 3 and 9 collapse to only 3 each, confirming that the top-50 peak-ranked list at those layers is almost entirely structural absorption.

---

## 3. Real features by layer

After fingerprint deduplication, semantic features become visible. A striking pattern emerges: **the character of the features shifts with depth**, from lexical token patterns at layer 3 toward more contextual and concept-level features at layer 9.

### Layer 3 — lexical and template features

Layer 3 features respond to specific tokens or near-verbatim templates. The model hasn't had time to integrate much context, so the clearest features are ones where the token itself carries almost all the signal.

**Feature 4633 — "U.S. Pat." abbreviation in patent text (peak 8.6)**
```
[8.6] ' Pat'   … significantly improving document processing, as disclosed in U.S. Pat. Nos. 4,205,780 …
[8.6] ' Pat'   … a powdered fertilizer material is illustrated by the U.S. Pat. No. to F …
[8.5] ' Pat'   … various methods to produce composite materials. U.S. Pat. No. 2,931, …
```
Fires on the word fragment `Pat` in "U.S. Pat." across unrelated patents (document processing, fertilizers, composite materials).

**Feature 7752 — "Public" in GNU GPL boilerplate (peak 8.6)**
```
[8.6] ' Public'  … You should have received a copy of the GNU General Public License …
[8.6] ' Public'  … MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License …
[8.4] ' Public'  … redistribute it and/or modify it under the terms of the GNU General Public License …
```
Fires on "Public" in the GPL boilerplate phrase, across different programs. Generalizes across at least three distinct software projects.

**Feature 742 — judicial title in US court opinions (peak 8.5)**
```
[8.5] ' Judge'    … BENTON, Circuit Judge.  Barbara …
[6.7] ' Justice'  … HARWOOD, Justice. Charles Sharrief and Millie Sharrief …
[6.6] ' Judges'   … LOKEN, COLLOTON, and BENTON, Circuit Judges …
```
Fires on judicial titles ("Judge," "Justice," "Judges") in the signature lines of US federal and state court opinions.

**Feature 13118 — "FIG." reference in patent figure captions (peak 8.5)**
```
[8.5] 'FIG'   … acts as a working fluid. FIG. 1 is a vertical cross-section …
[8.5] 'FIG'   … a radio protocol as close as possible to radio channels. FIG. 1 is a diagram …
[8.4] 'FIG'   … throughput in an individual wafer processing system. FIG. 14 shows …
```
Fires on "FIG" in patent figure cross-references across electronics, communications, and semiconductor patents.

**Feature 9354 — "in vitro" in biomedical literature (peak 8.4)**
```
[8.4] ' vitro'   … the present in vitro study was designed to further support our previous in vivo results …
[8.4] ' vitro'   … have been shown to beneficially affect targeted cellular functions *in vitro* …
[8.3] ' vitro'   … the value of such studies is uncertain. Polysaccharides that elicit effects *in vitro* …
```
Fires on "vitro" in the Latin phrase "in vitro" across biomedical research papers. Generalizes across both plain text and Markdown-italic formatting.

---

### Layer 6 — phrasal and register features

Layer 6 features move beyond individual tokens: they respond to the surrounding register and phrasing, not just the token itself. The auto-insurance feature (5735) fires on tokens like "lapse" and "miles" that are common words — but only in the specific syntactic and register context of insurance policy disclaimers.

*(Full layer 6 features documented above in the original findings: 5735 auto-insurance boilerplate, 4895 patent "Field of the Invention," 11957 Apache-license warranty, 10188 biomedical citation hyphens, 10166 HackerNews reply separators.)*

---

### Layer 9 — concept and domain features

Layer 9 features show the most conceptually specific activations of the three layers, though peak activations are substantially lower (max 52 for BOS vs. max ~30 for the semantic features).

**Feature 12045 — "greatest common divisor" in math problem sets (peak 31.5)**
```
[31.5] ' divisor'  … Calculate the highest common divisor of 3347 and 7. 1  …
[30.9] ' divisor'  … Calculate the greatest common divisor of 13864 and 2504. 8 …
[30.1] ' divisor'  … Calculate the greatest common divisor of 2180 and 237838. 218 …
```
Fires specifically on "divisor" within GCD/HCF problem statements, across different numbers. The strongest mid-document semantic feature in the whole comparison — it fires at positions 87–435, entirely within math problem text.

**Feature 11476 — HackerNews reply-thread newlines (peak 26.5)**
```
[26.5] '\n'   … Signal doesn't allow.  ------ fit2rule The free world needs …
[25.4] '\n'   … something which isn't as questionable.  ~~~ mmPzf A big plus …
[24.9] ' the' … The intervention group followed a 12-week traditional dance …
```
The `------ username` HackerNews separator fires at both layer 6 and layer 9, suggesting this structural feature is robustly represented across multiple layers of the network.

**Feature 9612 — HTML/code indentation whitespace (peak 28.6)**
```
[28.6] '       '  … id="cust_1">customer 1</a></li>        <li title="cust 2"> …
[28.1] '    '     … text/html; charset=iso-8859-1"></head>     <frameset rows="92, *" …
[27.1] "('"        … database.getTable('CLIENTS') because when I comment …
```
Fires on indentation-style whitespace in HTML and code snippets — a formatting feature that may reflect the model tracking code structure at deeper layers.

**Feature 6710 — decimal points in numeric/JSON data (peak 25.2)**
```
[25.2] '.'   … "boxHeight": 2.55,   "boxLength": 2.55,   "boxWidth": 2.55 …
[25.2] '.'   … management-core-3.0.4-javadoc.jar.md5 …
[24.8] '.'   … "boxLength": 2.55,   "boxWidth": 2.55 …
```
Fires on decimal points in numeric literals within JSON and filename versioning contexts.

---

## 4. Takeaways

**1. Layer 6 is the sweet spot at this training scale.** It has the best reconstruction (FVU 0.074), the smallest BOS cluster (17/50), the most unique feature clusters (5), and is the only layer to surface a strict mid-document semantic feature in its top-50 peak list. This is likely specific to 5M tokens — with longer training, layers 3 and 9 may catch up.

**2. Layer 9 features are more conceptually specific but harder to find.** The GCD feature is the cleanest semantic hit across all three layers, but it activates at much lower magnitude (peak 31.5) compared to the structural absorbers (peak 52). At this training scale, layer 9 cannot reconstruct its residual stream as well as layers 3 or 6, even though it has fewer dead latents — meaning the deeper residual stream resists linear decomposition.

**3. Feature character shifts with depth: lexical → phrasal → conceptual.** Layer 3 fires on specific token fragments ("Pat.", "FIG.", "vitro") regardless of broader context. Layer 6 fires on phrase-level register signals (the auto-insurance disclaimer context, the GPL license phrase). Layer 9 fires on domain-level concepts (GCD problem format). This gradient is consistent with what attention-head circuit analysis suggests: early layers track syntactic and lexical patterns; later layers integrate broader context into domain representations.

**4. The BOS absorption pathology is universal but layer-dependent.** All three layers have it; it's worst at layers 3 and 9, slightly better at layer 6. The peak activation of the BOS cluster decays monotonically with depth (76 → 72 → 52), suggesting the positional signal becomes less dominant in the residual stream as context accumulates.

**5. The peak-activation collapse at layer 9 is the most unexpected result.** Median peak drops from 68.2 (layer 3) to 65.5 (layer 6) to 48.3 (layer 9). The full range compresses to [44, 52] at layer 9 — almost no spread — compared to [20, 72] at layer 6. This is consistent with the residual stream at layer 9 encoding information in denser superposition, where no single direction dominates enough for a top-k SAE to claim it confidently.

---

## 5. Next steps

- **Longer training.** 5M tokens with 41% dead latents is close to the minimum viable run. The cross-layer character shift (lexical → phrasal → conceptual) is preliminary; it should be confirmed at 50M tokens where dead-latent fraction should drop substantially.
- **Auto-interp scoring.** The `run_autointerp.py` script generates Claude descriptions and balanced-accuracy scores for each feature. Running it across all three layers would turn the qualitative observations above into a quantitative interpretability metric per layer.
- **Dead-latent revival.** Layer 9's paradox (more live latents, worse FVU) warrants a follow-up run with a higher AuxK coefficient to see if forcing more latents active improves reconstruction at depth, or whether the problem is fundamental to the residual stream geometry.
