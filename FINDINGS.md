# Findings from the 5M-token smoke run

After the first end-to-end training run (Pythia-160M, layer 6, 16,384 features, top-k=32, 5M tokens), this doc walks through what the SAE actually learned — both the failure modes and the real positives. Reconstruction is already good (FVU 0.074, ~93% variance explained), but *which* latents carry that reconstruction turns out to be the interesting story.

All snippets below are reproducible from `dashboards/features_sample.json` (naive peak ranking) and `dashboards/features_sample_dedupe.json` (fingerprint-deduplicated). The full 16k-feature dashboard is gitignored; regenerate it with `scripts/build_dashboard.py`.

## 1. Pathology: the BOS super-cluster

Ranking by raw peak activation, roughly **30 of the top 50 latents are duplicates** firing at `token_pos=0` on whatever content token happens to start a document. Their top-8 examples are *the same eight documents in the same order* with near-identical activations:

```
feature 14690, max=71.7  Guard / Prom / Anthony / def / Ste / Sen / Work / Work
feature 2456,  max=69.8  Guard / Prom / Anthony / def / Ste / Sen / Work / Work
feature 6572,  max=69.5  Guard / Anthony / Prom / def / Ste / Sen / Work / Work
feature 4435,  max=68.9  Guard / Anthony / Prom / Ste / def / Sen / Work / Work
…
```

This is a known SAE failure mode: the residual stream at position 0 has high, predictable variance (nothing has been attended to yet, so the representation is close to the token embedding), and the top-k loss rewards whichever latents can soak up that easy variance. With 41% of latents dead, the live ones pile onto the easiest signal.

**Methodological takeaway:** if you inspect an SAE by peak activation, the top of your list is mostly an absorption artifact, not an interpretable feature. Everything downstream should either filter position 0 or deduplicate by doc-fingerprint.

## 2. Pathology: the paragraph-break cluster

A second, smaller duplication cluster fires on **newlines inside the first ~50 tokens** — specifically on the blank-line separators in code license headers (`/** … */`), HTML document structure (`<html>\n<body>`), and headings-then-body prose:

```
feature 10856, max=68.2
  pos=1   '\n'   /**  * ScriptDev2 is an extension for mangos providing …
  pos=1   '\n'   /*  * Licensed to the Apache Software Foundation …
  pos=1   '\n'   /*  * Copyright 2010-2013 Amazon.com, Inc. …
```

Features 10856, 2997, 1430, 5749, 5942, 8176, 10849, 14989 all share this fingerprint. Same story as the BOS cluster: duplicated latents chasing the same structural position.

## 3. Real features (dedupe-filtered)

Once you drop any latent whose top-doc set has Jaccard overlap > 0.5 with a previously-seen cluster (see `scripts/sample_dashboard.py --rank-by dedupe`), actual semantic concepts become visible. Below are the clearest hits; all activations are raw, unscaled top-k outputs.

### Feature 5735 — auto-insurance policy boilerplate (peak 20.2)

```
[20.2] displayed  … Rates *displayed* are estimates and are not guaranteed …
[19.1] lapse      … with no *lapse* in coverage. Vehicle is assumed to be garaged …
[18.9] miles      … driven 15,000 *miles* annually. These rates also include …
[17.8] primarily  … used *primarily* for commuting and is driven 15,000 miles …
```

A crisp, semantic feature: fires mid-document on the canonical phrases in auto-insurance rate-quote disclaimers. All examples are from mid-document token positions (380–510), not structural.

### Feature 4895 — patent application "Field of the Invention" (peak 9.7)

```
[9.7] Invention  1. Field of the *Invention* The present invention relates to a motor drive …
[9.7] Invention  1. Field of the *Invention* The invention relates generally to a device …
[9.7] Invention  1. Field of the *Invention* The present invention relates to … optical coherence tomograph…
```

Patent documents have a near-verbatim opening section, and this latent fires on the second "Invention" token of that opening, across unrelated patents (motor drives, telephony devices, optical tomography).

### Feature 11957 — permissive-license warranty disclaimer (peak 9.3)

```
[9.3] KIND   on an "AS IS" BASIS, * WITHOUT WARRANTIES OR CONDITIONS OF ANY *KIND* …
[9.3] KIND   on an "AS IS" BASIS, // WITHOUT WARRANTIES OR CONDITIONS OF ANY *KIND* …
[9.1] KIND    * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY *KIND* …
```

Fires on the `KIND` token inside the Apache 2.0 / BSD-style disclaimer clause, across `/* */`, `// //`, and ` *  *` comment syntaxes — so the feature generalizes across comment styles, not just one.

### Feature 10188 — academic citation markup hyphens (peak 17.0)

```
[17.0] -  …[Cunningham *et al*, 1998](#bib6){ref*-*type="other"}; [Rougier *et al*, 1998]…
[16.9] -  …[Van Cutsem *et al*, 1999](#bib17){ref*-*type="other"}). Two European phase III trials…
[15.2] -  …[Rougier *et al*, 1998](#bib14){ref*-*type="other"}). The main adverse events…
```

Fires on the hyphen inside `ref-type="other"` — a markup convention for bibliographic references in biomedical literature. Narrow but completely consistent.

### Feature 10166 — HackerNews reply-thread separators (peak 9.2)

```
[9.2] \n  …commercial gain. ------ Jerry2 It's unfortunate that mods …
[9.1] \n  …number, something that e.g. Signal doesn't allow. ------ fit2rule The free world …
[9.1] \n  …bills they are told to pass. ------ canada_dry A perfect fit really. …
```

Fires on newlines that precede the `------ username` reply separator specific to HackerNews comment dumps in the Pile.

## 4. Takeaways

1. **Reconstruction quality is not the same as interpretability quality.** FVU 0.074 looks great, but the reconstruction is being carried disproportionately by duplicated positional/structural features. The semantically meaningful latents have much lower peak activations (9–20) than the positional ones (56–71).

2. **Ranking features by peak activation is actively misleading.** At this training scale the top of the list is almost entirely position-0 absorption artifacts. Fingerprint-based dedupe is a cheap, effective fix that's worth adding to any SAE-inspection workflow.

3. **The live-latent budget is the bottleneck.** With 6,734/16,384 latents dead, the ~9,650 live ones have to cover both structural absorption and semantics, so they duplicate rather than specialize. The obvious lever is longer training (and/or higher AuxK coefficient) to revive dead latents — this is the next experiment.

## 5. Limitations

- **Single 5M-token run.** Numbers are from one training run on one layer; no error bars, no seeds.
- **No auto-interp scores yet.** The descriptions above are my own reads of the top examples. The `run_autointerp.py` script will produce Claude-generated descriptions + balanced-accuracy scores — I'll add those numbers once I've done a run with an API key set.
- **No cross-layer comparison.** The whole point of the roadmap's layers 3/6/9 writeup is to see whether this duplication pathology is layer-specific. Not done yet.
- **Dashboard covers only 500 documents.** A feature that doesn't fire in the first 500 Pile docs will be invisible here. Scaling `--num-docs` is cheap.
