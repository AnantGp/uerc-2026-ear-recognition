# Unconstrained Ear Recognition: UERC 2026

**A compact metric-learning approach to ear verification, developed by Visionary Victors for the UERC 2026 accuracy, model-size, and edge-inference challenges.**

**Team:** Gourav Gupta and Anant Gupta · Arogyapandit Private Limited, Jaipur, India

**Challenge:** [UERC 2026](http://uerc.fri.uni-lj.si/) · **Participation:** Track 1 and Track 2

[Method](docs/METHOD.md) · [Code guide](experiments/historical/README.md) · [Results](results/official_final_results.json) · [Provenance](docs/PROVENANCE.md)

> This is a research record, not a production biometric system. The preserved scripts are historical experiments; the exact final submission packages and trained weights are not included or verified here.

## The research problem

Ear recognition aims to verify whether two ear images belong to the same person. In unconstrained settings, pose, illumination, blur, occlusion, and differences in image capture make verification difficult. UERC 2026 adds a practical constraint: a system must also be small or fast enough for resource-limited devices.

The challenge evaluates two related settings:

- **Track 1: Lightweight model design.** Recognition accuracy is evaluated together with model parameter count and memory footprint.
- **Track 2: Edge deployment.** Recognition accuracy and inference time are measured on a CPU-only Raspberry Pi 4.

Our objective was to learn compact, discriminative image embeddings that could be compared efficiently with cosine similarity.

## Approach

Our team described the final submissions as **ConvNeXt ArcFace** models:

```text
Ear image -> resize/crop + normalization -> ConvNeXt backbone
          -> projection + batch normalization -> 256-D L2-normalized embedding
Two embeddings -> cosine similarity -> verification score
```

- **Backbones:** a compact ConvNeXt-Zepto-style model for Track 1 and a ConvNeXt-Tiny variant for Track 2, implemented with `timm`.
- **Training:** ImageNet-pretrained initialization, public competition training identities, an ArcFace-style additive angular-margin objective, AdamW, mixed precision, and cosine learning-rate scheduling.
- **Augmentation:** random resized crops, horizontal flips, mild color jitter, occasional grayscale and Gaussian blur, and random erasing.
- **Evaluation:** 224 × 224 center-cropped inputs with ImageNet normalization, followed by L2-normalized embeddings and cosine similarity. Track 2 used horizontal-flip test-time augmentation by averaging embeddings before normalization.
- **Model selection:** validation-pair verification rates, equal error rate, and ROC AUC. The team's description reports no additional ear datasets or external identity annotations, apart from generic pretrained weights.

These details summarize our submitted team description; the exact training run, checkpoint, and submission code have not yet been recovered and cross-checked against it. See [provenance and reproducibility](docs/PROVENANCE.md).

## Official final results

The following values are from the organizers' final-results email dated May 18, 2026. Higher verification rate and AUC are better; lower EER is better. Rankings below are by the competition's composite score, not verification accuracy alone.

| Metric | Track 1 | Track 2 |
| --- | ---: | ---: |
| Official rank | 7th of 10 teams | 7th of 8 teams |
| VER @ 0.1% false acceptance rate | 0.1607 | 0.1500 |
| VER @ 1% false acceptance rate | 0.3484 | 0.3053 |
| Equal error rate | 0.2224 | 0.2534 |
| ROC AUC | 0.8598 | 0.8286 |
| Composite score | 0.2803 | 0.1817 |

Track 1 reported **1,965,217 parameters** and a model size of **7,860,872 bytes**. Track 2 reported a **1.2357-second mean per-image inference time** and **0.0193-second standard deviation** on the organizers' edge benchmark.

For context, the Track 1 winner achieved VER@0.1% of 0.5079 with approximately 1.78 million parameters; the Track 2 winner achieved VER@0.1% of 0.5117. Our small Track 1 model was competitive in size, but substantially behind the winner in recognition accuracy. The organizers' results included a reference baseline with Track 1 VER@0.1% of 0.2623 and AUC of 0.9215 with approximately 87.6 million parameters. This is an accuracy-versus-size comparison, not evidence that our model outperformed the baseline overall.

**Metric note:** VER@0.1% is the proportion of genuine comparisons accepted at a 0.1% false acceptance rate. Results are reported on the organizers' evaluation protocol and should not be treated as performance on other datasets or populations.

## What this work shows

The submission brought Track 1 parameter count down to approximately 1.97 million, but did not retain the reference baseline's verification performance. Track 2 also remained behind the leading systems in both verification accuracy and reported inference speed. The result illustrates why backbone size alone is not enough: discriminative embeddings and the actual deployment path must be evaluated together.

The research contribution here is a documented competition implementation and evaluation, not a new backbone or a claim of state-of-the-art recognition. Promising follow-up work would test stronger representation learning and carefully measured inference optimizations against a fixed validation protocol.

## Repository contents

```text
.
├── README.md
├── docs/                  # Method walkthrough and evidence boundaries
├── experiments/
│   └── historical/        # Locally preserved sprint scripts; see caveats
├── results/               # Organizer-reported results, not rerun metrics
├── tests/                 # Syntax, template, and archive-integrity checks
└── REPOSITORY_METADATA.md # Suggested GitHub description and topics
```

The historical scripts cover candidate training, validation, orchestration, and submission packaging. They are preserved for inspection, not advertised as an exact reproduction of the final competition run. They depend on the competition toolkit and data layout, use original GCP paths, and have not been tested in this repository. Packaging and orchestration scripts can write output files; inspect them before running.

**Not included:** competition datasets and image pairs, sequestered features, trained weights, final submission ZIPs, environment lockfiles, or the organizers' draft paper. Dataset and model-asset redistribution must be checked against their applicable terms. No original license has yet been selected for this repository.

## Reproducibility status

Exact reproduction is **not currently established**. The final checkpoint, submission package, training logs, environment versions, and authoritative dataset access instructions are needed to verify the final architecture, hyperparameters, preprocessing, and measured results. The scripts are useful as historical context but should not be presented as a runnable reproduction of the reported scores.

Before running experiments, obtain the competition data and toolkit under their applicable terms, inspect each script and its path arguments, and use a controlled environment with compatible PyTorch, torchvision, `timm`, NumPy, Pillow, SciPy, scikit-learn, and tqdm. Avoid untrusted PyTorch checkpoints: deserializing them can execute code.

Repository checks need only Python 3.10 or newer and do not download data, load checkpoints, or import the training dependencies:

```bash
python3 -m unittest discover -s tests -v
```

Passing these checks establishes syntax and archive integrity only, not training correctness or reproduction of the official results.

## Responsible use

This work is intended for research on consented, appropriately licensed data. Ear images and identity embeddings are biometric information. No privacy, fairness, spoof-resistance, or operational security assessment is established by these competition results; deployment would require separate evaluation and governance.

## References

- Z. Liu et al., [A ConvNet for the 2020s](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html), CVPR 2022.
- J. Deng et al., [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html), CVPR 2019.
- R. Wightman et al., [PyTorch Image Models (`timm`)](https://github.com/huggingface/pytorch-image-models).
- [UERC 2026 challenge website](http://uerc.fri.uni-lj.si/).

The organizers' UERC 2026 paper draft describes this team's approach. A verified final publication link and DOI should be added once available; no publication status is asserted here.
