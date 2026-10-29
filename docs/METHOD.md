# Method walkthrough

## Objective and scope

The task is pairwise ear verification: given two ear images, return a score reflecting whether they belong to the same identity. Track 1 emphasizes a compact model; Track 2 measures performance on a CPU-only Raspberry Pi 4. The description below separates the team's submitted method description from details observed in the local experimental source.

## Image-to-embedding pipeline

The team's method description specifies RGB image preprocessing with resize and center crop to 224 by 224 pixels and standard ImageNet normalization. A ConvNeXt backbone produces image features. A linear projection followed by batch normalization maps those features to a 256-dimensional embedding, which is L2-normalized.

For normalized embeddings `u` and `v`, cosine similarity is their dot product:

```text
score(u, v) = sum_i u_i * v_i
```

Larger scores indicate greater similarity. A usable decision threshold depends on the false acceptance rate required by the evaluation or application; a single default threshold is not claimed here.

## Identity-supervised training

The submitted description reports generic ImageNet-pretrained initialization, identity supervision from the official public training package, and an ArcFace-style angular-margin classification objective. It reports no additional ear datasets or external identity annotations.

In the preserved implementation, `EmbeddingBackbone` produces normalized features and `ArcMarginProduct` compares them with normalized, learned class weights. A margin is applied to the target class angle, and scaled logits are passed to cross-entropy. The identity classifier is a training component: verification uses the embedding network, not predictions over the training identities.

`train_rank1.py` implements AdamW, mixed precision when using CUDA, cosine learning-rate scheduling, gradient clipping, and a class-balanced weighted sampler. Its default settings are implementation defaults, not verified final-run hyperparameters. The source also implements random resized crops, horizontal flips, color jitter, occasional grayscale conversion and Gaussian blur, and random erasing.

## Track-specific choices

- **Track 1:** the team's submitted description identifies a ConvNeXt-Zepto-style backbone. The organizers reported 1,965,217 parameters and a model size of 7,860,872 bytes for the final entry. These are reported measurements, not values recalculated from a checkpoint in this repository.
- **Track 2:** the submitted description identifies a ConvNeXt-Tiny variant and horizontal-flip test-time augmentation. The original and flipped embeddings are averaged and normalized. This entails additional model evaluation; its benefit must be weighed against measured deployment cost.

No score-level ensemble of independently trained models was reported. The source explores additional backbones and feature post-processing modes, but their presence must not be read as evidence that they were used in the final submissions.

## Evaluation and selection

The local validator supports all pairs in `pairs_val.csv` when its pair limit is zero, embeds unique images, and computes pairwise similarity scores. It reports verification rates at 0.1% and 1% false acceptance rate, EER, and ROC AUC. Its ROC interpolation and discrete EER calculation must be cross-checked against the official toolkit before treating local numbers as protocol-equivalent.

The historical scripts' names and gate thresholds reflect an ambitious development target, not a rank-one result. The final official ranks were seventh in both tracks. No training or model selection should use hidden labels, and the repository does not contain sequestered features or labels.

## Interpretation

The final model was small, but verification accuracy remained below the competition reference baseline and leading teams. This is a useful limitation, not an ablation result: without controlled experiments, the accuracy gap cannot be attributed to any particular backbone, loss, augmentation, or training duration. Future changes should be evaluated on a fixed protocol with both recognition and deployment costs reported.

See [provenance](PROVENANCE.md) for the missing artifacts needed to establish exact reproducibility and the [historical source guide](../experiments/historical/README.md) for implementation caveats.
