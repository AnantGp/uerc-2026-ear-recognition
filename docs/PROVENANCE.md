# Provenance and reproducibility

## Evidence used for this record

1. The team's local UERC workshop-description LaTeX archive supplies the method summary and team names. It describes both tracks, pretrained initialization, augmentation, the embedding head, loss, optimizer, and Track 2 flip augmentation.
2. The organizers' May 18, 2026 final-results email supplies the precise metrics, ranks, parameter count, reported model size, and Track 2 inference timings. Track 1 ranks exclude the unranked baseline.
3. The locally preserved `uerc_rank1_sprint` directory supplies the Python scripts copied unchanged into `experiments/historical/`.

These sources are distinct. The local scripts were not verified against the submitted ZIPs, exact checkpoint, or training logs. In particular, the scripts explore multiple candidate architectures; their presence is not evidence that every candidate was submitted or produced the reported results. The organizers' paper draft was consulted as corroboration, not distributed here.

## Before claiming reproducibility

- Recover the final Track 1 and Track 2 ZIPs, trained weights, training logs, configuration, dependency versions, and checksums from the original environment.
- Confirm the final submission architecture and inference preprocessing against both the submitted method description and the packaging code.
- Verify the model evaluation protocol, identity splits, data access, and timing methodology using the official toolkit.
- Confirm redistribution permissions for all code, weights, data, embeddings, and third-party material before uploading any of them.
- Run independent, documented setup and smoke tests before presenting commands as reproducible.

## Security and data handling

Biometric images, identity labels, sequestered embeddings, checkpoints, and credentials are excluded from this record. Embeddings may themselves be sensitive biometric information. Never commit secrets or private evaluation data. The `.gitignore` is a safeguard, not a substitute for reviewing the actual staged files.
