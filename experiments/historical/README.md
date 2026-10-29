# Historical experiment source

These six Python files were copied unchanged from the locally preserved competition sprint directory. Their SHA-256 digests are recorded in [source_checksums.json](source_checksums.json). The digests identify this local snapshot; they do not prove that it is the exact code sent to the organizers.

## Code map

- [`rank1_lib.py`](rank1_lib.py): candidate definitions, embedding model, angular-margin classifier, dataset readers, transforms, feature extraction, ROC metrics, post-processing, and packaging helpers.
- [`train_rank1.py`](train_rank1.py): identity-supervised training, validation, and checkpoint writing.
- [`validate_rank1.py`](validate_rank1.py): candidate or submission evaluation on public validation pairs.
- [`package_rank1.py`](package_rank1.py): submission-file templates, feature extraction, candidate packaging, and ZIP verification.
- [`orchestrate_rank1.py`](orchestrate_rank1.py): historical Track 1 candidate scheduling and gate-based packaging.
- [`orchestrate_t2.py`](orchestrate_t2.py): historical Track 2 candidate scheduling and flip-augmentation evaluation.

The `rank1` names describe the sprint's goal, not its outcome. MobileNetV4, EdgeNeXt, and additional ConvNeXt candidates in these files are experimental configurations, not separately verified submissions or published ablation results.

## Inspect before execution

This is an archive, not an installable training package. No pinned environment or matching trained checkpoint is available here. Imports include PyTorch, torchvision, `timm`, NumPy, Pillow, SciPy, scikit-learn, and tqdm. Generated submission code additionally expects the official `uerc26_dataset` and `uerc26_utils` modules and their working-directory conventions.

- Training, validation, and packaging defaults use the original GCP filesystem layout. Orchestrators contain hard-coded absolute paths. Use an isolated copy and review every output location before execution.
- `safe_replace_submission_dir` removes an existing candidate directory with a `vv_` prefix before recreating it. ZIP creation can replace an existing ZIP. Orchestration is therefore not a read-only smoke test.
- Checkpoint loading can deserialize Python objects. Load only trusted checkpoints. Some loads use `strict=False`, so missing and unexpected keys require explicit inspection rather than assuming success.
- The local ROC/EER calculation is not established as identical to the official protocol. Pair reading skips malformed rows, and feature-based evaluation can exclude missing pairs. Audit counts before interpreting metrics.
- Post-processing options are experiments. Feature extraction and generated image inference are separate code paths; their preprocessing, TTA, and any fitted centering or whitening must agree before a new submission is considered valid.
- Gate thresholds and `--force-gamble` are historical controls, not evidence of a passed validation gate or a guaranteed ranking.

The top-level [repository checks](../../tests/test_repository.py) parse these files and the generated Python templates without importing or executing the training code. They cannot validate model performance, checkpoint compatibility, or runtime deployment.
