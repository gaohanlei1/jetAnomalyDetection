# Momentum-only JetCLR on the existing training pipeline

Select `--model jetclr` in `scripts/run_train_lejepa_part.py`. Keep the existing
dataset arguments and training loop. The default model remains `semi-sup-triplet`.
JetClass and CMS both supply the same raw momentum fields needed by this model.

## Inputs and representations

The model accepts the existing full feature tensor and padding mask. It extracts
only `part_px`, `part_py`, `part_pz`, using the configured feature names, and
reconstructs absolute `(eta, phi, pt)` in float32. Energy, fractions, relative
angles, charge, PID and track features in the input tensor are ignored.

The node embedding has three inputs, with no dataset z-score normalization and
no pt-fraction conversion. When pairwise attention bias is enabled, it uses
massless four-vectors reconstructed from the same coordinates as the embedding.
The dataset's energy channel is deliberately not used under this assumption.

`forward` returns CLS; `representation_head` produces FCN1, also exposed by
`forward_representation`. A separate FCN2 head supplies the contrastive objective.
Full-view inference does not augment data. Returned `z_views` are FCN1 features
for compatibility with the existing evaluation code; `projection_views` are FCN2.

## Views and augmentation

`num_global_views + num_local_views` total views are generated. View zero is the
original sample; only the remaining V-1 views are independently augmented. The
original is zero-padded to the daughter-slot capacity, with those slots masked.
Global/local group labels are retained; augmented views in both groups use the
same distribution. No crop or pt-drop is applied.
At least two views are required. `num_negative_views` and all corrupted-negative
settings are ignored. Negatives for contrastive learning are other samples in
the current global batch, not artificially corrupted views.

Rotation is disabled. Each augmented view applies distortion, collinear splitting, then
common eta and phi shifts per sample. Phi is wrapped to `[-pi, pi)`.
Padding is excluded. Daughter slots are appended rather than truncated; each
mother is split at most once. All daughter directions equal the mother's after
distortion, and daughter transverse momenta sum to the mother's pt.

These are **implementation defaults**, not undocumented paper constants:

| Option | Default | Meaning |
| --- | --- | --- |
| `--jetclr-distortion-scale` | 0.1 | GeV; angular Gaussian sigma = scale / max(pt, pt_floor) |
| `--jetclr-pt-floor` | 0.1 | GeV; floor used only in the Gaussian width |
| `--jetclr-split-probability` | 0.1 | Per-particle eligibility probability |
| `--jetclr-max-splits` | 4 | Maximum mothers split per sample and view |
| `--jetclr-split-min-fraction` | 0.1 | z sampled uniformly between this value and 1 minus it |
| `--jetclr-eta-shift-max` | 0.5 | Common shift uniform in [-max, max] |
| `--jetclr-phi-shift-max` | pi | Common shift uniform in [-max, max] |
| `--jetclr-temperature` | 0.1 | Cosine-similarity temperature |

The existing `--no-pairwise-bias` switch still works. Augmentation parameters,
effective feature definitions, disabled rotation, and zero corrupted views are
recorded in `summary.json`. Existing dataset-schema metadata describes the
incoming tensor, not the three actual embedding inputs.

## Loss and compatibility

The loss follows the paper's printed Eq. (1), with only other-sample negatives
in the denominator. For more than two views, average over all ordered
same-sample positive view pairs, excluding every same-sample view from the
denominator. This explicitly extends the two-view equation. A negative loss is
possible and is not itself a failure. Labels and LeJEPA/triplet/classification
weights are unused.

DDP gathers embeddings with autograd support. Per-rank batch sizes must match,
and the global batch must contain at least two samples. Logits and logsumexp
use float32 even under AMP.

`total_loss` is the contrastive objective. Legacy loop metric fields for invariant,
SIGReg, triplet and classification objectives are returned as zero placeholders;
they do not describe terms of this loss. Training/validation batch loops and
the optimizer are unchanged. Model selection, checkpoint metadata/loading and
the downstream model factory recognize `jetclr`.

This is a single-jet ParT adaptation with rotation removed, not a reproduction
of all LHCO dijet experiments. A two-prong jet is still one jet; it is distinct
from an event containing two separately clustered jets.

Run tests from the repository root with:

```sh
python -m unittest discover -s tests -p test_jetclr.py -v
```

Tests include two local Gloo workers for checking the distributed loss/gradient.
