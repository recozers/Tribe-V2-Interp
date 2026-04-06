# TRIBE v2 Feature Visualization

Feature visualization for Meta's [TRIBE v2](https://github.com/facebookresearch/tribev2) brain encoding model. Finds the video input that maximally activates a target cortical region (default: V1, primary visual cortex) by gradient-based optimization through the full differentiable pipeline.

**Reference:** Olah et al., ["Feature Visualization"](https://distill.pub/2017/feature-visualization/), Distill 2017 — same technique, applied to a brain encoding model instead of a classifier.

## Pipeline

```
Fourier coefficients (optimized)
    → IRFFT (reconstruct spatial signal)
    → color decorrelation (natural RGB)
    → sigmoid → frames in [0, 1]
    → random jitter (translation augmentation)
    → ImageNet normalize
    → V-JEPA 2 ViT-G (frozen, fp16)
    → extract hidden states at layers [20, 30, 39]
    → spatial pool (8192 tokens → 32 temporal features)
    → TRIBE v2 FmriEncoderModel (frozen, fp16, average-subject mode)
    → cortical predictions (20,484 vertices)
    → index target ROI vertices (Glasser parcellation)
    → loss = −mean(ROI activation) + λ_fft · spectral_penalty + λ_temp · temporal_smoothness
    → backprop to Fourier coefficients
```

## Usage

```bash
# Default: V1, with lambda sweep + 5 restarts
modal run feature_viz.py

# Target a different region
modal run feature_viz.py --target-roi MT
modal run feature_viz.py --target-roi FFA

# Skip the lambda sweep (use default λ_fft=1e-4)
modal run feature_viz.py --skip-sweep

# Custom steps/restarts
modal run feature_viz.py --full-steps 3000 --n-restarts 3
```

**Requires:** A Modal account with an `HF_TOKEN` secret (named `huggingface-secret`) and access to the `facebook/vjepa2-vitg-fpc64-256` and `facebook/tribev2` HuggingFace repos.

**Hardware:** A100 80GB on Modal. Runs for ~2-4 hours depending on step count.

## Output

Results are saved to the Modal volume `tribe-v2-weights` under `outputs/{roi}/`:

```
outputs/V1/
├── sweep/                      # λ_fft sweep (5 values × 500 steps each)
│   ├── lambda_1e-04/
│   │   ├── frames_0000.png     # frame grids every 200 steps
│   │   ├── frames_final.png
│   │   └── loss_curve.png
│   ├── lambda_1e-03/
│   │   └── ...
│   └── comparison.png          # bar chart comparing sweep results
├── full/                       # full runs (5 restarts × 2000 steps)
│   ├── restart_0/
│   │   ├── frames_0000.png     # progressive: 64→128→256 visible in grids
│   │   ├── frames_final.png
│   │   └── loss_curve.png
│   └── ...
├── best_frames.png             # 8-frame grid from best restart
├── best_individual/            # all 64 frames as separate PNGs
│   ├── frame_00.png
│   └── ...
├── selectivity.png             # bar chart: all ROIs, optimized vs random
└── validation.json             # machine-readable selectivity metrics
```

## Design decisions

### Fourier space parameterization

Optimizing raw pixels produces high-frequency noise that exploits the model's patch embedding but isn't interpretable. Following Olah et al., we parameterize the input in Fourier space: the optimizer updates complex Fourier coefficients, and frames are reconstructed via inverse FFT. This naturally biases toward smoother images since each coefficient controls a global spatial frequency rather than a single pixel.

A quadratic spectral penalty (`freq_weight ∝ radius²`) further discourages high-frequency energy, acting as a differentiable low-pass filter.

### Progressive multi-scale (64 → 128 → 256)

Optimizing at full 256×256 resolution from the start gives the optimizer ~12M free parameters and a rough loss landscape. We instead start at 64×64 (only low-frequency coefficients), find coarse structure (blobs, dominant orientation), then upsample to 128×128 and 256×256 by zero-padding the Fourier spectrum. New high-frequency coefficients are initialized to zero so the image stays smooth right after each upscale.

Step budget split: 20% at 64×64, 30% at 128×128, 50% at 256×256.

V-JEPA 2 always receives 256×256 input — the zero-padded IRFFT produces a full-resolution image that's band-limited to the current stage's frequency range.

### Color decorrelation

Natural images have highly correlated RGB channels. Without decorrelation, the optimizer produces rainbow noise because it can independently push R, G, B in unrelated directions. We apply the ImageNet color correlation matrix (SVD square root of pixel covariance, from Lucid) after the IFFT, so the optimizer works in a decorrelated 3-channel space that naturally produces realistic color structure.

### Cosine-annealed learning rate

Each progressive stage uses a cosine schedule: higher LR at the start for exploration, decaying to a lower LR for refinement. Stage 1 (64×64) uses 0.03→0.003; later stages use 0.01→0.001. This replaces a fixed LR which either converges too slowly (low) or bounces around local optima (high).

### Temporal smoothness penalty

A gentle L2 penalty on frame-to-frame pixel differences (`λ_temporal = 0.1 × λ_fft`). V1 responds to motion and temporal flicker, so we don't want to suppress all temporal variation — just prevent frame-to-frame noise that exploits V-JEPA 2's tubelet boundaries (pairs of frames are grouped by the 3D patch embedding). The weight is deliberately small.

### Modality dropout (zero text/audio)

TRIBE v2 was trained with modality dropout (p=0.3), randomly zeroing out entire modalities during training. We exploit this by providing only video features — the model's `aggregate_features` auto-fills zeros for missing text and audio modalities. This is equivalent to a training condition the model has seen, so predictions remain meaningful.

### Lambda sweep selects by activation, not loss

The sweep evaluates multiple λ_fft values and picks the one that produces the highest raw ROI activation — not the lowest total loss. This is important because the total loss includes the spectral penalty term, which scales with λ. Selecting by loss would be biased toward low λ (less penalty = lower loss, regardless of activation quality).

### Validation: selectivity check

After optimization, we measure predicted activation across all ROIs in the Glasser parcellation (V1, V2, V3, V4, MT, FFA, PPA) for both the optimized stimulus and random noise baselines (5 seeds). Key metrics:

- **Lift over random:** target ROI activation (optimized) minus (random). Should be positive.
- **Selectivity ratio:** target activation / mean of all other ROIs. >1.0 means the stimulus is more activating for the target than for other regions. >1.5 is a meaningful result.

### Supported ROIs

| CLI name | HCP MMP1.0 label(s) | Region |
|----------|---------------------|--------|
| `V1`     | V1                  | Primary visual cortex |
| `V2`     | V2                  | Secondary visual cortex |
| `V3`     | V3                  | Third visual area |
| `V4`     | V4                  | Fourth visual area |
| `MT`     | MT                  | Middle temporal (motion) |
| `FFA`    | FFC                 | Fusiform face complex |
| `PPA`    | PHA1, PHA2, PHA3    | Parahippocampal place areas |

Any raw HCP label can also be passed directly (e.g., `--target-roi MST`).

## Model details

- **V-JEPA 2 ViT-G:** 40-layer vision transformer, 1408 hidden dim, ~1B params. Input: 64 frames × 3 channels × 256×256. Tubelet size 2 → 32 temporal tokens, patch size 16 → 256 spatial tokens per temporal position (8192 total). We extract hidden states at layers 20, 30, 39 (50%/75%/100% depth), matching TRIBE v2's `layers_to_use = [0.5, 0.75, 1.0]`.

- **TRIBE v2:** Transformer-based brain encoder mapping multimodal features → cortical surface predictions (fsaverage5, 20,484 vertices). Loaded in average-subject mode for unseen-subject generalization. Video features expected as `(B, n_layers, hidden_dim, T)`.

- **Token ordering:** V-JEPA 2's Conv3d patch embedding flattens tokens in (T, H, W) order. We reshape `(B, 8192, D)` → `(B, 32, 256, D)` and spatial-average-pool to `(B, 32, D)`.

- **Hidden states indexing:** V-JEPA 2 uses HuggingFace's `OutputRecorder` — `hidden_states` is 0-indexed with length `num_hidden_layers` (no embedding entry). `hidden_states[39]` is the last layer output.

- **Gradient checkpointing + hidden states:** We use `output_hidden_states=True` rather than forward hooks for extracting layer features. Forward hooks interact unreliably with gradient checkpointing (hooks fire during both the original forward and the recompute-during-backward, potentially capturing detached tensors). `output_hidden_states` is explicitly supported by HuggingFace's checkpointing implementation. The extra ~1.6 GB for storing 40 vs 3 hidden states is acceptable given our 18 GB peak on A100-40GB.
