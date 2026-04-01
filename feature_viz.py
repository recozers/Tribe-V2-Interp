"""
Feature visualization for TRIBE v2 brain encoding model.

Finds the video input that maximally activates a target cortical region
(default: V1) by gradient-based optimization through the full pipeline:

    random frames  ──►  V-JEPA 2 (frozen)  ──►  TRIBE v2 (frozen)  ──►  cortical predictions
         ▲                                                                      │
         └──────────────────── backprop ◄───────────────────────────────────────┘

Algorithm (after Olah et al., "Feature Visualization", Distill 2017):
  1. Start with 64 random noise frames at 256×256, optimize their Fourier
     coefficients instead of pixels directly
  2. Each step: random jitter → ImageNet normalize → V-JEPA 2 → extract
     layer features → zero text/audio → TRIBE v2 transformer → cortical preds
  3. Loss = −mean(target region activation) + λ_fft · spectral_penalty(coeffs)
  4. Backprop to Fourier coefficients, reconstruct frames with inverse FFT

Usage:
    modal run feature_viz.py                              # default V1
    modal run feature_viz.py --target-roi MT              # target MT
    modal run feature_viz.py --target-roi FFA --skip-sweep
"""

import modal
import os

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = modal.App("tribe-v2-feature-viz")
weights_vol = modal.Volume.from_name("tribe-v2-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    # Core ML stack
    .pip_install(
        "torch>=2.5.1,<2.7",
        "torchvision>=0.20,<0.22",
        "numpy==2.2.6",
        "einops",
        "pyyaml",
    )
    # HuggingFace
    .pip_install("transformers>=4.50", "huggingface_hub")
    # neuraltrain / neuralset (TRIBE v2 dependencies)
    .pip_install(
        "neuralset==0.0.2",
        "neuraltrain==0.0.2",
        "x_transformers==1.27.20",
        "pydantic>=2",
        "exca",
    )
    # Other tribev2 deps that must be importable
    .pip_install(
        "moviepy>=2.2.1",
        "soundfile",
        "julius",
        "langdetect",
        "spacy",
        "Levenshtein",
        "gtts",
    )
    # TRIBE v2 itself
    .pip_install("tribev2 @ git+https://github.com/facebookresearch/tribev2.git")
    # Visualization + parcellation
    .pip_install("matplotlib", "mne", "nibabel")
)

# ── Constants ──────────────────────────────────────────────────────────────

CACHE = "/cache"
VJEPA2_REPO = "facebook/vjepa2-vitg-fpc64-256"
TRIBE_REPO = "facebook/tribev2"

NUM_FRAMES = 64       # V-JEPA 2 expects 64 frames per clip
FRAME_SIZE = 256      # 256×256 input resolution
TUBELET_SIZE = 2      # temporal patch grouping
PATCH_SIZE = 16       # spatial patch size
T_TOKENS = NUM_FRAMES // TUBELET_SIZE    # 32 temporal tokens
S_TOKENS = (FRAME_SIZE // PATCH_SIZE) ** 2  # 256 spatial tokens

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Common ROI names → HCP MMP1.0 (Glasser) label(s)
ROI_MAP = {
    "V1": ["V1"],
    "V2": ["V2"],
    "V3": ["V3"],
    "V4": ["V4"],
    "MT": ["MT"],
    "FFA": ["FFC"],                         # fusiform face complex
    "PPA": ["PHA1", "PHA2", "PHA3"],        # parahippocampal areas
}

FSAVERAGE5_VERTS = 10242  # vertices per hemisphere


# ---------------------------------------------------------------------------
# Main function — runs on A100 80 GB
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={CACHE: weights_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=18000,
)
def feature_viz(
    target_roi: str = "V1",
    skip_sweep: bool = False,
    sweep_steps: int = 500,
    full_steps: int = 2000,
    n_restarts: int = 5,
    seed: int = 42,
):
    import random
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel.freesurfer as nbfs
    import numpy as np
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel

    device = torch.device("cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    hf_token = os.environ.get("HF_TOKEN")
    os.environ["HF_HOME"] = f"{CACHE}/hf"

    out_root = Path(CACHE) / "outputs" / target_roi
    out_root.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # 1. Load V-JEPA 2 — frozen video feature extractor
    # ==================================================================
    #
    # V-JEPA 2 ViT-G: 40-layer vision transformer operating on 64-frame
    # video clips at 256×256.  Tubelet size 2 → 32 temporal tokens,
    # patch size 16 → 16×16 = 256 spatial tokens, total 8192 tokens/clip.
    # Hidden dim 1408.
    #
    # We extract intermediate hidden states at layers corresponding to
    # [0.5, 0.75, 1.0] network depth (matching TRIBE v2's training config
    # in defaults.py → data.layers_to_use).

    print(">>> [1/4] Loading V-JEPA 2 (ViT-G)...")
    vjepa = AutoModel.from_pretrained(
        VJEPA2_REPO,
        cache_dir=f"{CACHE}/vjepa2",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        token=hf_token,
    )
    vjepa.to(device).eval()
    for p in vjepa.parameters():
        p.requires_grad_(False)

    n_enc_layers = vjepa.config.num_hidden_layers  # 40
    enc_dim = vjepa.config.hidden_size             # 1408

    # V-JEPA 2 uses OutputRecorder — hidden_states is 0-indexed with
    # length num_hidden_layers (no embedding entry).  hidden_states[i]
    # is the output of encoder layer i.  Clamp to valid range.
    layer_indices = [
        min(int(0.5 * n_enc_layers), n_enc_layers - 1),    # 20 → clamped ok
        min(int(0.75 * n_enc_layers), n_enc_layers - 1),   # 30 → clamped ok
        n_enc_layers - 1,                                    # 39 (last layer)
    ]
    print(f"    {n_enc_layers} layers, dim={enc_dim}, "
          f"extracting at {layer_indices}")
    weights_vol.commit()

    # ==================================================================
    # 2. Load TRIBE v2 — frozen brain encoder
    # ==================================================================
    #
    # TribeModel.from_pretrained downloads config.yaml + best.ckpt from
    # HuggingFace, reconstructs the FmriEncoderModel, and loads weights
    # in average-subject (unseen subject) mode.
    #
    # The model's forward() expects a batch object whose .data dict maps
    # modality names ("video", "text", "audio") to feature tensors of
    # shape (B, n_layers, feature_dim, T).  Missing modalities are
    # zero-filled automatically — this matches the modality dropout used
    # during training, so the model is trained to handle it.

    print(">>> [2/4] Loading TRIBE v2 brain encoder...")
    from tribev2.demo_utils import TribeModel

    tribe_xp = TribeModel.from_pretrained(
        TRIBE_REPO,
        cache_folder=f"{CACHE}/tribe_features",
        device="cuda",
    )
    tribe = tribe_xp._model   # the FmriEncoderModel
    tribe.eval()
    for p in tribe.parameters():
        p.requires_grad_(False)

    print(f"    feature_dims:       {tribe.feature_dims}")
    print(f"    n_outputs (verts):  {tribe.n_outputs}")
    print(f"    n_output_timesteps: {tribe.n_output_timesteps}")
    print(f"    params:             {sum(p.numel() for p in tribe.parameters()):,}")

    # Sanity-check that our V-JEPA 2 features will slot in correctly
    vid_n_layers, vid_dim = tribe.feature_dims["video"]
    assert vid_dim == enc_dim, (
        f"Hidden dim mismatch: TRIBE wants {vid_dim}, V-JEPA 2 has {enc_dim}"
    )
    if vid_n_layers != len(layer_indices):
        # Adjust extraction points to match what TRIBE was trained on
        layer_indices = [
            min(int(f * n_enc_layers), n_enc_layers - 1)
            for f in np.linspace(0.5, 1.0, vid_n_layers)
        ]
        print(f"    adjusted layer_indices → {layer_indices}")
    weights_vol.commit()

    # ==================================================================
    # 3. Look up target ROI vertex indices
    # ==================================================================
    #
    # TRIBE v2 predicts activation for every vertex on the fsaverage5
    # cortical mesh (10 242 vertices/hemisphere × 2 = 20 484 total).
    # We identify which of those vertices belong to the target ROI using
    # the HCP MMP1.0 (Glasser) parcellation.
    #
    # The annot files map each fsaverage vertex → region label.  We
    # download them via MNE, parse with nibabel, and filter to fsaverage5
    # resolution (the first 10 242 vertices of each hemisphere).

    print(f">>> [3/4] Looking up {target_roi} vertices (Glasser parcellation)...")

    # Resolve common name → HCP label(s)
    wanted_labels = ROI_MAP.get(target_roi, [target_roi])

    # MNE's HCP parcellation reader expects fsaverage surfaces to exist.
    # Fetch fsaverage first, then add the HCP MMP1.0 annotation files.
    import mne
    subjects_dir = Path(CACHE) / "freesurfer"
    subjects_dir.mkdir(parents=True, exist_ok=True)
    mne.datasets.fetch_fsaverage(
        subjects_dir=str(subjects_dir), verbose=False,
    )
    mne.datasets.fetch_hcp_mmp_parcellation(
        subjects_dir=str(subjects_dir), accept=True, verbose=False,
    )

    roi_verts = []
    for hemi_prefix, offset in [("lh", 0), ("rh", FSAVERAGE5_VERTS)]:
        annot_path = (subjects_dir / "fsaverage" / "label"
                      / f"{hemi_prefix}.HCPMMP1.annot")
        labels_arr, ctab, names = nbfs.read_annot(str(annot_path))

        for i, name_bytes in enumerate(names):
            name = (name_bytes.decode("utf-8")
                    if isinstance(name_bytes, bytes) else str(name_bytes))
            # L_V1_ROI-lh → V1  /  R_V1_ROI-rh → V1
            clean = name
            if clean.startswith(("L_", "R_")):
                clean = clean[2:]
            clean = clean.replace("_ROI", "")
            if clean.endswith(("-lh", "-rh")):
                clean = clean[:-3]

            if clean in wanted_labels:
                # read_annot(..., orig_ids=False) remaps vertex labels to the
                # positional colortable index, so matched vertices use `i`.
                verts = np.where(labels_arr == i)[0]
                verts = verts[verts < FSAVERAGE5_VERTS]
                roi_verts.extend(verts + offset)

    roi_verts = torch.tensor(
        sorted(set(roi_verts)), dtype=torch.long, device=device,
    )
    print(f"    {len(roi_verts)} vertices for {target_roi} "
          f"(of {tribe.n_outputs} total)")
    assert len(roi_verts) > 0, (
        f"No vertices found for '{target_roi}'.  "
        f"Valid names: V1 V2 V3 V4 MT FFA PPA (or raw HCP labels like FFC, MST, ...)"
    )
    weights_vol.commit()

    # ==================================================================
    # Helper: ImageNet normalization tensors
    # ==================================================================

    mean_t = (torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32)
              .reshape(1, 1, 3, 1, 1))
    std_t = (torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32)
             .reshape(1, 1, 3, 1, 1))

    # ==================================================================
    # Helper: Fourier parameterization + spectral regularization
    # ==================================================================
    #
    # Progressive multi-scale: we optimize Fourier coefficients at
    # increasing resolutions (64 → 128 → 256).  At each stage the
    # spectrum tensor only covers frequencies up to that resolution,
    # so coarse structure is found first.  When upscaling, the existing
    # coefficients are kept and new high-frequency coefficients are
    # initialized to zero (smooth start).
    #
    # V-JEPA 2 always needs 256×256 input, so we zero-pad the spectrum
    # to full size before IRFFT — mathematically equivalent to a
    # band-limited image at full resolution.

    # Color decorrelation matrix (from Lucid / Olah et al.).
    # Natural images have highly correlated R/G/B channels.  This matrix
    # is the SVD square-root of ImageNet's pixel covariance — multiplying
    # by it maps from an *uncorrelated* 3-channel space to natural RGB,
    # so the optimizer can move each decorrelated channel independently
    # without producing rainbow noise.
    color_corr = torch.tensor(
        [[0.26, 0.09, 0.02],
         [0.27, 0.00, -0.05],
         [0.27, -0.09, 0.03]],
        device=device, dtype=torch.float32,
    )
    # Normalize so max column norm = 1 (keeps sigmoid range reasonable)
    color_corr = color_corr / color_corr.norm(dim=0, keepdim=True).max()

    def frames_from_spectrum(spectrum_params):
        """Inverse FFT → color decorrelation → sigmoid → frames in [0, 1].

        spectrum_params may be smaller than full resolution (progressive
        multi-scale).  We zero-pad to 256×256 before IRFFT so V-JEPA 2
        always gets the resolution it expects.
        """
        spectrum = torch.view_as_complex(spectrum_params)
        # spectrum shape: (1, 64, 3, cur_h, cur_w_rfft)
        cur_h, cur_w_rfft = spectrum.shape[-2], spectrum.shape[-1]
        full_w_rfft = FRAME_SIZE // 2 + 1  # 129

        # Zero-pad to full resolution if needed
        if cur_h < FRAME_SIZE or cur_w_rfft < full_w_rfft:
            full = torch.zeros(
                1, NUM_FRAMES, 3, FRAME_SIZE, full_w_rfft,
                dtype=spectrum.dtype, device=spectrum.device,
            )
            # Place low-frequency coefficients: positive freqs at top,
            # negative freqs at bottom (standard FFT layout)
            pos_h = (cur_h + 1) // 2   # positive frequency rows
            neg_h = cur_h // 2          # negative frequency rows
            full[:, :, :, :pos_h, :cur_w_rfft] = spectrum[:, :, :, :pos_h, :]
            if neg_h > 0:
                full[:, :, :, -neg_h:, :cur_w_rfft] = spectrum[:, :, :, -neg_h:, :]
            spectrum = full

        logits = torch.fft.irfft2(
            spectrum, s=(FRAME_SIZE, FRAME_SIZE), norm="ortho",
        )
        # Apply color correlation: einsum over channel dim
        logits = torch.einsum("btchw, cd -> btdhw", logits, color_corr)
        return torch.sigmoid(logits)

    def spectral_penalty(spectrum_params):
        """Penalize high-frequency Fourier energy of the optimized coefficients.

        Builds frequency weights on the fly to match the current spectrum
        resolution (which changes across progressive stages).
        """
        spectrum = torch.view_as_complex(spectrum_params)
        cur_h, cur_w_rfft = spectrum.shape[-2], spectrum.shape[-1]
        fy = torch.fft.fftfreq(cur_h, device=device, dtype=torch.float32)
        fx = torch.fft.rfftfreq(cur_h * 2 - 1 if cur_w_rfft > 1 else 1,
                                device=device, dtype=torch.float32)[:cur_w_rfft]
        freq_r = torch.sqrt(fy[:, None].pow(2) + fx[None, :].pow(2))
        # Quadratic penalty: higher frequencies penalized more
        fw = (freq_r / (freq_r.max() + 1e-8)).pow(2)
        fw = fw.reshape(1, 1, 1, cur_h, cur_w_rfft)
        power = spectrum.abs().pow(2)
        return (fw * power).mean()

    # ==================================================================
    # Helper: full differentiable forward pass
    # ==================================================================

    class Batch:
        """Minimal stand-in for neuralset.dataloader.SegmentData.
        The TRIBE v2 model just needs batch.data[modality] tensors and
        batch.data.get("subject_id", None)."""
        def __init__(self, data_dict):
            self.data = data_dict

    def forward_pass(frames):
        """frames (1, 64, 3, 256, 256) float32 [0,1] → cortical preds.

        Pipeline:
          1. ImageNet-normalize
          2. V-JEPA 2 forward → hidden states at target layers
          3. Reshape (B, 8192, D) → (B, 32, 256, D) → spatial mean → (B, 32, D)
          4. Stack layers → (B, n_layers, D, 32)  (TRIBE v2 format)
          5. TRIBE v2 forward with only video features (text/audio = zeros)
          6. Returns (1, n_vertices, n_output_timesteps)
        """
        normed = (frames - mean_t) / std_t

        # V-JEPA 2 (fp16 via autocast, SDPA for memory-efficient attention)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = vjepa(pixel_values_videos=normed, output_hidden_states=True)

        # Extract target-layer hidden states, pool over spatial tokens.
        # V-JEPA 2 Conv3d patch embedding flattens in (T, H, W) order,
        # so the first S_TOKENS=256 entries at each temporal position are
        # the spatial patches for that frame-pair.
        feats = []
        for li in layer_indices:
            h = out.hidden_states[li]                   # (1, T_tok*S_tok, D)
            h = h.reshape(1, T_TOKENS, S_TOKENS, -1)   # (1, 32, 256, D)
            h = h.mean(dim=2)                           # (1, 32, D)  — spatial pool
            feats.append(h)

        # Stack layers: (1, n_layers, 32, D) → permute to (1, n_layers, D, 32)
        # to match TRIBE v2's expected (B, L, D, T) format in aggregate_features
        vid_features = torch.stack(feats, dim=1).permute(0, 1, 3, 2)

        # TRIBE v2 forward — only video key present; text/audio auto-zeroed
        batch = Batch({"video": vid_features})
        with torch.amp.autocast("cuda", dtype=torch.float16):
            preds = tribe(batch)   # (1, n_vertices, n_output_timesteps)

        return preds

    # ==================================================================
    # Helper: single optimization run
    # ==================================================================

    # Progressive resolution stages.  Steps are split proportionally
    # across stages (more steps at higher res where there's more to refine).
    STAGES = [
        # (spatial_res, fraction_of_total_steps)
        (64,  0.20),   # coarse blobs and dominant orientation
        (128, 0.30),   # mid-frequency edges and structure
        (256, 0.50),   # full detail refinement
    ]

    def _make_init_spectrum(res, run_seed):
        """Random spectrum at a given spatial resolution."""
        torch.manual_seed(run_seed)
        w_rfft = res // 2 + 1
        init = torch.rand(1, NUM_FRAMES, 3, res, res, device=device)
        logits = torch.logit(init.clamp(1e-4, 1 - 1e-4))
        return torch.view_as_real(
            torch.fft.rfft2(logits, norm="ortho")
        ).detach().clone()   # (1, 64, 3, res, w_rfft, 2)

    def _upsample_spectrum(spectrum_params, new_h):
        """Embed existing low-res spectrum into a larger tensor.

        New high-frequency coefficients are initialized to zero so the
        image stays smooth right after upsampling.
        """
        old = torch.view_as_complex(spectrum_params.detach().clone())
        old_h, old_w = old.shape[-2], old.shape[-1]
        new_w = new_h // 2 + 1
        full = torch.zeros(
            1, NUM_FRAMES, 3, new_h, new_w,
            dtype=old.dtype, device=old.device,
        )
        pos_h = (old_h + 1) // 2
        neg_h = old_h // 2
        w_copy = min(old_w, new_w)
        full[:, :, :, :pos_h, :w_copy] = old[:, :, :, :pos_h, :w_copy]
        if neg_h > 0:
            full[:, :, :, -neg_h:, :w_copy] = old[:, :, :, -neg_h:, :w_copy]
        return torch.view_as_real(full).requires_grad_(True)

    def run_optim(n_steps, lam_fft, run_seed, run_dir):
        """Progressive multi-scale optimization run.

        Optimizes Fourier coefficients at 64 → 128 → 256, spending more
        steps at higher resolutions.  Returns (final_frames, loss_list).
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        random.seed(run_seed)

        # Temporal smoothness weight — gentle, 10× less than spectral penalty.
        lam_temporal = lam_fft * 0.1

        losses = []
        global_step = 0
        spectrum_params = None

        for stage_idx, (res, frac) in enumerate(STAGES):
            stage_steps = max(1, int(n_steps * frac))
            w_rfft = res // 2 + 1
            print(f"      stage {stage_idx}: {res}×{res}  "
                  f"({stage_steps} steps, spectrum {res}×{w_rfft})")

            # Initialize or upsample from previous stage
            if spectrum_params is None:
                spectrum_params = _make_init_spectrum(res, run_seed)
                spectrum_params.requires_grad_(True)
            else:
                spectrum_params = _upsample_spectrum(spectrum_params, res)

            # Fresh optimizer + cosine schedule per stage
            lr_max = 0.03 if stage_idx == 0 else 0.01
            lr_min = 0.003 if stage_idx == 0 else 0.001
            opt = torch.optim.Adam([spectrum_params], lr=lr_max)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=stage_steps, eta_min=lr_min,
            )

            for step in range(stage_steps):
                opt.zero_grad()
                frames = frames_from_spectrum(spectrum_params)

                # Random jitter (reflect-padded)
                pad = 8
                frames_2d = frames.reshape(-1, 3, FRAME_SIZE, FRAME_SIZE)
                padded = F.pad(frames_2d, [pad] * 4, mode="reflect")
                padded = padded.reshape(1, NUM_FRAMES, 3,
                                        FRAME_SIZE + 2 * pad,
                                        FRAME_SIZE + 2 * pad)
                dx = random.randint(0, 2 * pad)
                dy = random.randint(0, 2 * pad)
                jittered = padded[..., dy:dy + FRAME_SIZE, dx:dx + FRAME_SIZE]

                # Forward through frozen pipeline
                preds = forward_pass(jittered)

                # Loss
                roi_act = preds[:, roi_verts, :].float().mean()
                freq_reg = spectral_penalty(spectrum_params)
                temporal_reg = (frames[:, 1:] - frames[:, :-1]).pow(2).mean()
                loss = -roi_act + lam_fft * freq_reg + lam_temporal * temporal_reg

                loss.backward()
                opt.step()
                scheduler.step()

                lv = loss.item()
                losses.append(lv)

                if global_step % 50 == 0 or (stage_idx == len(STAGES) - 1
                                              and step == stage_steps - 1):
                    cur_lr = scheduler.get_last_lr()[0]
                    print(f"    step {global_step:4d}  loss={lv:+.4f}  "
                          f"act={roi_act.item():.4f}  fft={freq_reg.item():.4f}  "
                          f"temp={temporal_reg.item():.4f}  lr={cur_lr:.4f}")

                if global_step % 200 == 0:
                    with torch.no_grad():
                        frames_vis = frames_from_spectrum(spectrum_params)
                    _save_grid(frames_vis,
                               run_dir / f"frames_{global_step:04d}.png",
                               f"step {global_step} ({res}×{res})")

                global_step += 1

        # Save final state
        with torch.no_grad():
            final_frames = frames_from_spectrum(spectrum_params)
        _save_grid(final_frames, run_dir / "frames_final.png", "final")
        _save_losses(losses, run_dir / "loss_curve.png", f"λ_fft = {lam_fft}")
        weights_vol.commit()

        return final_frames.detach().clone(), losses

    # ==================================================================
    # Helper: visualization
    # ==================================================================

    def _save_grid(frames_t, path, title=""):
        """Save grid of 8 evenly-spaced frames from the 64-frame clip."""
        f = frames_t[0].detach().cpu().clamp(0, 1)   # (64, 3, H, W)
        idxs = np.linspace(0, NUM_FRAMES - 1, 8, dtype=int)
        fig, axes = plt.subplots(1, 8, figsize=(24, 3))
        for i, idx in enumerate(idxs):
            axes[i].imshow(f[idx].permute(1, 2, 0).numpy())
            axes[i].set_title(f"f{idx}", fontsize=9)
            axes[i].axis("off")
        if title:
            fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_losses(losses, path, title=""):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(losses)
        ax.set(xlabel="Step", ylabel="Loss", title=title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)

    # ==================================================================
    # 4. Lambda_fft sweep
    # ==================================================================

    best_lambda = 1e-4   # fallback if sweep is skipped

    if not skip_sweep:
        print("\n>>> [4a] Lambda_fft sweep <<<")
        lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        final_losses = {}

        for lam in lambdas:
            print(f"\n  ── λ_fft = {lam} ──")
            sweep_dir = out_root / "sweep" / f"lambda_{lam:.0e}"
            _, losses = run_optim(sweep_steps, lam, seed, sweep_dir)
            final_losses[lam] = losses[-1]

        # Comparison bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        labels_str = [f"{l:.0e}" for l in lambdas]
        objectives = [-final_losses[l] for l in lambdas]
        ax.bar(labels_str, objectives)
        ax.set(xlabel="λ_fft", ylabel="Final objective (higher = better)",
               title="Lambda sweep")
        fig.tight_layout()
        fig.savefig(str(out_root / "sweep" / "comparison.png"), dpi=150)
        plt.close(fig)

        best_lambda = min(final_losses, key=final_losses.get)
        print(f"\n    Best λ_fft = {best_lambda} "
              f"(final loss = {final_losses[best_lambda]:.4f})")
        weights_vol.commit()

    # ==================================================================
    # 5. Full runs with best lambda
    # ==================================================================

    print(f"\n>>> [4b] Full optimization ──  λ_fft={best_lambda}  "
          f"{full_steps} steps × {n_restarts} restarts <<<")

    best_loss = float("inf")
    best_frames = None

    for r in range(n_restarts):
        print(f"\n  ── Restart {r + 1}/{n_restarts} ──")
        run_dir = out_root / "full" / f"restart_{r}"
        frames, losses = run_optim(full_steps, best_lambda, seed + r * 1000, run_dir)

        if losses[-1] < best_loss:
            best_loss = losses[-1]
            best_frames = frames
            print(f"    *** new best  (loss = {best_loss:.4f})")

    # Save overall best result
    _save_grid(best_frames, out_root / "best_frames.png",
               f"Best result — {target_roi}, λ_fft={best_lambda}")

    # Save all 64 frames individually
    ind_dir = out_root / "best_individual"
    ind_dir.mkdir(exist_ok=True)
    bf = best_frames[0].cpu().clamp(0, 1)
    for i in range(NUM_FRAMES):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(bf[i].permute(1, 2, 0).numpy())
        ax.axis("off")
        fig.savefig(str(ind_dir / f"frame_{i:02d}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ==================================================================
    # 6. Validation — selectivity check + random baseline
    # ==================================================================
    #
    # Key question: is the optimized stimulus *selective* for the target
    # ROI, or does it just produce high activation everywhere?
    #
    # We measure mean predicted activation for every ROI in ROI_MAP,
    # for three conditions:
    #   (a) the optimized frames (best restart)
    #   (b) random noise frames (baseline)
    #   (c) anti-optimized: what the start looked like (random init)
    #
    # A good result: the target ROI bar is tall for optimized, and
    # similar to random for non-target ROIs.

    print("\n>>> [5/5] Validation — selectivity & baseline <<<")

    # Build vertex indices for all ROIs using the already-downloaded annots
    all_roi_verts = {}
    for roi_name, roi_labels in ROI_MAP.items():
        verts = []
        for hemi_prefix, offset in [("lh", 0), ("rh", FSAVERAGE5_VERTS)]:
            annot_path = (subjects_dir / "fsaverage" / "label"
                          / f"{hemi_prefix}.HCPMMP1.annot")
            labels_arr, ctab, names = nbfs.read_annot(str(annot_path))
            for i, name_bytes in enumerate(names):
                name = (name_bytes.decode("utf-8")
                        if isinstance(name_bytes, bytes) else str(name_bytes))
                clean = name
                if clean.startswith(("L_", "R_")):
                    clean = clean[2:]
                clean = clean.replace("_ROI", "")
                if clean.endswith(("-lh", "-rh")):
                    clean = clean[:-3]
                if clean in roi_labels:
                    v = np.where(labels_arr == i)[0]
                    v = v[v < FSAVERAGE5_VERTS]
                    verts.extend(v + offset)
        all_roi_verts[roi_name] = torch.tensor(
            sorted(set(verts)), dtype=torch.long, device=device,
        )
    print(f"    ROIs indexed: {', '.join(f'{k}({len(v)})' for k, v in all_roi_verts.items())}")

    # Measure activation for optimized frames and random baseline
    def measure_all_rois(frames_input):
        """Forward pass, return {roi_name: mean_activation} dict."""
        with torch.no_grad():
            preds = forward_pass(frames_input)  # (1, n_verts, n_t)
        return {
            name: preds[:, vidx, :].float().mean().item()
            for name, vidx in all_roi_verts.items()
            if len(vidx) > 0
        }

    # Optimized stimulus
    act_optimized = measure_all_rois(best_frames)
    print(f"    Optimized:  {act_optimized}")

    # Random baseline (average over a few seeds for stability)
    random_acts = []
    for rs in range(5):
        torch.manual_seed(seed + 9000 + rs)
        rand_frames = torch.rand(
            1, NUM_FRAMES, 3, FRAME_SIZE, FRAME_SIZE,
            device=device, dtype=torch.float32,
        )
        random_acts.append(measure_all_rois(rand_frames))
    act_random = {
        name: np.mean([ra[name] for ra in random_acts])
        for name in random_acts[0]
    }
    act_random_std = {
        name: np.std([ra[name] for ra in random_acts])
        for name in random_acts[0]
    }
    print(f"    Random:     {act_random}")

    # ── Selectivity bar chart ──
    roi_names = list(all_roi_verts.keys())
    x = np.arange(len(roi_names))
    width = 0.35

    opt_vals = [act_optimized.get(r, 0) for r in roi_names]
    rnd_vals = [act_random.get(r, 0) for r in roi_names]
    rnd_errs = [act_random_std.get(r, 0) for r in roi_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_opt = ax.bar(x - width / 2, opt_vals, width, label="Optimized")
    bars_rnd = ax.bar(x + width / 2, rnd_vals, width, yerr=rnd_errs,
                      capsize=3, label="Random (5 seeds)")

    # Highlight the target ROI
    for i, name in enumerate(roi_names):
        if name == target_roi:
            bars_opt[i].set_edgecolor("red")
            bars_opt[i].set_linewidth(2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, fontsize=11)
    ax.set_ylabel("Mean predicted activation")
    ax.set_title(f"Selectivity check — optimized for {target_roi}")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(str(out_root / "selectivity.png"), dpi=150)
    plt.close(fig)

    # ── Summary stats ──
    target_opt = act_optimized.get(target_roi, 0)
    target_rnd = act_random.get(target_roi, 0)
    other_opt = [act_optimized[r] for r in roi_names if r != target_roi]
    selectivity_ratio = target_opt / (np.mean(other_opt) + 1e-8) if other_opt else float("nan")

    summary = {
        "target_roi": target_roi,
        "target_activation_optimized": target_opt,
        "target_activation_random": target_rnd,
        "lift_over_random": target_opt - target_rnd,
        "selectivity_ratio": selectivity_ratio,
        "all_rois_optimized": act_optimized,
        "all_rois_random": act_random,
    }
    import json
    with open(out_root / "validation.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n    Lift over random:   {target_opt - target_rnd:+.4f}")
    print(f"    Selectivity ratio:  {selectivity_ratio:.2f}× "
          f"(target / mean other ROIs)")

    weights_vol.commit()

    print(f"\n{'='*60}")
    print(f"  Done!  target={target_roi}  best_loss={best_loss:.4f}")
    print(f"  Results in Modal volume 'tribe-v2-weights'")
    print(f"    → outputs/{target_roi}/sweep/        (lambda sweep)")
    print(f"    → outputs/{target_roi}/full/          (5 restarts)")
    print(f"    → outputs/{target_roi}/best_frames.png")
    print(f"    → outputs/{target_roi}/best_individual/")
    print(f"    → outputs/{target_roi}/selectivity.png")
    print(f"    → outputs/{target_roi}/validation.json")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    target_roi: str = "V1",
    skip_sweep: bool = False,
    sweep_steps: int = 500,
    full_steps: int = 3000,
    n_restarts: int = 5,
    seed: int = 42,
):
    feature_viz.remote(
        target_roi=target_roi,
        skip_sweep=skip_sweep,
        sweep_steps=sweep_steps,
        full_steps=full_steps,
        n_restarts=n_restarts,
        seed=seed,
    )
