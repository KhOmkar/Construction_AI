# Construction Site AI: Local Implementation Plan

*Combines the project prompt and the experiment-tracking addendum · Runs locally on your PC · 1 November 2026*

## 1. What this plan is

This plan turns the two agent prompts (the main project prompt and the tracking and evaluation addendum) into one ordered build plan for a local machine. It defines the phases, what each phase must produce, and the checkpoint (gate) you approve before the agent moves on. The prompts were written for Colab, so section 3 gives an override block to paste first. This version is fixed to an RTX 3050 (6 GB), 100 epochs per generative model, one Jupyter notebook per model, and a final live dashboard. The focus is the five generative models, their evaluation and their outputs. There is no detector experiment.

## 2. Your hardware: RTX 3050 with 6 GB (Tier B)

Your RTX 3050 with 6 GB of VRAM is Tier B: all five models fit at 64x64 with mixed precision, 128x128 is out of reach, and diffusion is the slow one. Phase 0 still has the agent confirm the hardware. Settings by tier, for reference:

| Tier | Hardware | Resolution | Batch size | What to expect |
| --- | --- | --- | --- | --- |
| A | No NVIDIA GPU (CPU only) | 32x32 or 64x64 | 16 to 32 | Autoencoder and VAE are fine. GAN, diffusion and transformer will be very slow. Consider renting a cloud GPU for those three. |
| B | NVIDIA GPU with 6 to 8 GB | 64x64 | 64 to 128 | All five models are feasible at 64x64 with mixed precision. Diffusion is the slowest. |
| C | NVIDIA GPU with 12 to 24 GB | 64x64, then 128x128 | 128 to 256 | Comfortable. 128x128 and multiple seeds become realistic. |

Time per model depends heavily on the GPU, so I give no hour counts here. The agent measures seconds per epoch in a smoke test (Phase 2) and projects total time before any long run.

## 3. Local override block (paste before the main prompt)

This replaces the Colab constraints in the main prompt:

```
LOCAL OVERRIDE (replaces all Colab references in the earlier instructions)
- This project runs LOCALLY on my PC, not on Colab. Ignore anything about Google Drive, Colab notebooks and session disconnects.
- My GPU is an NVIDIA RTX 3050 with 6 GB VRAM (Tier B). Phase 0 must still confirm OS, CPU, RAM, free disk, GPU name, VRAM, CUDA version and Python version, and report any difference from this.
- Create an isolated environment (venv, conda or uv) with a pinned requirements.txt, and install the CUDA build of PyTorch that matches my driver. Tell me before installing anything large.
- Train EVERY generative model (Autoencoder, VAE, GAN, Diffusion, VQ-VAE + Transformer) for exactly 100 epochs. Never lower the epoch count silently: if a run is too slow, propose a smaller model or a per-epoch sample cap and ask me.
- Training runs as command-line scripts: `python -m src.train --config configs/vae.yaml`. Each model ALSO gets its own Jupyter notebook (training view, graphs, evaluation, input-to-output demo) that imports from src/ and never duplicates model code.
- At the end, build a local dashboard (Streamlit) with all models, their features, graphs and live generation from the trained checkpoints.
- Keep checkpoint and resume support, since long runs can be interrupted by power, sleep or crashes.
- Use num_workers carefully (on Windows start with 0 or 2). Keep the preprocessed crop arrays in RAM or memory-mapped files for speed.
- Disable sleep and avoid other GPU-heavy programs during training. Log GPU temperature and memory.
- Write all logs, checkpoints and outputs locally under the project folder. Never write into data/.
```

## 4. Target repository layout

```
construction-genai/
  data/                  raw downloads (read-only, never modified)
  processed/             scenes/ (unified YOLO) and crops/ (npz per split)
  configs/               one YAML per model and per tier
  src/
    common/              dataset, seed, checkpoint, logger, metrics, grids
    models/              ae, vae, gan, diffusion, vqvae, transformer
    train.py             unified entry point
    evaluate.py          shared generation metrics
    compare/             cross-model evaluation and output grids
    analysis/            aggregate.py, plots, tables, registry builder
  notebooks/             00_data_overview, 01_autoencoder, 02_vae, 03_gan,
                         04_diffusion, 05_transformer, 06_model_comparison
  dashboard/             app.py, pages/, registry.yaml, assets/
  runs/                  one folder per run (config, logs, samples, checkpoints)
  results/               all_runs.csv, tables/, figures/, notebooks_html/
  reports/               data_audit, dataset_stats, licenses, metrics_definitions, final_report
  tests/                 smoke tests (scripts, notebooks, dashboard)
  PROGRESS.md  README.md  requirements.txt
```

## 5. Phase plan at a glance

| Phase | Name | Main output | Gate you approve |
| --- | --- | --- | --- |
| 0 | Hardware check and data audit | hardware report, data\_audit.md, class-mapping table | Mapping table and folder plan |
| 1 | Data preparation | unified scenes, deduplicated manifest, splits, 64x64 crops, captions, dataset\_stats.md, notebook 00 | Sample grids look correct |
| 2 | Scaffold and tracking | environment, logger, metrics library, config system, notebook template, smoke tests | Dummy 2-epoch run logs every required field |
| 3 | Autoencoder and VAE | both trained for 100 epochs, notebooks 01 and 02, anomaly AUROC, latent plots, beta sweep | Results and notebooks reviewed |
| 4 | GAN and Diffusion | both trained for 100 epochs, notebooks 03 and 04, synthetic image folders, FID tracking, guidance sweep | Results and notebooks reviewed |
| 5 | VQ-VAE and Transformer | trained for 100 epochs, notebook 05, token-generated images, temperature sweep | Results and notebook reviewed |
| 6 | Cross-model evaluation and comparison | notebook 06, shared-protocol metrics for all five models, side-by-side output grids | Comparison reviewed |
| 7 | Aggregation and report | all\_runs.csv, tables, figures, final\_report.md, dashboard registry | Final review of results |
| 8 | Dashboard | live app with all models, graphs, output gallery and live generation | You test every page |

The order matters. Phases 0 to 2 produce nothing flashy but make every later result trustworthy, so do not skip or rush them.

## 6. Phase details

### Phase 0: Hardware check and data audit

1. Detect hardware and pick the tier (section 2). Save to `reports/hardware.md`.
2. Walk `data/`, identify each dataset's format (YOLO, VOC, COCO, CSV, folders), image counts, annotation counts, classes and counts, image sizes, existing splits, licenses, and corrupt files.
3. Propose the class-mapping table to the unified schema: helmet, no\_helmet, vest, no\_vest, person. Anything ambiguous becomes a question to you.
4. **Gate:** you review the audit and approve or correct the mapping.

### Phase 1: Data preparation

1. Convert all sources to one YOLO-format scene dataset with a `manifest.csv` (image id, source, original path, split).
2. Remove exact and near duplicates with perceptual hashing and report counts.
3. Split 80 / 10 / 10 by image with a fixed seed. Keep any dataset's own test split as the real-only `final_test` that no generative model ever trains on.
4. Build 64x64 crops (128x128 optional) saved as compressed arrays per split, with a minimum crop size.
5. Generate auto-captions from labels.
6. Produce `dataset_stats.md` and PNG sample grids (16 crops per class).
7. **Gate:** you inspect the grids for label noise and approve.

### Phase 2: Scaffold, tracking and smoke tests

This is where the addendum is built, before any model code. It also creates the notebook template (section 8) and the dashboard data contract (section 9).

1. Environment, config system (YAML), seeds, checkpoint save and resume.
2. Logger that writes the run folder: `config.yaml`, `env.json`, `train_log.csv`, `epoch_log.csv`, `eval_metrics.json`, `timing.json`, samples and checkpoints. Logs append and flush every epoch, and resume continues the same log.
3. Metrics library: reconstruction (MSE, PSNR, SSIM, LPIPS), anomaly (AUROC, AUPRC, FPR at 95% TPR, F1), generation (FID, KID, Inception Score, precision, recall, density, coverage, class-conditional accuracy, LPIPS diversity, nearest-neighbour novelty), cost (parameters, FLOPs, memory, speed).
4. Train the reference classifier on real crops (needed for class-conditional accuracy) and report its test accuracy.
5. Aggregation script skeleton and `metrics_definitions.md`.
6. Smoke test: a 2-epoch dummy run per model that verifies every required field exists. The agent also records seconds per epoch to project real training times for your tier.
7. **Gate:** you review the smoke-test output and the time projection.

### Phase 3: Autoencoder and VAE

- **Autoencoder (100 epochs):** trained on compliant crops only. Latent size sweep (64, 128, 256, 512). Metrics: reconstruction on val and test per class, anomaly AUROC and AUPRC, threshold chosen on validation only, error histograms. Notebook: `01_autoencoder.ipynb`.
- **VAE and conditional VAE (100 epochs):** KL warm-up, beta sweep (0.5 to 4), active units, interpolations, UMAP or t-SNE of latents, reconstruction versus KL trade-off table. Notebook: `02_vae.ipynb`.
- **Gate:** you review the results and both notebooks before moving on.

### Phase 4: GAN and Diffusion

- **Conditional GAN (100 epochs):** spectral norm, DiffAugment, FID every 5 epochs to pick the best checkpoint, diversity (mode-collapse) tracking. Output: a labeled folder of synthetic no\_helmet and no\_vest images. Notebook: `03_gan.ipynb`.
- **Conditional diffusion (100 epochs):** small UNet sized for 6 GB, class embedding, EMA, classifier-free guidance, DDIM sampling. Sweeps over guidance scale and sampling steps, EMA versus non-EMA, per-timestep loss buckets. Output: a labeled synthetic folder. Notebook: `04_diffusion.ipynb`.
- **Gate:** you review the results and both notebooks.

### Phase 5: VQ-VAE and Transformer

- **VQ-VAE (100 epochs):** codebook usage, perplexity, dead codes, reconstruction metrics.
- **Transformer (100 epochs):** token cross-entropy, perplexity, top-1 and top-5 accuracy, per-position loss, sweeps over temperature and top-k, generation metrics. Notebook: `05_transformer.ipynb` covers both stages.
- **Optional text add-on:** only after everything above is finished.
- **Gate:** you review the results and the notebook.

### Phase 6: Cross-model evaluation and comparison

1. Re-evaluate the best checkpoint of every model with one identical protocol: same real reference set, same sample count, same preprocessing, same feature extractor.
2. Reconstruction metrics for the autoencoder and VAE; shared generation metrics for the VAE, GAN, diffusion and transformer (FID, KID, Inception Score, precision, recall, density, coverage, class-conditional accuracy, LPIPS diversity, nearest-neighbour novelty).
3. Output comparison: the same class and seed across the generators, side-by-side grids per class, best and worst samples, and failure cases with a short explanation.
4. Bootstrap confidence intervals for FID, KID and anomaly AUROC; 3 seeds only for the cheap models (autoencoder and VAE).
5. Notebook `06_model_comparison.ipynb` shows the tables, charts and side-by-side outputs.
6. **Gate:** you review the comparison before the report and dashboard.

### Phase 7: Aggregation and report

- `results/all_runs.csv`, Markdown and LaTeX tables, loss and metric plots (PNG and PDF at 300 dpi), `experiment_log.md`, and HTML exports of the executed notebooks in `results/notebooks_html/`.
- `dashboard/registry.yaml`: the best run and checkpoint for each model, which the dashboard reads.
- `final_report.md`: motivation, datasets and licenses, methods, experiments, results, discussion, limitations, ethics (privacy of workers).

### Phase 8: Dashboard

1. Build `dashboard/` as a Streamlit app with the pages listed in section 9, reading only `registry.yaml`, `runs/` and `results/`.
2. Test it: every page loads, live generation works for all five models, switching models frees GPU memory, and cached samples show when the GPU is busy.
3. Write one-command run instructions in `README.md` and capture screenshots for the report.
4. **Gate:** you click through every page and approve.

## 7. Training policy: 100 epochs on a 6 GB GPU

Every generative model trains for exactly 100 epochs, and the plan keeps that feasible with small models, mixed precision and resumable runs. Starting settings for your RTX 3050:

| Model | Input | Batch size | Notes for 6 GB |
| --- | --- | --- | --- |
| Autoencoder | 64x64 crops | 128 | Small conv net, fits easily |
| VAE and conditional VAE | 64x64 crops | 128 | Same footprint as the autoencoder |
| Conditional GAN | 64x64 crops | 64 | Generator and discriminator share the GPU, use mixed precision |
| Diffusion | 64x64 crops | 32 to 64 | Small UNet (about 64 base channels, attention only at 16x16), fp16, gradient accumulation to an effective batch of 128 |
| VQ-VAE | 64x64 crops | 128 | Codebook 512, 16x16 token grid |
| Transformer | 256 tokens per image | 32 to 64 | About 6 layers, 8 heads, width 384, fp16 |

Batch sizes are starting points. The agent runs an out-of-memory probe and uses the largest batch that stays under about 85 percent of VRAM.

- An epoch is one pass over the training crops. Early stopping is off, so every run completes 100 epochs.
- Learning-rate schedules span the 100 epochs (cosine for autoencoder, VAE, diffusion and transformer; the GAN keeps its own schedule).
- Checkpoint every epoch. Headline results use the best checkpoint by validation metric within the 100 epochs (FID every 5 epochs for the GAN), and the epoch-100 result is reported beside it.
- The smoke test measures seconds per epoch and projects the 100-epoch time per model. If one run is longer than you can leave the PC on, the agent proposes a smaller model or a per-epoch sample cap and asks you. It never lowers 100 on its own.
- Run long jobs as scripts in a terminal, not inside a notebook, one at a time on this GPU. Close games, video and other GPU programs while training.
- On a laptop, plug in, use the high-performance power plan and watch GPU temperature, since thermal throttling slows long runs.

## 8. Jupyter notebooks: one per model

Each model has its own notebook that runs top to bottom and shows training, graphs, evaluation and input-to-output examples. Notebooks import code from `src/`, so no model code is copied into them.

| Notebook | Covers | Key outputs |
| --- | --- | --- |
| `00_data_overview.ipynb` | Datasets, unified classes, splits, crops | Class counts, imbalance chart, sample grids, duplicate-removal summary |
| `01_autoencoder.ipynb` | Autoencoder and anomaly detection | Loss curves, reconstructions, error heatmaps, ROC curve, AUROC |
| `02_vae.ipynb` | VAE and conditional VAE | Reconstruction and KL curves, samples by class, interpolations, latent plot |
| `03_gan.ipynb` | Conditional GAN | G and D losses, D(real) vs D(fake), FID vs epoch, fixed-noise progression |
| `04_diffusion.ipynb` | Conditional diffusion | Loss and per-timestep loss, samples by class and guidance scale, denoising strip |
| `05_transformer.ipynb` | VQ-VAE stage, then GPT stage | Codebook usage, token loss and perplexity, samples by temperature |
| `06_model_comparison.ipynb` | All five models side by side | Shared metrics table, FID and KID charts, output grids per class, diversity and novelty plots |

Every model notebook (01 to 05) uses the same ten sections, in this order:

1. Goal, role in the project, and architecture summary with parameter count.
2. Config and environment: the YAML used, GPU, seed, run\_id.
3. Data: loaded crops with example images per class.
4. Model definition, imported from `src/models`.
5. Training: a `RUN_TRAINING` switch. On launches the 100-epoch script and tails its log with a live loss plot; off loads the saved run, so the notebook reruns in seconds.
6. Training graphs: every logged loss (train vs validation), learning rate, gradient norm, plus model-specific curves (KL and beta for the VAE, D and G for the GAN, and so on).
7. Evaluation: model-specific metrics and the shared metrics table, read from `eval_metrics.json`.
8. Input to output: the model demo in the table below, plus a cell where you load your own image.
9. Sweeps and ablations, where they were run.
10. Conclusions and limitations, written by you.

| Model | Input to output demo |
| --- | --- |
| Autoencoder | Test crop in; reconstruction and error heatmap out, with a compliant or violation verdict at the chosen threshold |
| VAE | Crop in, reconstruction out; class and seed in, generated crop out; two crops in, interpolation strip out |
| GAN | Class, count and seed in, image grid out; the same noise shown at epochs 1, 25, 50 and 100 |
| Diffusion | Class, guidance scale, steps and seed in, image out, plus the denoising steps |
| Transformer | Class, temperature and top-k in, image out, with sampling shown token by token |

Notebook rules:

- Restart and run all must pass for every notebook (the agent checks it with `jupyter nbconvert --execute`).
- Commit notebooks with their outputs saved, and export an HTML copy to `results/notebooks_html/` for submission.
- Figures made in notebooks are also saved to `results/figures/`, so the report and dashboard reuse them.
- `RUN_TRAINING` defaults to off, so nobody retrains from scratch by accident.
- Blur or crop faces in any image shown in a notebook that leaves your machine.

## 9. Dashboard: all models, features, graphs and live output

A local Streamlit app (one command, opens in your browser) lets you pick any of the five models, see its training graphs and metrics, and generate live output from the trained checkpoints. The agent may propose Gradio for a page if it is simpler, and asks before switching.

| Page | What it shows | Live output |
| --- | --- | --- |
| Overview | Project summary, dataset statistics, headline comparison table | None |
| Data explorer | Class counts, imbalance chart, sample grids | Pick a class and browse crops |
| Autoencoder | Loss curves, reconstruction metrics, ROC curve and AUROC | Upload an image or pick a test crop: reconstruction, error heatmap, verdict, adjustable threshold |
| VAE | Reconstruction and KL curves, latent plot | Pick a class and seed to sample; slider to interpolate between two crops |
| GAN | G and D losses, FID vs epoch | Class, count and seed: generated image grid |
| Diffusion | Loss curves, FID vs guidance scale | Class, guidance scale, steps and seed: image with progress bar and denoising strip |
| Transformer | Loss and perplexity curves, codebook usage | Class, temperature, top-k and seed: image generated with progress shown |
| Compare models | Shared metrics table and bar charts (FID, KID, precision, recall, class accuracy, diversity, novelty, time per image) | Same class and seed across all models, side by side |
| Output gallery | Saved sample grids per model, class and epoch (progress from epoch 1 to 100) | Browse by model, class and epoch |
| About | Metric definitions, licenses, privacy notice | None |

Every model page also has a run selector (defaults to the best run), a seed box for reproducible output, download buttons for images and CSVs, the latency of each generation, and a GPU panel with memory and temperature.

Design rules:

- **Data contract:** the dashboard only reads `dashboard/registry.yaml` (best run and checkpoint per model), `runs/` and `results/`. It never trains.
- **6 GB budget:** load one model on the GPU at a time, free it when you switch pages, and use fp16 inference.
- **GPU busy:** while a training run is active, live generation is disabled and the page shows cached samples with a banner, so the dashboard and training never fight over VRAM.
- **Speed:** the agent measures latency per model. If diffusion at 50 DDIM steps is too slow for a live demo, it adds a labeled 20-step fast mode.
- **Privacy:** uploaded images stay in memory and are never saved, and a blur-faces option is available.

## 10. Experiment budget

With 100 epochs per model on a 6 GB GPU, diffusion and the GAN are the expensive runs, so extra seeds go only where they are cheap. Priority order:

| Priority | Experiment | Epochs | Seeds |
| --- | --- | --- | --- |
| Must | One tuned run of each of the five generative models | 100 | 1 |
| Must | Shared-protocol evaluation of all five models (same reference set, sample count and feature extractor) | none (uses trained models) | 1, with bootstrap confidence intervals |
| Should | Autoencoder latent-size sweep and VAE beta sweep | 100 for the chosen setting; sweep points at 100 if the smoke test shows each under about 30 minutes, otherwise 30 | 1 |
| Should | Diffusion guidance and steps sweep, transformer temperature and top-k sweep | none (sampling only) | 1 |
| Could | Extra seeds for autoencoder and VAE (cheap) | 100 | 3 |
| Could | Extra seeds for GAN, diffusion and transformer | 100 | only if time allows |
| Not planned | 128x128 versions (too heavy for 6 GB) and any detector experiment | none | none |

Sampling-time sweeps cost little because they reuse a trained model. Training-time sweeps and extra seeds cost the most.

## 11. Rules that protect your results

- Never train or tune on the final real-only test set. Thresholds, checkpoints and hyperparameters are chosen on validation only.
- Never overwrite a previous run. Every run gets a new run\_id.
- Use the same evaluation protocol for every generator: same reference set, same sample count, same preprocessing, same feature extractor.
- Compare model families only on shared metrics (FID, KID, precision and recall, class accuracy, diversity, novelty). Raw losses are only for within-model training curves.
- Record every dataset's license in `reports/licenses.md`.

## 12. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| 6 GB of VRAM runs out of memory | Mixed precision, smaller batches with gradient accumulation, small diffusion UNet, out-of-memory probe, no other GPU programs open |
| 100 epochs of diffusion take too long | Small UNet, per-epoch sample cap proposed to you (never a silent epoch cut), overnight resumable runs, one job at a time |
| GPU throttles from heat (especially on a laptop) | Plug in, high-performance power plan, log temperature, pause between jobs |
| Notebooks and scripts drift apart | Notebooks import from src/ and load saved runs, restart-and-run-all test in the smoke tests |
| Live dashboard generation is slow or competes with training | One model on the GPU at a time, fp16, labeled fast mode for diffusion, cached samples while training runs |
| Mixed label conventions across datasets | Phase 0 mapping gate and Phase 1 sample-grid inspection |
| Duplicate images across sources leaking between splits | Deduplication before splitting |
| Training crashes or power loss | Checkpoint and resume, append-only logs |
| GAN instability or collapse | DiffAugment, spectral norm, FID tracking, diversity metric |
| FID unreliable on small sets | Report sample counts, use KID and precision and recall alongside |
| Generated images look good but copy training data or lack diversity | Nearest-neighbour novelty check, LPIPS diversity, precision and recall beside FID |
| Worker privacy in figures, notebooks and dashboard | Blur or crop faces, never save dashboard uploads, offer a blur option |
| Licenses (for example AGPL-3.0 for Construction-PPE) | Track in licenses.md and check before publishing |

## 13. How to use this plan with the agent

1. Paste the local override block (section 3).
2. Paste the main project prompt.
3. Paste the tracking and evaluation addendum.
4. Paste sections 7 to 9 of this plan (training policy, notebooks, dashboard) as the fourth instruction block.
5. Tell the agent: follow this plan phase by phase, stop at every gate, and keep `PROGRESS.md` updated.
6. Review at each gate, and do not let the agent skip Phase 2.

## 14. Definition of done

- All five models trained for 100 epochs each, with every loss and metric from the addendum logged for each run.
- Each model has its own notebook that runs top to bottom and shows training, graphs, evaluation and input-to-output examples.
- The dashboard starts with one command and shows all five models, their graphs and live output.
- The shared-metric comparison table exists for all five models with confidence intervals, and the output gallery shows samples from each.
- Figures and LaTeX tables are generated from `all_runs.csv`, not typed by hand.
- `metrics_definitions.md` and `licenses.md` are complete.
- Any teammate can reproduce a run from its saved `config.yaml` and `env.json`.
