# neural-dsc

Filtered directed simplicial complexes (DSCs) of neural networks, built from the
activations and parameters of a trained multilayer perceptron for a given set of
inputs. This repository holds the code, the networks and the results behind

> Michael Bleher. *The Tangled Web They Weave: Exploring Neural Representations
> with Directed Simplicial Complexes.* Accepted at NeurReps 2026.


## Installation

The environment is managed with [uv](https://docs.astral.sh/uv/) and pinned in
`uv.lock` to the versions the results were computed with (Python 3.11, PyTorch
2.6.0 on CPU).

```bash
git clone https://github.com/subthaumic/neural-dsc
cd neural-dsc
uv sync
uv run pytest
```

`uv sync` installs the CPU build of PyTorch; no GPU is needed.

## Layout

| Path | Holds |
| --- | --- |
| `src/neural_dsc/` | the package: DSC construction (`acdc/`), networks (`models/`), data and input selection (`data/`, `utils/`), reading and writing complexes (`io/`), drawing (`viz/`) |
| `models/` | the twelve quadrant networks: handcrafted, trained and three untrained seeds for each of the two architectures |
| `scripts/` | `compute_` scripts build complexes and write `results/`; `plot_` scripts read `results/` and draw `figs/`; `report_` scripts read `results/` and print the numbers quoted in the text |
| `results/` | everything the compute scripts wrote for the article, including the complexes, the selected inputs, the $p$-values and the run logs |
| `figs/` | every figure, as drawn from `results/` by the plot scripts |
| `tests/` | unit tests of the package |

Complexes are stored as plain text, one simplex per line with its filtration
value. Quadrant $Q_{k+1}$ of the article is class `k` in file names
(`dsc_class_0.txt` is $Q_1$).

## Reproducing the article from the shipped results

```bash
uv run python scripts/plot_quadrants.py
uv run python scripts/plot_quadrants_untrained.py
uv run python scripts/plot_handcrafted_dsc_logic.py
uv run python scripts/plot_inference_sensitivity.py
uv run python scripts/plot_pvalue_trajectories.py
uv run python scripts/plot_hyperparameter_sweep.py
uv run python scripts/plot_silver_standard.py
uv run python scripts/plot_dsc_vs_activation_separation.py

uv run python scripts/report_accuracies.py
uv run python scripts/report_twohot_recovery.py
uv run python scripts/report_manuscript_stats.py --all
uv run python scripts/report_correction_sensitivity.py
uv run python scripts/report_threshold_vs_closure.py
uv run python scripts/analyze_quadrant_hypotheses.py
```

The plot scripts overwrite the figures in `figs/`. To compare them with the
figures of the article source, run
`uv run python scripts/compare_article_figures.py <article>/figs`, which
rasterises both and compares them pixel by pixel (needs `pdftoppm`).

### Figures

| Article | File in the article's `figs/` | Drawn by | File in `figs/` |
| --- | --- | --- | --- |
| Fig. 2 | `figure2/{mono,poly}_q1_{input,activations,dsc}.pdf` | `plot_quadrants.py` | `quadrants/{mono,poly}_handcrafted/` |
| Fig. 3 | `quadrant_*network.pdf`, `quadrants_*_predicted_scatter.pdf` | `plot_quadrants.py` | `quadrants/<network>/quadrants_<network>_{network,predicted_scatter}.pdf` |
| Fig. 4 | `inference_sensitivity/probe_count_convergence.pdf` | `plot_inference_sensitivity.py` | `inference_sensitivity/` |
| Fig. 5 | `pvalue_trajectories/*.pdf` | `plot_pvalue_trajectories.py` | `pvalue_trajectories/` |
| Figs. 6, 7 | `hyperparameter_sweep/{dt_filtration,probe_tolerance}_stability.pdf` | `plot_hyperparameter_sweep.py` | `hyperparameter_sweep/` |
| Figs. 8--10 | `{mono,poly}_onehot_dsc_logic.pdf`, `poly_q1_vs_random_dsc.pdf` | `plot_handcrafted_dsc_logic.py` | `handcrafted_logic/` |
| Figs. 11, 12 | `trained_dsc/{mono,poly}_q{1..4}.pdf` | `plot_quadrants.py` | `quadrants/{mono,poly}_trained/dsc_class_{0..3}.pdf` |
| Fig. 13 | `{mono,poly}_jaccard_matrix_full.pdf` | `plot_quadrants.py` | `quadrants/{mono,poly}_handcrafted/jaccard_matrix_full.pdf` |
| Fig. 14 | `silver_standard.pdf` | `plot_silver_standard.py` | `silver_standard/` |
| Figs. 15, 16 | `dsc_vs_activation/*.pdf` | `plot_dsc_vs_activation_separation.py` | `dsc_vs_activation/` |

Figure 1 is an illustration and is not drawn here. The exact file-to-file
correspondence is the dictionary `ARTICLE_FIGURES` in
`scripts/compare_article_figures.py`.

### Tables and numbers

| Article | Source |
| --- | --- |
| Accuracies (App. C) | `report_accuracies.py`, on the 1,000-point evaluation set; writes `results/accuracies.csv` |
| Table 2, sizes of the one-hot complexes | `report_manuscript_stats.py`; `results/quadrants_<network>/dsc_statistics_onehot.csv` |
| Table 3, trained against untrained | `results/quadrants_{mono,poly}_untrained/summary.csv` |
| Table 4, recovery of the two-hot inputs | `report_twohot_recovery.py`; writes `results/twohot_recovery.csv` |
| Table 5, comparison with circuit discovery | `results/silver_standard/summary_report.txt` |
| Choice of the decision rule (App. D) | `results/inference_sensitivity/inference_sensitivity.json` |
| Cost of the correction (App. G.2) | `report_correction_sensitivity.py`, `report_threshold_vs_closure.py` |
| Composition checks (App. I.1) | `analyze_quadrant_hypotheses.py`; writes `results/quadrant_hypothesis_checks.json` |

## Rerunning the experiments

The compute scripts reuse whatever is already in their output directory, so on
the shipped `results/` they only check it: each loads the cached complexes,
recomputes the derived files and compares its configuration with the stored
`config_fingerprint.json`, stopping on a mismatch. To recompute an experiment
from scratch, delete its directory under `results/` first. Some producers read
the output of others, so run them in this order:

```bash
uv run python scripts/compute_quadrants_mono_handcrafted.py
uv run python scripts/compute_quadrants_poly_handcrafted.py
uv run python scripts/compute_quadrants_mono_trained.py
uv run python scripts/compute_quadrants_poly_trained.py
uv run python scripts/compute_quadrants_mono_untrained.py   # reuses the trained one-hot inputs
uv run python scripts/compute_quadrants_poly_untrained.py
uv run python scripts/compute_pvalue_trajectories.py         # fingerprint includes the trained checkpoints
uv run python scripts/compute_silver_standard.py             # reads the handcrafted complexes
uv run python scripts/compute_inference_sensitivity.py
uv run python scripts/compute_hyperparameter_sweep.py
```

The networks themselves are in `models/`; the trained and untrained producers
reuse these checkpoints and create a network only when its checkpoint is
missing. Each producer runs its independent constructions in parallel and leaves
four CPUs free. `scripts/README.md` has notes on the individual scripts.

## Citation

Until the article is published, please cite it as

```bibtex
@misc{Bleher2026tangled,
  author = {Bleher, Michael},
  title  = {The Tangled Web They Weave: Exploring Neural Representations
            with Directed Simplicial Complexes},
  year   = {2026},
  note   = {Accepted at NeurReps 2026; arXiv preprint to appear}
}
```

The code itself is archived on Zenodo (DOI to appear).

## License

MIT, see `LICENSE`.
