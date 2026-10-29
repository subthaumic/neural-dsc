# Scripts

Scripts are named by role:

| prefix | what it does | cost |
| --- | --- | --- |
| `compute_` | builds complexes, writes `results/`. No figures. | CPU-hours on a fresh output directory, seconds on a filled one |
| `plot_` | reads `results/`, writes `figs/` | seconds to minutes |
| `report_` | reads `results/`, prints the numbers quoted in the article | seconds |

The split forces the compute scripts to persist everything the figures need,
which is what makes the figures reproducible without a rerun. **Every figure
lives under `figs/`, none under `results/`**, so `results/` can be diffed and
archived without carrying pictures, and a figure can be redrawn without going
near a producer.

| compute | its figures |
| --- | --- |
| `compute_quadrants_{mono,poly}_{handcrafted,trained}.py` | `plot_quadrants.py` → `figs/quadrants/<context>/`; `plot_handcrafted_dsc_logic.py` → `figs/handcrafted_logic/` |
| `compute_quadrants_{mono,poly}_untrained.py` | `plot_quadrants_untrained.py` → `figs/quadrants/<arch>_untrained/seed<n>/` |
| `compute_hyperparameter_sweep.py` | `plot_hyperparameter_sweep.py` |
| `compute_inference_sensitivity.py` | `plot_inference_sensitivity.py` |
| `compute_pvalue_trajectories.py` | `plot_pvalue_trajectories.py` |
| `compute_silver_standard.py` | `plot_silver_standard.py` |

`plot_dsc_vs_activation_separation.py` has no compute half: it reads the
quadrant producers' complexes and input sets, runs the networks on those inputs
to compare activation distances with DSC distances, and writes both
`results/dsc_vs_activation/summary.csv` and the figures.

## Drawing a complex

`neural_dsc.viz.complex` holds the one visual language for directed simplicial
complexes, and every DSC figure in the article goes through it. Complexes are
drawn on the network they were built from, input and output layers included,
because a simplex is a claim about particular neurons. Neurons are unlabelled
and all one colour: position already says which layer a unit is in.

Both orderings of a 2-simplex are drawn. A patch is bounded by the arcs of its
own faces, so `(a, b, c)` rests on the drawn `a -> b` arc and `(b, a, c)` on the
drawn `b -> a` arc, and the two overlap in everything except the lens between
those arcs. Fill opacity is scaled down as the count grows, because seventy
patches at the handcrafted opacity composite into an opaque slab.

Nothing above dimension two is drawn. The trained complexes reach dimensions
three and four (up to 138 simplices at dimension three).

## Order of the producers

Each quadrant producer parallelises over its independent DSC builds, using one
persistent outer pool and leaving four CPUs free; individual constructions are
serial. The handcrafted and trained producers expose 9 work units (four one-hot,
four adjacent two-hot and one random control); each untrained producer exposes
12 (four classes across three seeds).

The trained producers must precede the corresponding untrained producers,
because the latter reuse their one-hot input sets, and both must precede
`compute_pvalue_trajectories.py`, whose fingerprint includes the trained
checkpoints' sha256. `compute_silver_standard.py` reads the handcrafted
complexes. The mono and poly halves are independent of each other, so on a
large machine they can run side by side:

```bash
( uv run python scripts/compute_quadrants_mono_handcrafted.py & \
  uv run python scripts/compute_quadrants_poly_handcrafted.py & wait ) \
&& ( uv run python scripts/compute_quadrants_mono_trained.py & \
     uv run python scripts/compute_quadrants_poly_trained.py & wait ) \
&& ( uv run python scripts/compute_quadrants_mono_untrained.py & \
     uv run python scripts/compute_quadrants_poly_untrained.py & wait ) \
&& uv run python scripts/compute_pvalue_trajectories.py \
&& uv run python scripts/compute_silver_standard.py
```

`compute_inference_sensitivity.py` and `compute_hyperparameter_sweep.py` build
their own complexes and can run at any point. Run long jobs with
`PYTHONUNBUFFERED=1`; piped stdout block-buffers, which makes a healthy job look
hung.

Every producer reuses what is already in its output directory and checks its
configuration against the stored `config_fingerprint.json`, stopping with the
differing keys on a mismatch. Delete an output directory to recompute it.

## Notes on individual scripts

`compute_inference_sensitivity.py` decides the whole decision rule -- epsilon,
alpha, the correction, the trajectory length and the probe count -- by sweeping
them jointly and scoring the resulting complexes. It resumes from
`work_units.pkl` when interrupted. `compute_pvalue_trajectories.py` is its
complement: the sweep discards the raw p-values once it has scored a complex,
so the one question it cannot answer is what a p-value does along a trajectory.
That is one construction per context, not a grid. It takes contexts as
arguments (`mono_handcrafted:0`) and an `--outdir`, which makes it the cheapest
way to rebuild a complex from scratch and compare it with the stored one.

`compute_silver_standard.py` is a pure consumer: it loads the complexes the
quadrant producers wrote and fails on a fingerprint mismatch, naming the
producer script and the differing keys. It never builds a complex itself.
Delete `results/silver_standard/` and rerun it after any producer rerun.

`plot_ablation_trajectory.py` has no compute half and needs no cache: it
rebuilds one model and one input set and draws the intervention itself --
every neuron's activation against the ablation s, for a chosen ablation set and
target. With `--subsets` it draws each proper subset of the ablated set beside
the joint one, which is what makes a higher-order effect visible.

`report_correction_sensitivity.py` reads the p-value caches and reports what
choosing Benjamini--Yekutieli over Benjamini--Hochberg costs: where each cutoff
falls, which simplices the two part company over, and whether the handcrafted
networks' four classes still come out isomorphic under each.

`report_threshold_vs_closure.py` reads the same caches and splits each loss into
the simplex that was tested and failed and the simplex that was never posed
because a face had already failed. Only the first kind is visible in a
trajectory plot.

`report_cycles.py` asks whether the filtration ever leaves a hole: a 2-simplex
whose own s* is larger than any of its edges' stands the three edges alone over
an interval, and that interval is a 1-cycle. It reports late filling, persistent
H1 by GF(2) reduction, and the vertex triples that are never filled. It
self-tests against six complexes with known homology before reading any data.
On the ordered chain complex a reciprocal edge pair is a cycle nothing can
bound, which accounts for every ordered H1 bar the handcrafted networks appear
to have.

`report_manuscript_stats.py` summarises the `dsc_statistics_*.csv` files of one
or all experiments (`--all`), optionally as LaTeX tables (`--latex`).

`report_accuracies.py` evaluates every network on the 1,000-point evaluation
set and writes `results/accuracies.csv`. `report_twohot_recovery.py` reads the
four quadrant producers' Jaccard matrices and writes the table on recovery of
the two-hot inputs to `results/twohot_recovery.csv`.

## Checks

`check_input_sets.py` regenerates every input set of the four quadrant
producers -- one-hot, adjacent two-hot and random control -- with the current
selection code, compares them with the stored ones, and checks that the
untrained producers used the trained one-hot sets. It exits with 1 on any
mismatch.

`compare_article_figures.py ARTICLE_FIGS_DIR` compares the figures in `figs/`
with those of the article source, rasterised at 72 dpi and compared pixel by
pixel, because two PDFs of the same picture differ in their metadata.

## Output structure

Each quadrant producer writes to `results/quadrants_<arch>_<kind>/`:

- `dsc_class_<k>.txt`, `dsc_twohot_<ij>.txt`, `dsc_random.txt`: the complexes
- `probe_data_*.pt` and `probe_data_*_diagnostics.json`: the selected inputs
- `jaccard_*.json`, `jaccard_matrix_full.csv`: Jaccard distances between the complexes
- `dsc_statistics_*.csv`, `dsc_analysis_summary.txt`: simplex counts by
  dimension, maximal dimension, filtration statistics
- `config_fingerprint.json`: the configuration the complexes were built with
