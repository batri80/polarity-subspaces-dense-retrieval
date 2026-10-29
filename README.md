# Polarity Subspaces in Dense Retrieval

Code, data and pre-registration notes for

> Batri Krishnan. *Polarity Subspaces in Dense Retrieval: Representation Geometry, Task-Dependent Relevance, and Cross-Task Interference.* Submitted to *Information Processing & Management*.

The study fits a low-rank model of how dense-retriever embeddings change under negation, tests the resulting polarity subspace by projection against random subspaces and simpler linear baselines, and uses a training-free readout that removes the subspace to examine interference between retrieval tasks that treat negation differently (NevIR, ExcluIR and SciFact), including a controlled negation of SciFact claims with syntactic controls.

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | Training, evaluation and analysis code. `so_train.py` is the entry point; `so_analysis.py` holds the subspace analyses and readouts; `semantic_odometry_probe.py` holds the subspace fitting (Eq. 1–2) and the lexical negation detector. |
| `src/so_scope_eval.py` | Evaluation of the negation cue and scope detector against BioScope. |
| `src/so_ctrl_prepare.py`, `src/so_ctrl_validate.py` | Export of the SciFact claims, and automated validation of the controlled negations with two NLI models and a CoLA classifier. |
| `analysis/` | Statistics scripts that reproduce every number in the paper from the stored per-query results. |
| `controlled_negation/` | The validated negations and controls (`ctrl_v2.json`, `ctrl_variants.json`), the automated validation (`ctrl_validation.json`), the author's check of a stratified sample of 40 claims, the audit sheet for all claims, and the construction scripts. |
| `preregistration/` | Dated pre-registration notes for the reflection test (26 Oct 2026) and the readout test (27 Oct 2026). The controlled-negation rule was fixed on 28 Oct 2026. |
| `results/revision_runs/` | Per-query results of the main runs (RQ1 to RQ3): four zero-shot encoders, E5 fine-tuned with three seeds, MPNet and GTE with two. |
| `results/controlled_negation_run/` | Per-query results of the run that evaluated the validated negations and syntactic controls (E5, MPNet), including per-query NFCorpus and FiQA scores. |
| `results/statistics/` | Statistics outputs (Markdown and JSON) as reported in the paper, and the BioScope evaluation log. |
| `results/earlier_runs/` | Per-query results of the earlier runs used for the secondary analyses (anchored training, random-subspace control, reflection test); read by `analysis/so_stats.py` and `analysis/so_stats_cross.py`. |
| `figures/` | Figure script and the figures of the paper. |
| `logs/` | Console log of the main runs. |

## Requirements

Python 3.10 or later and a CUDA GPU with at least 10 GB of memory for fine-tuning (all reported runs used a single NVIDIA T4). Install with

```bash
pip install -r requirements.txt
```

Models and datasets are downloaded from the Hugging Face Hub on first use (NevIR, SciFact, NFCorpus, FiQA and Natural Questions). ExcluIR is not on the Hub; download `corpus.json` and `test_manual_final.json` from the ExcluIR authors and place them in the working directory. BioScope is downloaded by `so_scope_eval.py` from the University of Szeged.

## Reproducing the experiments

Main runs (one command per encoder; E5 shown):

```bash
cd src
python so_train.py --model intfloat/e5-base-v2 --modes zeroshot plain_ft --epochs 2 --seeds 1 2 3 \
    --excluir test_manual_final.json --scifact --analysis_pack --out_dir rv_ft_e5
```

The other encoders are `sentence-transformers/all-mpnet-base-v2`, `thenlper/gte-base` and `facebook/contriever-msmarco` (zero-shot only for Contriever).

Controlled-negation run with the validated claims and syntactic controls:

```bash
python so_train.py --model intfloat/e5-base-v2 --modes zeroshot plain_ft --epochs 2 --seeds 1 2 3 \
    --scifact --analysis_pack --analysis_light --ctrl_file ../controlled_negation/ctrl_variants.json --out_dir ctrl_e5
```

Scope-detector validation:

```bash
python so_scope_eval.py --out scope_eval.json
```

Fine-tuned runs are not bit-for-bit reproducible because of GPU non-determinism; repeated runs differed by up to 0.02 in NevIR accuracy, with every reported effect keeping its direction and approximate size.

## Reproducing the statistics

From the repository root, using the stored results:

```bash
python analysis/so_stats_rv.py        # subspace removal, baselines, controlled negation (revision runs)
python analysis/so_stats_rv2.py       # natural vs affirmative claims, per-seed penalties
python analysis/split_by_rule.py      # controlled negation split by rule type
python analysis/so_stats_ctrl.py      # validated negations, syntactic controls, paired preservation tests
```

Outputs are written to `results/statistics/`. Bootstrap intervals use fixed random seeds and reproduce the reported values exactly.

## Data and licences

The code is released under the MIT licence (`LICENSE`). The controlled-negation files are derived from SciFact claims (Wadden et al., 2020; CC BY-NC 2.0) and are released under the same terms; they contain claim texts and document identifiers but no abstracts. NevIR, ExcluIR, SciFact, NFCorpus, FiQA and BioScope are not redistributed here and remain under their original licences.

## Citation

If you use this code or data, please cite the paper (see `CITATION.cff`).

## Contact

Batri Krishnan, Sharda School of Computing Science & Engineering, Sharda University, Greater Noida, India (batri.k@sharda.ac.in).
