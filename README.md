# Bayes-optimal vs. MLE recovery in the Gaussian planted matching

Simulation code for *Bayes-optimal estimation for Gaussian planted matchings*.
It measures how well two estimators recover the planted matching as a function of
the signal strength $C$:

* **MLE**: $M_{\rm MLE}=\arg\max_M\langle\Pi_M,W\rangle$, computed exactly by linear assignment.
* **Bayes-OPT**: $M_{\rm OPT}=\arg\max_M\sum_{e\in M}p_e(W)$, where the posterior
  marginals $p_e$ come from swap-chain MCMC and the rounding is an exact
  sparse assignment.

The model is $W=\mu\Pi_{M^\ast}+Z$ with $\mu^2=C\log n$ and a uniformly random planted $M^\ast$.
Recovery is the overlap $|\widehat M\cap M^\ast|/n = 1-r_n$.

Every instance also records $q_n=\mathbb E\langle|M\cap M^\ast|\rangle/n=1-\mathsf{mmse}_n$.
By the Nishimori identity this is the overlap of a posterior draw, so
`configs/qn_scaling.yaml` reproduces the write-up's MCMC figure as a check on the sampler.

## The sampler

`pmsim/swapchain.py` contains three exact Metropolis–Hastings kernels on perfect matchings.
All of them move by cyclic rotations, and a rotation of length 2 is the swap move from the write-up.

| kind | move | when to use |
|---|---|---|
| `uniform` | swap chain exactly as in the write-up | reference / ablation |
| `rotation` | informed 2–5-cycle rotations | ablation |
| `walk` (default) | ejection chains with natural closure, any length ≤ `max_len` | everything |

Why `walk`: close to $C\approx2$–$4$, posterior draws differ from $M^\ast$ and from each other by
**long alternating cycles** (we measured lengths of 13–300). Two-swaps can only build those through
high-energy intermediate states. In the `walk` kernel each displaced row draws a new column from
$q_i(j)\propto e^{\mu W_{ij}+b_j}$ until the walk returns to the starting column. The MH ratio then
reduces to $\prod_\ell(1-q_{\rm old})/(1-q_{\rm new})$, so the move is close to rejection-free.

The column prices $b_j$ come from a few Sinkhorn sweeps. They cancel exactly around any cycle, so
they affect efficiency only and never the target distribution. In our calibration runs, the
random-start and MLE-start chains agreed at $n=1024,\ C=3$ after about $20\,n\log n$ proposals.
Plain swaps needed about $2\times10^4\,n\log n$.

Efficiency details:

* Each move costs $O(L)$, using $\pi$ and $\pi^{-1}$ plus $O(1)$ alias tables.
* The MH ratio is a product of ratios of entries of $Q$, so there is no `exp` or `log`
  in the hot loop.
* Marginals are **exact** ergodic averages: each row adds its holding time to an integer
  counter only when it leaves its current edge.
* Random numbers come from an inline xoshiro256** stream per chain.
* Four chains per instance run in lock-step, so their DRAM-bound memory accesses overlap.

**Correctness.** `tests/` checks each kernel against brute-force enumeration of all $n!$
matchings. This covers annealing, $\varepsilon$-mixing, capped rejection, Sinkhorn prices and
grouped counts. Run it with `python -m pytest -q tests`.

**Convergence diagnostics (per instance).** Two chains start at a random matching (with an
annealed burn-in) and two start at the MLE. They approach the posterior from opposite sides, so
the gap between them (`q_rand` vs `q_mle`, and `bayes_agree`) measures non-convergence. The
`diagnostics` figure plots this gap. If it is not small next to the Bayes–MLE gap at a given
$(n,C)$, increase `burn_factor`. The Nishimori energy check $\sqrt n(\langle\Pi_M,W\rangle/n-\mu)$
is stored as `ez_rand` / `ez_mle` and should be $O(1)$.

## Layout

```
pmsim/
  model.py        instance generation (float32 W, random planted permutation)
  swapchain.py    numba kernels (uniform / rotation / walk), Sinkhorn, alias tables
  estimators.py   MLE, Bayes rounding, run_instance (all estimators + diagnostics)
  experiment.py   plan / run (sharded, resumable) / aggregate / plot CLI
  plots.py        figures in the style of the write-up;  style.py  rcParams
  theory.py       AoN limit, MLE rate delta(C), q_n leading term + scalar reference
configs/          recovery.yaml (main), qn_scaling.yaml (validation), smoke.yaml
slurm/            Torch: config.sh, setup_env.sh, pyrun.sh, submit.sh, *.sbatch
tests/            exactness tests against enumeration
```

## Running locally

```bash
pip install -r requirements.txt
python -m pytest -q tests
python -m pmsim.experiment run  configs/smoke.yaml --workers 4   # about 1 minute
python -m pmsim.experiment plot configs/smoke.yaml                # -> results/smoke/figures/
```

## Running on NYU Torch

Log in with `ssh <NetID>@login.torch.hpc.nyu.edu` (NYU network or VPN), then:

```bash
cd $SCRATCH && git clone https://github.com/<you>/planted-matching-sim.git && cd planted-matching-sim
my_slurm_accounts                              # find your torch_pr_... account
export TORCH_ACCOUNT=torch_pr_XXX_XXXXX        # or edit slurm/config.sh
bash slurm/setup_env.sh                        # one time: Apptainer overlay + Miniforge, runs tests
bash slurm/submit.sh configs/qn_scaling.yaml   # validation run (sampler vs theory)
bash slurm/submit.sh configs/recovery.yaml     # main experiment
```

* `submit.sh` prints a cost/memory plan, then submits a job array. Each array task gets
  64 CPUs and runs one shard of instances in a process pool, with jobs balanced by
  longest-processing-time first. A dependent job then aggregates the results and draws
  the figures.
* Results are written to `$SCRATCH/planted-matching-sim/<config>/`: `raw/shard_*.jsonl`
  (one line per instance), `results.csv`, `summary.csv`, and `figures/*.pdf|png`.
* Runs are **resumable**. Finished instances are skipped, so re-running `submit.sh` after a
  timeout or a preemption only computes what is missing. Set `PMSIM_PREEMPT=1` to also use
  preemptible partitions.
* Set `PMSIM_ENV_MODE=venv` (and optionally `PMSIM_PY_MODULE`) to use a plain virtualenv
  instead of the overlay.
* Memory: about $40n^2$ bytes per worker (2.6 GiB at $n=8192$). The runner lowers the number
  of workers automatically to fit `--mem`.
* `python -m pmsim.experiment plan CONFIG` gives a rough core-hour estimate. The default
  `recovery.yaml` (5,040 instances up to $n=8192$) is roughly 50 core-hours.

## Figures

| file | content |
|---|---|
| `recovery_vs_C` | overlap of Bayes-OPT, MLE and a posterior draw vs $C$, one panel per $n$, with the AoN limit |
| `bayes_advantage` | paired difference $\mathrm{ov}(M_{\rm OPT})-\mathrm{ov}(M_{\rm MLE})$ vs $C$ for each $n$ |
| `error_scaling` | $r_n$ vs $n$ (log–log) for $C\in(2,4)$, against the rate $n^{-\delta(C)}$, $\delta(C)=\tfrac12-\tfrac14\sqrt{C(4-C)}$ |
| `qn_scaling` | $q_n$ vs $n$ for $C\in\{\tfrac14,\tfrac23,1\}$ with the leading term and scalar reference (as in the write-up) |
| `diagnostics` | random-start vs MLE-start chain agreement |

Error bars are 95% intervals over instances. Set `PMSIM_USETEX=1` to typeset the figures with
a system LaTeX installation.
