# JAX版

`scgvb/energy/numpy_backend.py` のNumPy・複素ステップ微分版はそのまま残している。
`scgvb/energy/jax_backend.py` はそのクラスを継承し、エネルギー評価と勾配を
倍精度のJAX JIT・自動微分に置き換える。Augmented Lagrangian、停滞時の
規格化エネルギーによる補助処理、収束判定は既存実装を共用する。

```python
from scgvb.energy.jax_backend import SCWF_Energy

scwf = SCWF_Energy(mol, device='cpu')  # NVIDIA環境では 'gpu'
energy, C, Cs = scwf.augmentedLagrangian(C, Cs)
```

従来版を選ぶ場合は `from scgvb.energy.numpy_backend import SCWF_Energy` を使用する。
`calc_diatomic_mol.py` は `--backend numpy`（既定値）または `--backend jax` で
切り替えられる。JAXの `--device` を省略するとJAXの既定デバイスを使用する。

## 実行範囲と精度

- モジュール読込時に `jax_enable_x64=True` を設定し、積分・係数をfloat64で扱う。
- PySCFの積分生成・参照FCI、SciPyの最適化制御はCPU上で動く。
- JAX版では、積分と置換データをインスタンス作成時にデバイスへ転送して保持する。
  分子や積分を変更するときはインスタンスを作り直す。
- エネルギー・勾配評価をJITコンパイルし、自動微分で全変数の勾配を計算する。
  同じ点の目的関数と勾配は一度の評価を共有する。
- SciPyとの間では係数と結果を転送する。最適化全体がGPU内で完結するわけではない。
- 実数軌道用。複素ステップ入力の代わりに `target_gradient` / `rayleigh_gradient`
  を使用する。物理的な複素軌道を扱う実装ではない。
- JITは配列形状が変わると再コンパイルする。同じ形状の距離走査では再利用できる。
- 既定の置換方式では配置ペアの保存量 `(N!)²` は変わらない。
  行列式方式と固定閉殻の活性空間版については `ACTIVE_SPACE.md` を参照。

## 検証と計測

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest \
    tests.test_jax_energy tests.test_physical_energy tests.test_optimizer \
    tests.test_energy_regression tests.test_spin_function -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scgvb.benchmarks.jax --device cpu
```

JAXテストはJAX未導入ならスキップする。既存版はJAXをimportしない。
H₂一重項・三重項、HeH⁺、LiHで固定パラメータのエネルギーと勾配を比較する。
ゼロの重なり因子、複数のスピン係数、キャッシュ更新も検証する。
H₂・HeH⁺・LiHの最適化後のエネルギー・ノルムを独立な行列式展開で照合する。

2026-09-18、Apple M4 CPU、Python 3.12.2、JAX 0.11.2で、現在のLiHスクリプトの
10 Åと0.5〜2.5 Åの全21距離点が完了し、グラフ出力を確認した。
10 ÅのJAXエネルギーは `-7.782107830840677 Eh`。
演算順序が異なるため、最適化経路や局所解までのビット単位の一致は保証しない。

LiH / STO-3G / 1.6 Åのエネルギー＋勾配評価の実測例（20回平均）：

| 評価 | 時間 |
|---|---:|
| JAX初回（コンパイル込み） | 0.163 s |
| NumPy＋複素ステップ微分 | 12.368 ms |
| JAX自動微分（コンパイル後） | 0.266 ms |

この評価部分では約46倍。全計算の速度比ではない。計測では同一点キャッシュを
各測定前に解除し、ホストへの結果取得まで含めてGPUでも同期されるようにしている。
NVIDIA GPU上での実測は、このMacでは行っていない。

## Bounded determinant spaces: stable expectation values

The optional `--energy-evaluation determinant` (NumPy and JAX) evaluates the same
SCWF through determinant amplitudes in an
orthonormal AO basis (`scgvb/energy/stable_jax.py`). These amplitudes are derived from
nonorthogonal orbital coefficients and spin coefficients; they are not additional
optimization variables. No Full CI eigenvalue problem is solved. The original
nonorthogonal orbitals remain the output. Permutation evaluation remains the default.
The optional evaluator supports at most six electrons in each spin sector and
at most 2,000,000 entries in each determinant, compound-minor, primitive-spin
coefficient, or transformed two-electron integral array. It supports 12e/12o singlets. Polynomial compound matrices,
checkpointed reverse mode, tiled Hamiltonian application and an analytic energy
VJP replace factorial minor expansion and dense transition tensors.
See [HPC_README.md](HPC_README.md) for the single-GPU distribution and benchmarks.

The norm is a sum of squared amplitudes, which avoids cancellation between
positive and negative permutation contributions. Polynomial minors also allow
derivatives at singular subblocks. This reduces one source of numerical error,
but does not guarantee convergence near a vanishing wavefunction or a singular
parameterization. `--verbose` and result metadata identify the evaluator.
The augmented Lagrangian algorithm and NumPy backend remain available.

This is an experimental comparison option, not a demonstrated Li2 convergence fix.
Initialization no longer builds permutation data in determinant mode.
Fixed inactive orbitals reduce the working electron count and orbital dimension;
see [ACTIVE_SPACE.md](ACTIVE_SPACE.md).
Python API: `SCWF_Energy(mol, compute_fci=False, energy_evaluation="determinant")`.

## Optional joint orbital/spin correction

`--joint-correction` (TOML `[optimizer] joint_correction = true`) adds one bounded
joint correction after each complete alternating cycle. It requires determinant
energy evaluation, `method = "alternating"`, and `spin_solver = "projected"`.
It is disabled by default; `--no-joint-correction` preserves previous behavior.
Existing simultaneous, BFGS, wavefunction and single-orbital methods remain available.

`scgvb/optimization/coupled.py` selects at most six tangent directions from orbital
and spin gradients, the projected spin response, and an earlier accepted joint
step. It forms a small physical wavefunction tangent Gram matrix, including
orbital/spin cross terms. A central difference of retracted physical gradients
approximates the reduced energy Hessian: at most twelve perturbed gradient
calls, without constructing the full parameter Hessian. The JAX metric uses
JVPs of derived normalized determinant amplitudes, not independent CI variables.
NumPy uses complex-step derivatives of those amplitudes.

Metric whitening, a positive curvature shift, and a physical trust radius
produce a descent proposal. Actual energy decrease, the unchanged cycle norm
floor, orbital displacement <= 0.1, and wavefunction distance <= 0.05 are checked
before accepting. An inaccurate curvature estimate or failed line search retains
the incoming state. The full normalized physical gradient <= 1e-6 and the existing
normalization tolerance remain required for convergence. Rank in the small
correction space is not a convergence criterion.

This is a finite-difference subspace Newton approximation, not the complete
orbital/spin Newton solve. Timing, accepted corrections, metric spectrum,
curvature spectrum, finite-difference asymmetry, trial rejections and before/after
states are recorded in the optional optimization diagnostics. Fe(CO)5 performance
and convergence must be assessed by the controlled comparison, not inferred
from the small-system validation.

Fixed-orbital spin initialization is independently selectable with
`--initial-spin-guess eigen` and `--initial-spin-overlap-cutoff` (default 1e-10).
It applies before every chosen optimizer and before inactive-orbital relaxation.
The default `uniform` preserves legacy initialization and saved restart Cs.
Explicit `eigen` also re-solves Cs on restored orbitals. Determinant evaluation
uses the centered projected solver; permutation evaluation uses spin matrices.
The initialization and optimizer spin cutoffs are independent.

`--joint-correction --joint-correction-start 21` uses ordinary alternating
updates for sweeps 1–20 and adds the existing joint correction at the end of
sweep 21 and every subsequent sweep. This preserves the current state at
switching. The default start is 1. Numbering is local to each optimizer call,
including restarts and inner inactive-orbital solves. Physical convergence in
warmup ends the solve without waiting for the correction stage. Bounded warmup
can continue across an energy plateau until the scheduled stage; the five-cycle
stall counter restarts at the switch. Energy increases, iteration limits and
full gradient/normalization tests remain enforced. This schedule is not yet
validated for Fe(CO)5 convergence; see `INPUT_FILE.md` for details.
