# Spin Coupled Wave Function

このリポジトリにはプログラム本体、利用説明、分子構造・TOML入力例を収録しています。
テストコード、検証データ、計算結果、ログは含めていません。
説明書にあるテスト実行や保存済み計算結果からの再開には、別途それらのファイルが必要です。

## コード構成

ライブラリは `scgvb` パッケージにまとめています。NumPyとJAXの計算方式・最適化アルゴリズムは共通のパッケージから利用できます。

```text
scgvb/
  energy/        # NumPy・JAXのエネルギー評価、行列式、複合小行列式
  orbitals/      # 活性空間、AO・π軌道、初期軌道、再開、分子積分
  optimization/  # 同時・交互・一軌道・core最適化、スピン初期値、診断
  spin/          # スピン関数、分枝図、電子配列、置換表現
  io/            # TOML入力、軌道出力、軌道ビューア
  analysis/      # 電子密度、原子電荷、結合次数
  cli/           # 分子計算・二原子分子スキャンのコマンド処理
  benchmarks/    # 性能測定と評価器比較
tools/           # HPC配布アーカイブ作成
hpc/             # 実行環境確認、ジョブスクリプト、診断コマンド
examples/        # 分子構造とTOML入力例
tests/           # ローカルの回帰テスト（GitHub・配布対象外）
```

これまでの `python calc_molecule.py`、`python calc_diatomic_mol.py`、
`python generate_orbitals.py`、`python build_hpc_bundle.py` は引き続き利用できます。
Pythonからのインポートは、次のようにパッケージ名を指定してください。

```python
from scgvb.cli.molecule import calculate, build_molecule
from scgvb.energy.numpy_backend import SCWF_Energy
from scgvb.orbitals.active_space import prepare_active_space
```

`python -m scgvb.cli.molecule` や `python -m scgvb.analysis.population` でも実行できます。
パッケージをインストールする場合は `python -m pip install .`、開発中は
`python -m pip install -e .` を使います。インストール後は `scgvb-calculate`、
`scgvb-diatomic`、`scgvb-orbitals`、`scgvb-population` コマンドも利用できます。
JAXは `requirements-cpu.txt` または `requirements-gpu-cuda13.txt` で追加してください。

ローカルのテスト一式は `python -m unittest discover -s tests -t . -p 'test_*.py'` で実行します。

## 入力ファイルからの実行

分子・基底関数・活性空間・最適化設定をTOMLファイルにまとめられます。

```bash
python calc_molecule.py examples/inputs/allyl.toml
```

アリルラジカル（cc-pVDZ、3π電子、core緩和）の設定例を同梱しています。
[入力ファイルの書き方](INPUT_FILE.md)を参照してください。既存のCLIも使用できます。

Codes for "Augmented Lagrangian method for optimizing non-orthogonal orbitals"

# Dependency
- pyscf
- numpy
- scipy
- sympy
- matplotlib
- jax (optional, JAX backend only)

# Usage
General polyatomic single-point calculations (no Full CI; optional HF initial orbitals):

```bash
python calc_molecule.py --xyz examples/h3_plus.xyz --charge 1 --multiplicity 1 \
    --backend jax --device cpu --output h3_plus.json
python calc_molecule.py --xyz examples/h4.xyz --output h4.json
```

See [MOLECULE_INTERFACE.md](MOLECULE_INTERFACE.md) for geometry input, results,
the Python API and electron-count limitations. The existing diatomic test is
retained separately below.

Display the nonorthogonal orbitals saved in a calculation result:

```bash
python generate_orbitals.py h3_plus.json --output-dir h3_orbitals
```

Open `h3_orbitals/index.html` in a browser, or use the generated Cube files in
a molecular viewer. See [ORBITAL_VIEWER.md](ORBITAL_VIEWER.md).

Existing NumPy / complex-step augmented-Lagrangian version (default):

```bash
python calc_diatomic_mol.py
```

JAX float64 / automatic-differentiation version, using the same optimizer:

```bash
python calc_diatomic_mol.py --backend jax --device cpu
```

On a machine with a working NVIDIA/CUDA JAX installation:

```bash
python calc_diatomic_mol.py --backend jax --device gpu
```

Molecule, charge, spin, basis and distance range are configured in
`scgvb/cli/diatomic.py`. An explicitly requested unavailable GPU raises an error;
it does not silently fall back to CPU. See [JAX_BACKEND.md](JAX_BACKEND.md).

The diatomic scan defaults to `--energy-evaluation both`: permutation and
determinant SCGVB optimizations run independently, alongside a Full CI reference.
Both methods receive the same initial guesses and then each carries its own
converged `C` and `Cs` to the next distance. If one method fails, its curve stops
at its last converged point while the other method and Full CI continue. There
is no evaluator switching in a comparison. Single-method runs remain available
with `--energy-evaluation permutation|determinant`; `--no-evaluator-fallback`
is retained as a compatibility option (comparison runs never use a fallback).

```bash
python calc_diatomic_mol.py --backend jax --device cpu --output-dir runs/diatomic
```

`plot_energy.pdf` (also saved as PNG) compares **absolute total energies** in Eh for determinant,
permutation and Full CI on a common axis. `plot_Cs.pdf` (also PNG) shows the squared spin
coefficients of both SCGVB methods, using colors for coefficient indices and
line styles for methods. These squared coefficients are not generally physical
configuration probabilities for nonorthogonal spatial functions.
`diatomic_results.json` retains the raw converged energies, `C`, signed `Cs`,
Full CI references, and failure distances/messages. These files are updated
after every geometry, and completed points are saved on interruption as well.
The 10 Angstrom reference is recorded separately; its failure does not prevent
the absolute-energy scan. There is no automatic checkpoint loading between runs.
Determinant-space size limits still apply.

For JAX determinant evaluation with at most four electrons and more AO basis
functions than electrons, an exact moving orbital-span representation avoids
the expensive full AO determinant contraction. QR is only an internal change
of representation, not an orthogonality constraint on the optimized orbitals.
Rank-deficient orbital matrices use the original polynomial AO evaluator so
that derivatives remain defined. Larger electron counts retain the existing
memory-bounded evaluator.

Use `--initial-guess hf` for RHF/ROHF starting orbitals, or `--initial-guess core` for core-Hamiltonian orbitals. The default `random` retains the previous initialization. See `MOLECULE_INTERFACE.md`.

## 初期スピン結合係数と最適化方式

軌道初期値、スピン結合係数 `Cs` の初期値、最適化方式は独立に選べます。
`Cs` はスピン結合基底の係数であり、軌道と `Cs` から生成する行列式展開係数とは別です。

| TOML設定 | 既定値 | 選択できる動作 |
|---|---|---|
| `[calculation] initial_spin_guess` | `"uniform"` | 新規計算は等係数 `1/sqrt(fs)`。`"eigen"` は初期軌道を固定してスピンの一般化固有値問題を解く。 |
| `[calculation] initial_spin_overlap_cutoff` | `1e-10` | 初期スピン求解の計量固有値に対する相対閾値。最適化用の閾値とは独立。 |
| `[optimizer] method` | `"simultaneous"` | 従来の同時最適化。`"alternating"` は固定軌道でのスピン求解と、係数を固定した軌道更新を交互に行う。 |
| `[optimizer] orbital_update` | `"bfgs"` | 交互最適化の軌道更新。`"wavefunction"` は波動関数計量で前処理し、`"single_orbital"` は一軌道ずつ一般化固有値問題を解く。 |
| `[optimizer] spin_solver` | `"canonical"` | 交互最適化のスピン求解。`"projected"` は実際の波動関数の保持部分空間で、中心化したHamiltonianを解く。 |
| `[optimizer] orbital_micro_iterations` | `0` | 軌道更新の途中でスピンを解き直す間隔。BFGS系では採用ステップ数、一軌道更新では固有値求解の試行回数で数える。`0` は全軌道ブロック終了後に求解する。 |
| `[optimizer] joint_correction` | `false` | 交互更新の末尾に、軌道・スピンの同時補正を追加する。 |
| `[optimizer] joint_correction_start` | `1` | 同時補正を追加し始めるスイープ番号。交互最適化もその後引き続き実施する。 |

初期スピン係数を固有値問題から決め、交互最適化する設定例です。
分子・基底関数・活性空間は使用する入力ファイルで指定します。

```toml
[calculation]
energy_evaluation = "determinant"
initial_guess = "hf"
initial_spin_guess = "eigen"
initial_spin_overlap_cutoff = 1e-10

[optimizer]
method = "alternating"
orbital_update = "single_orbital"
spin_solver = "projected"
orbital_micro_iterations = 3
max_inner_iterations = 12
joint_correction = false
```

この例は一軌道更新を3回試行するごとにスピン係数を解き直し、1サイクルの軌道試行を
最大12回に制限します。`initial_spin_guess = "eigen"` は従来の `simultaneous` にも使えます。
`initial_json` による再開では、`"uniform"` は保存された `Cs` を保持し、
`"eigen"` を明示すると読み込んだ軌道上で係数を解き直します。

`wavefunction`、`single_orbital`、`projected` と同時補正はdeterminant評価を必要とします。
同時補正にはさらに `method = "alternating"` と `spin_solver = "projected"` が必要です。
詳細な制約、CLI/Python API、ノルム保護と収束判定は
[交互最適化の説明](INPUT_FILE.md#同時最適化と交互最適化)と
[スピン初期値・同時補正の開始時期](INPUT_FILE.md#スピン初期値と同時補正の開始時期を独立に指定する)を参照してください。

残差方向を追加する `CoupledOptimization.adaptive_correction` と、軌道だけを減衰する
`SpinResponseCorrection.response_correction` は、明示的に呼ぶ実験用APIです。
通常の `joint_correction = true` は従来の最大6方向の補正を使い、これらの実験方式へ自動では切り替わりません。
保存状態から同時補正だけを比較する診断スクリプトは [HPC_README.md](HPC_README.md#保存状態から同時補正だけを比較する)に記載しています。
Fe(CO)5の試行では収束が得られておらず、同梱の
[段階的補正の入力例](examples/inputs/fe_co5_staged_joint.toml)は比較用の設定です。

## Fixed inactive orbitals / active-space SCGVB

The existing simultaneous optimizer remains the default. Use
`--optimization-method alternating` (or `[optimizer] method = "alternating"`
in TOML) to solve spin coefficients at fixed orbitals and alternate with
bounded orbital updates with feasible backtracking. Both alternating blocks
share a relative wavefunction-norm guard based on the accepted state; it is
separate from the spin-metric eigencutoff. Both methods require the same full physical-gradient and
normalization checks. See [INPUT_FILE.md](INPUT_FILE.md) for the spin-overlap
cutoff, iteration limits, and diagnostic history.

Add `--orbital-micro-iterations 5` (or `orbital_micro_iterations = 5` under
`[optimizer]`) to re-solve spin coefficients after every short orbital block.
These blocks share one norm floor and one total orbital-step budget per outer
cycle; BFGS restarts when the spin objective changes. The default `0` retains
the original full orbital block. Intermediate spin acceptance fractions and
residuals are recorded in the micro-update history.

Add `--optimize-core` to relax the selected inactive orbitals while keeping them
doubly occupied. This alternates SCGVB optimization with orbital rotations;
it does not turn the calculation into FCI. Without this flag, inactive orbitals
remain fixed. See [ACTIVE_SPACE.md](ACTIVE_SPACE.md) for examples, tolerances,
saved diagnostics and the CPU/GPU division of work.

Full SCGVB remains the default. Fixed closed shells, manual HF MO selection,
a benzene 6-pi-electron preset, and orientation-independent planar hydrocarbon
`--active-space pi` selection (including COT with 8 pi electrons) are available. The determinant evaluator
works in NumPy and JAX without constructing electron-permutation tables.
See [ACTIVE_SPACE.md](ACTIVE_SPACE.md) for CH4/benzene commands, limitations, and validation.

## Single-GPU 12-pi-electron distribution

The determinant backend now supports 12e/12o singlets using bounded intermediate
arrays and a custom reverse-mode derivative. See [HPC_README.md](HPC_README.md)
for GB200/Linux aarch64 setup, validation, benchmarks and run scripts.
Build the portable source archive with `python build_hpc_bundle.py`.

Twisted sp2 hydrocarbons such as biphenyl can use `--active-space pi
--pi-normal-mode local --pi-electrons 12`. The included `examples/biphenyl_45deg.xyz`
is an idealized test geometry, not an optimized structure. See `ACTIVE_SPACE.md`.

### Cyclododecahexaene: 12-pi-electron ring benchmark

`examples/cyclododecahexaene.xyz` contains C12H12 ([12]annulene), with carbon
atoms 1--12 followed by hydrogens 13--24. This is an **idealized planar test
geometry, not an optimized equilibrium structure**. Ring C--C distances alternate
between 1.34 and 1.46 Angstrom; C--H distances are 1.08 Angstrom. The neutral
singlet input selects 12 pi electrons and uses cc-pVDZ, the JAX determinant
backend, and a GPU. Initially the 36 doubly occupied inactive orbitals are fixed.

```bash
python calc_molecule.py examples/inputs/cyclododecahexaene.toml --check
python calc_molecule.py examples/inputs/cyclododecahexaene.toml
```

Use `--device cpu` to override the device or `--optimize-core` to relax the
inactive/active spaces (additional CPU orbital-optimization cost). The check
performs HF and validates the active-space size; it does not establish SCGVB
convergence. For a nonplanar geometry, select local pi normals and validate its
coordination with `--check` before optimization.

14電子・14軌道の一重項を扱う場合は、行列式評価器と
`--determinant-size-profile large14` を明示的に指定する。既定は従来の `standard`。
TOML・Python API・再開・非活性軌道緩和の設定は [ACTIVE_SPACE.md](ACTIVE_SPACE.md) を参照。
フェナジンの通常コード用入力例は `examples/inputs/phenazine14.toml`。
