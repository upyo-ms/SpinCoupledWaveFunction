# 閉殻軌道と活性空間を用いたSCGVB

従来のfull SCGVBは既定動作のまま残している。オプションを指定した場合だけ、
二重占有の非活性軌道を固定し、残りの電子をSCGVBで扱う。
`--optimize-core` を追加すると、二重占有を保ちながら非活性軌道も最適化する。
構造スキャンでの軌道追跡、自動的な精度保証は含まない。

## 二重占有を保ったcore軌道の最適化

```bash
python calc_molecule.py --atom 'Li 0 0 0; H 0 0 1.6' --basis cc-pvdz \
  --frozen-core 1 --optimize-core --initial-guess hf \
  --backend jax --device cpu --energy-evaluation determinant \
  --output lih_relaxed_core.json
```

`--frozen-core` / `--inactive-mos` は閉殻軌道の初期選択に用いる。
`--optimize-core` 指定時は、選択した閉殻軌道をすべて緩和する。
指定しない場合はこれまでどおり固定する。非活性軌道がない場合はエラーとなる。
`--active-space pi --optimize-core` も使用可能で、その場合は1sとσ軌道を含む
全非活性閉殻が緩和対象となる。活性空間の次元と電子数は変えないが、活性空間自体も
回転するため、初期π射影空間に厳密に固定した計算ではない。

`CoreOptimization.py` はPySCFの二段階MCSCF軌道最適化を使う。
内側ではSCGVBの非直交軌道とCsのみを最適化し、その行列式展開から1・2粒子密度行列を
作る。FCI係数を独立に最適化したりFCI固有値問題を解いたりはしない。
外側では閉殻・活性・仮想空間の間を回転し、coreの二重占有とcore–active直交性を保つ。
活性SCGVB軌道同士には直交性を課さない。

`--core-max-iterations` は外側反復上限（既定50）、
`--core-gradient-tolerance` は軌道回転に対するエネルギー勾配の2ノルム許容値（既定1e-6）。
内側の物理勾配・規格化と外側の収束をすべて検査する。失敗時に収束値として返さない。
JSONの `core_optimization` に初期固定coreエネルギー、反復履歴、最終軌道勾配、
初期core係数、二重占有数を保存する。通常の軌道係数欄には最適化後の係数を保存する。

NumPy/JAXと両エネルギー評価器に対応する。JAXの指定デバイスは内側のSCGVB評価に適用し、
外側の積分変換・密度行列・軌道回転は現在CPUで行う。反復ごとのJAX評価器の再構築も必要。
活性空間の行列式サイズ制限は置換和を選択した場合も適用される。
12π系の軌道緩和の性能・収束は別途評価が必要で、固定core版と同じ速度を保証しない。

## 行列式サイズ設定：14電子対応

既定の `standard` は従来どおり各スピン6電子以下、主要配列200万要素以下。
`large14` は明示的に指定した場合だけ、最大14電子・14軌道、各スピン7電子以下、
主要配列1200万要素以下を許可する。無制限の解除ではなく、GPU空きメモリや
収束の保証でもない。14電子・14軌道の一重項は11,778,624行列式、429スピン関数。

```bash
python calc_molecule.py --input phenazine.toml --determinant-size-profile large14
# サイズ選択だけを確認
python calc_molecule.py --input phenazine.toml --determinant-size-profile large14 --check
```

TOMLでは `[calculation]` に `determinant_size_profile = "large14"` を指定する。
Pythonでは `calculate(..., determinant_size_profile='large14')` または
`SCWF_Energy(..., determinant_size_profile='large14')` を使う。NumPy/JAXに共通。
エネルギー方式は別設定なので、14電子では `energy_evaluation='determinant'` を使う。
置換和の上限はこのオプションでは解除しない。

選択した設定はコア緩和の内側・最終評価・電子密度解析にも引き継ぎ、結果JSONの
`calculation.determinant_size_profile` に保存する。再開計算では同じ設定を入力またはCLIで
指定する（再開JSONを読んでも既定の安全上限を自動的には引き上げない）。
保存JSONの電子密度再解析は保存された設定を使用し、古いJSONは `standard` として扱う。

rikyuのGB200 1基で14e/14oの独立エネルギー・勾配検算、フェナジンの固定非活性軌道と
全非活性軌道緩和の収束を確認した。ポリエンのエネルギー・勾配評価は約1.48秒、
GPUピーク実使用約3.22 GiB。ただしスピン・分子・基底・最適化方式が変わる場合の
速度やメモリは別途確認する。これらの実績は14電子一重項のものである。

## 実行例

プロジェクトのルートで実行する。PySCF、NumPy、SciPyが必要で、
`--backend jax` を選ぶ場合はJAXも必要。8活性電子の最適化にはJAXを推奨する。

```bash
# CH4：C 1sの1軌道（2電子）を固定、8電子を活性にする
python calc_molecule.py --xyz examples/ch4.xyz --frozen-core 1 \
  --energy-evaluation determinant --backend jax --device cpu \
  --initial-guess hf --output ch4_active.json

# ベンゼン：xy平面に置いた中性一重項、6π電子
python calc_molecule.py --xyz examples/benzene.xyz --active-space benzene-pi \
  --energy-evaluation determinant --backend jax --device cpu \
  --initial-guess hf --output benzene_pi.json

# 最適化前に選択結果とサイズを確認（活性空間の選択にはHFを実行）
python calc_molecule.py --xyz examples/benzene.xyz --active-space benzene-pi \
  --energy-evaluation determinant --check

# 小さい活性空間はNumPyでも計算できる
python calc_molecule.py --atom 'Li 0 0 0; H 0 0 1.6' --frozen-core 1 \
  --energy-evaluation determinant --initial-guess hf --output lih_active.json

# 既存のfull SCGVB：既定動作は変更していない
python calc_molecule.py --xyz examples/h4.xyz --output h4_full.json
```

## 一般の平面π系：COTの8π電子

`--active-space pi` はベンゼンに限定せず、平面炭化水素の炭素2p軌道を目印に
活性空間を選ぶ。分子面の法線を座標から求め、各炭素のMINAO 2px/2py/2pzを
法線方向に合成するため、分子をxy平面へ置き直す必要はない。
占有・空軌道の両部分空間に対するAVAS型の射影を対角化し、固有値0.2以上を採用する。
ROHFの単占有軌道はすべてπ空間に属する必要がある。

```bash
python calc_molecule.py --xyz examples/COT.xyz \
  --active-space pi --pi-electrons 8 \
  --energy-evaluation determinant --backend jax --device cpu \
  --initial-guess hf --output cot_pi.json
```

最適化せず確認する場合は同じ指定に `--check` を付ける。
Pythonでは `calculate(..., active_space='pi', pi_electrons=8, ...)`。
`--pi-electrons 8` は選ばれた電子数が8であることの検査で、省略時は射影から決定する。
電子数を強制的に変更する機能ではなく、不一致なら停止する。

提供された `examples/COT.xyz` では、56電子のうち8電子・8軌道が活性となり、
残り48電子を24本の固定閉殻軌道にまとめる。一重項スピン関数は14個。
STO-3G/JAX CPU/seed=40/HF初期値で `-303.880757161246 Eh` に収束した。
最大絶対勾配は1.28e-8、ノルム誤差は8.89e-16未満。
同じ固定閉殻のCASCIは `-303.884874619423 Eh`、HFは `-303.674298594311 Eh`。
計算結果と独立した参照値は `examples/active_space/cot_pi_*.json` に保存している。
これは提供された平面構造での単一点計算であり、構造最適化や大域的最小の保証は行わない。

現在の自動 `pi` 選択はC/Hのみ、非直線、全原子の面からの最大偏差1e-4 Å以下が対象。
各炭素から1本のπターゲットを作り、選択された軌道数が炭素数と一致することを検査する。
平面性と射影は化学的妥当性の完全な判定ではないため、飽和炭素が混在する系などでは
自動選択をそのまま適用せず、軌道を確認して手動空間を指定する。
非平面のsp2炭素系には下記のlocalモードを用いる。ヘテロ原子のπ寄与、部分的なπ原子指定は未対応。
`pi` と手動MO指定・`--frozen-core` は併用できない。
既存の `benzene-pi` は互換性のため、従来のxy平面・C6H6限定のまま残している。

## 非平面のπ系：ビフェニル

`--pi-normal-mode local` を指定すると、各炭素の3本のσ結合の方向から局所的な面の
法線を求め、その方向にMINAOの2px/2py/2pzを合成してπターゲットを作る。
分子全体が平面である必要はない。AVAS型射影、固定閉殻の有効積分、SCGVB評価器は共通。
2つの芳香環を独立に解くのではなく、環間相互作用を含む12電子の問題として解く。
非平面ではπ/σ分離は一般に厳密な対称性による分離ではなく、選択したモデル空間の近似になる。

```bash
python calc_molecule.py --xyz examples/biphenyl_45deg.xyz \
  --active-space pi --pi-normal-mode local --pi-electrons 12 \
  --energy-evaluation determinant --backend jax --device cpu \
  --initial-guess hf --output biphenyl_result.json
```

GPU環境では `--device gpu` に切り替える。Python APIでは `pi_normal_mode='local'`。
デフォルトの `global` は従来どおり分子全体の平面を使い、非平面構造は停止する。

付属のビフェニルはC-C=1.397 Åの正六角形2環をC-C=1.49 Åで結び、環間ねじれを45°に
した検証用構造であり、構造最適化済みではない。STO-3Gでは12電子・12軌道、
35本の固定閉殻軌道を選択できること、エネルギーと方向微分が独立なNumPy/PySCF評価と
一致することを確認した。この例のSCGVB最適化の収束はまだ確認していない。

localモードの結合推定はC-C <= 1.85 Å / C-H <= 1.30 Åの幾何学的判定を使う。
各炭素はちょうど3近接原子を持つ必要があり、局所平面からのσ結合の傾きが10°を
超える場合は停止する。sp3炭素や強いピラミッド化、結合切断付近を自動的にπ系とみなさない。
結合相手、局所法線、面からの傾き、距離閾値を結果JSONに保存する。
立体的に混み合った構造などで距離判定が不適切な場合は手動MO選択を検討する。

## 選択方法と制約

- `--frozen-core N`：HFの低エネルギー側からN本の空間軌道を二重占有・固定する。
  電子数ではなく軌道数。すべて二重占有であることを検査する。
- `--inactive-mos 1 2 ...`：固定するHF軌道を指定する。CLIは1始まり。
  `--frozen-core` と併用しない。
- `--active-mos 2 3 4 ...`：活性SCGVB軌道を展開するHF部分空間を指定する。
  占有軌道だけでなく、結合の変化を表現する反結合性・空軌道も含める。
  省略時は非活性軌道の直交補空間全体を用いる。
- すべてのHF占有軌道は非活性または活性に含める必要がある。
  重複、範囲外、活性と非活性の重なり、2未満の活性電子数はエラーになる。
- `--active-space benzene-pi`：PySCF AVASの `C 2pz`、閾値0.2で6電子・6軌道を選ぶ。
  中性一重項C6H6かつ分子がxy平面にある場合に限定する。
  6e/6oが得られない場合は停止する。内殻6軌道とσ占有12軌道を固定し、
  活性π空間の外への軌道変形は許さない。手動選択オプションとは併用しない。

MO番号はHFの計算結果に依存する。縮退や構造変化をまたいで同じ軌道を表すとは
限らない。現在の手動選択はcanonical HF軌道に対する指定で、局在化処理や追跡は
行わない。初期値の `hf` / `core` / `random` は活性空間内でのみ生成する。

## 共通エンジン

`ActiveSpace.py` が固定軌道 `inactive` とS-直交規格化した基底 `basis` を生成する。
最適化変数Xと出力のAO係数Cの関係は `C = basis @ X`。
これにより非活性軌道との直交性を常に保ち、活性軌道同士の非直交性は保たれる。
full版では従来どおりAO係数を直接最適化する。

閉殻密度をDとし、PySCFのJ/Kを用いて `V = J[D] - K[D]/2` を計算する。

- 有効一電子積分：`basis.T @ (h + V) @ basis`
- 二電子積分：活性基底へ変換した通常の電子反発積分
- 閉殻定数：`Tr[D h] + Tr[D V]/2`
- 全エネルギー：`E_nuc + E_inactive + numerator / norm`

スピン関数、軌道初期化、規格化、拡張Lagrangian最適化、Rayleigh商の勾配・精密化は
full版と共通。`NonorthgonalEnergy.py` と `NonorthgonalEnergyJAX.py` は両方の問題を扱う。
`calcEnergy` は活性部分の分子とノルムを返す。閉殻定数は最適化対象の分子に含めず、
最終全エネルギーに加える。

`--energy-evaluation determinant` は全電子置換テーブルを一切生成しない。
既存のJAX exterior-product評価器を拡張し、NumPy版を追加した。
SCGVBの非直交軌道を直交基底上の行列式に展開し、Hamiltonianを作用させる。
行列式係数は軌道とスピン係数から決まり、独立な最適化変数ではない。
NumPy版はPySCFのFCI用Hamiltonian収縮ルーチンを使うが、FCI固有値問題は解かない。
非直交行列式ペアの一般化Slater–Condon評価器とは別の方式である。

計算量は基底の行列式空間にも依存するため、無制限には拡張できない。
既定の `standard` ではNumPy/JAXとも各スピンの電子数は6以下とし、行列式係数配列、各次数の小行列式配列、
primitiveスピン係数配列、変換後の二電子積分配列にそれぞれ200万要素の上限を設ける。
12電子・12軌道の一重項（853,776行列式、132スピン関数）に対応する。
JAX版では小行列式の多項式再帰とcheckpoint、一体演算子の二乗和によるHamiltonian作用、
専用VJPにより、軌道数²×行列式数の中間配列とその逆伝播記録を避ける。
詳細と持ち込み用実行例は [HPC_README.md](HPC_README.md) を参照。
上限を超える場合は積分・スピン展開の生成前に停止する
（HFによる空間選択はその前に実行される）。

STO-3GのCH4ではn=8、8電子、一重項スピン関数14個、基底行列式4,900個。
ベンゼンπ系ではn=6、6電子、スピン関数5個、基底行列式400個。
これらはCASSCFと同じ変分空間を使うという意味ではない。
旧 `permutation` 評価器も比較用に残り、サイズ上限を満たす活性空間にも使える。
`--max-permutation-pairs` はこの旧方式だけに適用する。

## Python API

```python
from calc_molecule import calculate

result = calculate('Li 0 0 0; H 0 0 1.6',
                   inactive_mos=[0],       # Pythonは0始まり
                   active_mos=[1, 2, 3, 4, 5],
                   energy_evaluation='determinant', initial_guess='hf')
```

低レベルAPIでは `prepare_active_space(mol, ...)` の戻り値を
`SCWF_Energy(mol, compute_fci=False, active_space=space,
energy_evaluation='determinant')` に渡す。`space=None` はfull版。
活性空間指定時は `compute_fci=False` が必要。full FCIは固定閉殻近似と同じ
変分問題ではないため、参照値として暗黙には実行しない。

## 出力と軌道表示

既存schema version 1にフィールドを追加している。
`orbital_coefficients` は活性SCGVB軌道のAO係数（AO数×活性電子数）、
`inactive_orbital_coefficients` は固定閉殻軌道を1本ずつ保存する。
`optimization_basis_coefficients` と `optimization_orbital_coefficients` で
最適化に用いた部分空間と変数も保存する。`active_space` に選択根拠・HF参照値が入る。

`electronic_numerator_hartree` は有効Hamiltonianの活性部分、
`inactive_energy_hartree` は閉殻定数、`nuclear_energy_hartree` は核反発だけ。
`wavefunction_norm` と勾配は、固定閉殻を除いた活性SCGVBの規約による。

```bash
python generate_orbitals.py benzene_pi.json --output-dir benzene_orbitals
```

活性軌道を先に、続いて非活性軌道を1本ずつ出力する。
HTMLとmanifestで非活性・二重占有軌道を区別する。
古いfull版の結果ファイルも引き続き読み込める。

## 検証

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -p 'test_*.py' -v
```

旧置換評価とのエネルギー・ノルム・複素ステップ勾配の一致、JAX自動微分との一致、
特異な部分小行列での勾配、固定閉殻を含むfull SCGVBとの一致、
PySCF CASCIの有効一電子積分・閉殻定数との一致をテストする。
MO指定の検証、サイズ制限、最適化からJSON/Cube出力までの経路も対象。

付属の構造・STO-3G・HF初期値・seed=40・JAX CPUによる実測：

| 系 | HF / Eh | 固定閉殻SCGVB / Eh | 同じ固定閉殻のCASCI / Eh | 最大絶対勾配 |
|---|---:|---:|---:|---:|
| CH4、8活性電子 | -39.726724240941 | -39.794445033744 | -39.805558524322 | 1.04e-7 |
| C6H6、6π電子 | -227.890600566936 | -227.989954186596 | -227.997024016905 | 5.49e-9 |

両者ともノルム誤差は6e-16以下。SCGVBエネルギーはHF以下・同じ空間のCASCI以上に
あることを確認した。CASCIは今回の検証で別途実行したもので、通常の計算には含まれない。
局所最適化の結果であり、大域的最小や他の構造・基底での収束を保証しない。
結果と参照値は `examples/active_space/` に保存している。

## Explicit atomic-orbital targets

`active_space='ao'` supports MINAO target rows such as `orbitals=['1 2px 2py A',
'1 1s F']`. Atom indices are one-based; A selects active orbitals, I selects
relaxable doubly occupied orbitals, and F selects permanently frozen doubly
occupied orbitals. F columns lead the inactive basis and are excluded from
PySCF orbital rotations even with core optimization enabled. See INPUT_FILE.md
for selection defaults, projection thresholds, validation, and an executable
`examples/inputs/allyl_ao.toml` input.
