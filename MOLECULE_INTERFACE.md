# 多原子分子の非直交SCWF単一点計算

`calc_molecule.py` は一般の分子座標を入力し、軌道係数とスピン係数を最適化する。
核座標の構造最適化や距離走査は行わない。Full CIは実行しない。SCFは `--initial-guess hf` による初期軌道作成、または固定閉殻・活性空間の選択時に実行する。
NumPy/JAXの両方で、既存のAugmented Lagrangianと停滞時の補助処理を使用する。

## XYZ入力

```bash
# H3+：三原子・2電子・一重項、JAX CPU
python calc_molecule.py --xyz examples/h3_plus.xyz --charge 1 --multiplicity 1 \
    --basis sto-3g --backend jax --device cpu --output h3_plus.json

# H4：四原子・4電子・一重項、既存NumPy版
python calc_molecule.py --xyz examples/h4.xyz --output h4.json
```

XYZは原子数、コメント行、`元素 x y z` の原子行からなる1フレーム形式。
電荷・多重度はXYZのコメントから読み取らず、コマンドで明示する。
座標単位の既定値はAngstrom。Bohr座標なら `--unit Bohr` を指定する。

## 座標の直接指定

```bash
python calc_molecule.py --atom 'H 0 0 0; H 0.9 0 0; H 0.45 0.7794228634 0' \
    --charge 1 --multiplicity 1 --output h3_plus.json
```

`--multiplicity` は **2S+1**。PySCFの `mol.spin=2S` とは1だけ異なる。
一重項=1、二重項=2、三重項=3。電子数との整合性を検査する。
既定値は電荷0、多重度1、基底sto-3g、NumPy、乱数seed 40。
JAX版は `--backend jax`、NVIDIA環境でGPUを明示するなら `--device gpu`。
`--device` はJAX版だけで使用できる。

## 計算規模の事前確認

```bash
python calc_molecule.py --xyz examples/h4.xyz --check
```

full版の `--check` は積分・最適化を実行せず、電子数、AO数、スピン関数数、
配置ペア数、行列式数、密なスピン行列だけの保存バイト数を表示する。
活性空間を指定した場合は選択のためHFを実行するが、SCGVB最適化は行わない。

既定の `permutation` 方式は `(N!)²` 個の配置ペアを使い、100,000ペアを超える
計算を停止する。`--max-permutation-pairs` で変更できるが、計算量は改善されない。
`--energy-evaluation determinant` は置換表を生成せず、別の行列式空間サイズ制限を使う。
固定閉殻軌道と活性空間にも対応した。CH4・ベンゼンの実行例と制約は
[ACTIVE_SPACE.md](ACTIVE_SPACE.md) を参照。活性電子は2以上が必要。

## 出力

`--output` のJSON（既定 `scwf_result.json`）に以下を保存する。

- 入力構造（出力はAngstromに統一）、電荷、多重度、基底、計算設定。
- 全エネルギー・核反発エネルギー、電子エネルギーの分子 `Ee`（単位Hartree）。
- 波動関数ノルム `Delta`、規格化誤差、`Ee/Delta` の最大絶対勾配。
- `orbital_coefficients`：行がAO、列が非直交軌道の係数行列C。
- `spin_coefficients`：スピン基底の係数Cs。
- `ao_labels`、AO重なりS、軌道重なり `C.T @ S @ C`、スピン分岐図。
- 最適化中に発生した警告。弱い停留条件で返った場合の警告も保存する。

活性軌道同士の直交性は課さない。非活性軌道を指定した場合は活性・非活性間を直交させる。スピン係数の二乗を一般の状態確率とみなす処理も
行わない。最適化失敗時は終了コード1で終了し、新しい結果を書き込まない。
結果ファイルが既にある場合、成功時には上書きする。
局所解・収束経路は初期値に依存する。`--seed` で変更できる。
反復上限と規格化許容誤差は `--max-outer-iterations`、
`--max-inner-iterations`、`--epsilon` で設定する。

`--verbose` では、AL内部の制約誤差に加え、規格化後の候補の
`constraint` と `physical_gradient`（Ee/Delta の勾配の最大絶対値）を表示する。
補助的なRayleigh商最適化でエネルギーが改善し規格化条件を満たした候補は、
勾配が未収束でも次の反復の初期値として引き継ぐ。
ペナルティ係数は制約誤差の改善が不十分な場合にのみ増加させる。
改善候補がなく、制約誤差の5%以上の減少が5回連続で見られない場合、
または係数が1e8に達した場合は、診断情報付きで停止する。
この停止は収束を意味しない。反復上限だけを増やしても解除されない。
規格化誤差が小さくても物理的な勾配が大きければ、許容誤差を緩めるだけで
エネルギーの収束を保証することはできない。

## Python API

```python
from calc_molecule import calculate, read_xyz

result = calculate(read_xyz('examples/h3_plus.xyz'), charge=1, multiplicity=1,
                   backend='jax', device='cpu')
print(result['total_energy_hartree'])
print(result['orbital_overlap'])
```

既存の低水準クラスを直接使う場合は
`SCWF_Energy(mol, compute_fci=False)` とする。JAX版は `device='cpu'` も指定できる。
低水準クラスの `compute_fci=True` という既定値は、従来の二原子比較スクリプトと
独立検証テストの互換性のためだけに残している。今回の汎用インターフェースは
常にFalseを指定し、FCI実行オプションも設けていない。

## 動作確認

Mac CPU、STO-3Gで、H3+三角形は約 `-1.255812305378 Eh`、
1 Å間隔の直線H4は約 `-2.164905132939 Eh`（JAX版）となった。
H3+はNumPy版との一致も確認した。これらは指定形状・初期値での結果。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_molecule_interface -v
```

## Li2などで勾配が停滞する場合

`unit_orbital_norm` は各非直交軌道とスピン係数を個別に規格化したときの
波動関数ノルム。全体の拡大で隠れた打ち消し合いを確認する指標であり、
収束判定の許容誤差ではない。非常に小さければ係数の表現が数値的に
不安定になりうる。`normalized_constraint` だけで収束とは判定できない。

NumPy/JAXとも `--energy-evaluation determinant` を指定できる。
同じ非直交波動関数を行列式展開で評価し、Full CI最適化は行わない。
小さい行列式空間のみ対応する実験的オプションで、Li2の収束を保証しない。
既定値は従来の `permutation`。詳細は `JAX_BACKEND.md` を参照。

## 物理的な初期軌道

`--initial-guess hf` は閉殻でRHF、開殻でROHFの占有軌道を使う。
SCFが収束しなければ停止し、未収束の軌道を黙って採用しない。
`--initial-guess core` は一電子ハミルトニアンの一般化固有値問題を解き、
低い準位から占有する。電子間反発を考慮したSCFは行わない。
従来の乱数初期値は `--initial-guess random`（互換性のため既定値）。

二重占有MOを連続する2本の軌道に割り当て、続いて一重占有MOを配置する。
Csは従来どおり等しい初期値を使う。HF/coreでは、同一の対軌道から始める
対称性を崩すため、S内積でノルム0.02の小さな乱数摂動を各軌道に加え、
各軌道を規格化する。`--seed` で再現できる。
`--initial-perturbation 0` で摂動を無効にできるが、完全に同一の対軌道からの
出発では対軌道が分かれる方向へ動きにくい場合がある。
最適化中にMOの占有や軌道間の直交性を拘束するものではない。

```bash
python calc_molecule.py --xyz examples/li2.xyz --backend jax --device cpu \
  --max-permutation-pairs 600000 --initial-guess hf --verbose
```

Python APIでは `calculate(..., initial_guess='hf', initial_perturbation=0.02)`。
軌道のみ作る場合は `from InitialOrbitals import initial_orbitals` として
`C, metadata = initial_orbitals(mol, method='hf')` を使う。
初期化の設定は結果JSONの `calculation.initialization` に保存する。
HFは初期値の改善候補であり、SCWFの収束や大域的な最小値を保証しない。

## Mulliken・Löwdin・meta-Löwdin電荷とスピン分布

計算終了時に全原子のMulliken・Löwdin・meta-Löwdin電荷とスピン分布を
比較表で表示する。追加の入力設定は不要。結果JSONは以下の欄に保存する。

- `population_analysis.methods.mulliken`
- `population_analysis.methods.lowdin`
- `population_analysis.methods.meta_lowdin`

互換性のため `population_analysis` 直下の従来の値はMullikenのまま残す。
各方式の `atoms` 内の `index` は1始まり、`charge` は単位eの原子電荷、
`electron_population` は電子数の割り当て、`spin_population` はα−β電子数。

SCGVBの正規化した波動関数から1粒子密度行列を求め、二重占有の非活性軌道を
α・βに1電子ずつ加える。core最適化時には最終core係数を使う。
単純な軌道係数の二乗和や、独立したHF計算の電荷ではない。
AO密度DとAO重なりSから `diag(D S)` を原子ごとに集計し、核電荷から引く。
電子数・スピンの保存を検査し、原子電荷の合計は分子電荷、スピン分布の合計は
`2 M_S = multiplicity - 1` に対応する。

既存の結果ファイルを、再最適化せず解析する場合：

```bash
python PopulationAnalysis.py allyl_active.json --output allyl_population.json
```

出力を省略すると元の名前に `_population.json` を付けて保存し、元の計算結果を残す。
解析時はSCGVB軌道が張る空間で行列式展開するため、全AO空間の大きなCI展開は不要。
軌道方向を切り捨てず、線形従属の軌道にも対応する。12活性電子の解析も検証対象。
Mulliken電荷・スピン分布は基底関数や分割法に依存し、原子の酸化数そのものではない。

Löwdinは原子の事前直交化を行わない対称直交化を用い、`diag(S^(1/2) D S^(1/2))`
を原子ごとに集計する。meta-LöwdinはPySCFの `orth_ao` を使用し、事前直交化は
`ANO` と明示している。core・価電子・その他のAO群を区別して直交化し、
`T = X.T @ S` により `diag(T D T.T)` を集計する。このAO群分けは計算の
SCGVB活性／非活性軌道の選択とは別の概念である。
3方式は同じ最終SCGVB密度を使い、最適化やエネルギーを変更しない。
全方式で電荷・スピンの総和を検証する。基底関数への依存性は残る。

### 電荷解析の旧版出力について

`analysis_version = 2` では、Cholesky分解の未使用三角部分を明示的にゼロ化し、
重なり行列とSCGVB軌道の再構成を検査する。一部の実行環境で、旧版はこの未使用部分を
密度変換に混入させ、電子数・スピンの総和は合うものの原子分布が誤る問題があった。
Mulliken・Löwdin・meta-Löwdinの全方式に影響する。
既存の波動関数係数・最適化エネルギーには影響しないため、該当する旧版出力は
`PopulationAnalysis.py` で後処理を再実行する。対称な原子への電荷平均などは行わず、
元のSCGVB波動関数から密度を正しく再構成する修正である。
