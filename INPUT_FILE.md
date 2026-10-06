# 計算入力ファイル

`calc_molecule.py` はTOML形式の入力に対応する。コメントは `#`、真偽値は
`true` / `false`、文字列は引用符で囲む。既存のコマンドライン指定も使用できる。

初期値・最適化の説明は次の箇所にあります。

- [同時最適化と交互最適化](#同時最適化と交互最適化)：BFGS、波動関数計量、一軌道求解、スピン更新間隔、ノルム保護。
- [スピン初期値と同時補正の開始時期](#スピン初期値と同時補正の開始時期を独立に指定する)：独立した初期スピン求解、再開時の扱い、補正開始番号。
- [HPCでの実験用同時補正](HPC_README.md#保存状態から同時補正だけを比較する)：残差方向の追加とスピン応答付き補正の診断API。

```bash
python calc_molecule.py examples/inputs/allyl.toml
# 同じ意味
python calc_molecule.py --input examples/inputs/allyl.toml
# 活性空間と計算サイズだけ確認（HFによる選択は実行する）
python calc_molecule.py examples/inputs/allyl.toml --check
# 入力は共通にして、実行環境と出力先だけ変更
python calc_molecule.py examples/inputs/allyl.toml --device gpu --output allyl_gpu.json
```

## アリルラジカルの入力例

```toml
[molecule]
xyz = "../allyl.xyz"
basis = "cc-pvdz"
charge = 0
multiplicity = 2

[calculation]
backend = "jax"
device = "cpu"
energy_evaluation = "determinant"
initial_guess = "hf"

[active_space]
mode = "pi"
pi_electrons = 3
optimize_core = true

[output]
path = "allyl_active.json"
```

この例は `examples/inputs/allyl.toml` にある。LiHのcore緩和は `lih.toml`、
非平面ビフェニルの12π・固定core・GPU計算は `biphenyl.toml` を参照。

XYZを使わず、入力内に座標を書く場合は `[molecule]` の `xyz` の代わりに次を指定する。
両方を同時に指定することはできない。

```toml
atom = """
Li 0.0 0.0 0.0
H  0.0 0.0 1.6
"""
unit = "Angstrom"
```

## 設定項目

省略した項目は従来のCLI既定値を使用する。キー名はアンダースコアを使う。

| セクション | 使用可能なキー |
|---|---|
| `molecule` | `atom` または `xyz`（必須）、`basis`、`charge`、`multiplicity`、`unit` |
| `calculation` | `backend`、`device`、`energy_evaluation`、`initial_guess`、`initial_json`、`seed`、`initial_perturbation`、`initial_spin_guess`、`initial_spin_overlap_cutoff` |
| `active_space` | `mode`、`pi_electrons`、`pi_normal_mode`、`frozen_core`、`inactive_mos`、`active_mos`、`optimize_core` |
| `optimizer` | `method`、`spin_overlap_cutoff`、`orbital_overlap_cutoff`、`orbital_micro_iterations`、`orbital_update`、`spin_solver`、`joint_correction`、`joint_correction_start`、`epsilon`、`max_outer_iterations`、`max_inner_iterations`、`core_max_iterations`、`core_gradient_tolerance`、`max_permutation_pairs` |
| `output` | `path`、`verbose` |

`active_space.mode` は `full` / `pi` / `benzene-pi`。
`inactive_mos = [1]`、`active_mos = [2, 3]` のようにMO番号はCLIと同じ **1始まり**。
計算条件の意味と組合せ制約は [ACTIVE_SPACE.md](ACTIVE_SPACE.md) と同じ。
`optimize_core = true` は選択した非活性軌道を二重占有のまま緩和する。

```toml
[optimizer]
max_outer_iterations = 50
max_inner_iterations = 1000
core_max_iterations = 50
core_gradient_tolerance = 1e-6
```

## 同時最適化と交互最適化

省略時は従来の同時最適化 (`simultaneous`) を使う。交互最適化は明示的に選ぶ。

```toml
[optimizer]
method = "alternating" # "simultaneous" は従来の方法
spin_overlap_cutoff = 1e-10
max_outer_iterations = 50
max_inner_iterations = 1000
```

CLIは `--optimization-method alternating --spin-overlap-cutoff 1e-10`、
Python APIは `calculate(..., optimization_method='alternating')`。
既存入力への追記や明示指定がなければ計算方法は変わらず、自動切替も行わない。
`--check` に方式と閾値を表示し、結果JSONの `calculation.optimization_method` に保存する。
Fe(CO)5用の入力例は `examples/inputs/fe_co5_alternating.toml`。

交互最適化は、固定した非直交軌道でスピン関数間のHamiltonianと重なり行列を作り、
一般化固有値問題でスピン係数を解く。その係数を固定して軌道をBFGSで更新し、
スピン係数を再び解く。軌道には個別の規格化だけを使い、相互直交性を課さない。
行列式係数はSCGVB軌道とスピン係数から生成し、独立なFCI変数にしない。
`max_outer_iterations` は交互更新の回数、`max_inner_iterations` は各軌道更新の採用ステップ数の上限。
`optimize_core=true` の場合も各閉殻緩和ステップの内側で指定方式を使う。

`orbital_micro_iterations = 5` を追加すると、各サイクルの軌道更新を最大5採用ステップずつに
区切り、区切るたびにスピン係数を解き直す。スピン係数が変わるためBFGSの履歴はリセットする。
`max_inner_iterations` はサイクル内の全区間を合わせた採用ステップ数の上限であり、
ノルム下限もサイクル内で共通にする。区切るたびに下限を下げることはしない。
更新できずスピン係数も変わらない場合は、そのサイクルの更新を終了して全勾配を確認する。
スピン更新の回数、採用割合、残差は結果JSONの `micro_history` と詳細ログに保存する。
省略時または `0` は従来の長い軌道更新を使う。この設定は `alternating` のみで使用できる。
CLIは `--orbital-micro-iterations 5`、Python APIは `orbital_micro_iterations=5`。

交互最適化には次の任意オプションもある。既定値は従来どおり
`orbital_update = "bfgs"`、`spin_solver = "canonical"`。

```toml
[optimizer]
method = "alternating"
orbital_micro_iterations = 10
orbital_update = "wavefunction"
spin_solver = "projected"
```

`wavefunction` は、規格化した多電子波動関数の微分から軌道計量を構成し、
各軌道ブロックの初期BFGS逆行列を前処理する。相対1e-8のダンピングは探索方向にのみ
使い、物理エネルギーや軌道の相互重なりを変更しない。1ステップの規格化波動関数の
変化は `sqrt(1 - |overlap|²) <= 0.05` とし、従来の軌道ステップ上限0.1、
エネルギー下降条件、ノルム下限も適用する。計量はスピン更新後の各ブロックで再構成する。

`projected` は、重なり行列の保持方向と現在の状態から実際の行列式展開を作り、
その波動関数の重なりで直交化する。基準エネルギーを引いたハミルトニアンをこの小空間で
評価し、大きい元の行列を構成した後の差し引きによる精度損失を減らす。
純粋なスピン固有関数の基底順序、SCGVBの変分空間、全物理勾配による収束条件は同じ。

どちらも `energy_evaluation = "determinant"` が必要。
大きい系（軌道変数が36を超える系）の `wavefunction` 更新はJAXバックエンドを使う。
CLIは `--orbital-update wavefunction --spin-solver projected`、Python APIも同名の引数。
診断ログには軌道試行の棄却理由（ノルム、非有限値、エネルギー、波動関数変化）も記録する。

`orbital_update = "single_orbital"` は、他の軌道とスピン係数を固定し、
一軌道の係数に線形なSCGVB波動関数のRayleigh商を一般化固有値問題で解く。
重なりはAO重なりではなく、残りの軌道とスピン係数を含む多電子波動関数の重なり。
`orbital_overlap_cutoff`（既定1e-10）で数値的な保持方向を選び、現在の状態も
一つの方向として必ず含める。実際の行列式ベクトルの計量で直交化し、H−Eを投影する。
行列式係数は軌道とスピン係数から導出し、独立な変分変数にはしない。

固有値解は更新候補であり、採用には元の物理エネルギーの厳密な下降、サイクル共通の
ノルム下限、S計量での軌道ステップ上限0.1、規格化波動関数の距離上限0.05を要求する。
必要なら候補との混合割合を半分ずつ縮める。非直交軌道を相互に直交化する処理はない。
既存の全物理勾配1e-6と規格化誤差による収束条件を維持する。

この方式では `max_inner_iterations` はサイクル内の一軌道固有値解の試行回数の上限で、
`orbital_iterations` はそのうち採用した更新の数。試行数、候補と採用割合、棄却理由、
行列構築・固有値解の時間は `orbital_geometry` に保存する。軌道番号の巡回は
スピン更新やサイクルをまたいで継続する。12軌道を一巡してスピンを更新する例：

```toml
[optimizer]
method = "alternating"
orbital_update = "single_orbital"
orbital_overlap_cutoff = 1e-10
spin_solver = "projected"
orbital_micro_iterations = 12
max_inner_iterations = 12
max_outer_iterations = 3
```

`single_orbital` はNumPy/JAXとも `energy_evaluation = "determinant"` が必要。
一軌道分の派生行列式ベクトルだけを構成し、全軌道分を同時に保持しない。
保持ランクが変わってもGPUを再コンパイルしないよう投影基底幅を固定し、ゼロ方向を
物理計量で除外する。既定の同時最適化、BFGS、波動関数計量BFGSは引き続き選択可能。

`spin_overlap_cutoff` はスピン重なり行列の最大固有値に対する相対閾値。
閾値以下の方向を除くため、結果が閾値に依存する可能性がある。
この固有値閾値は波動関数ノルムの保護下限には使わない。
各交互更新サイクルの開始状態で、軌道を個別に規格化した波動関数ノルムを計算し、
その半分をスピン・軌道更新に共通の保護下限とする。次のサイクルでは採用状態から再計算する。
軌道更新はS計量をEuclidean座標に変換し、各軌道のノルムだけを固定する。
各採用ステップでの軌道変化はS計量ノルムで最大0.1に制限し、
Armijo backtrackingで物理エネルギーの低下とノルム下限を確認する。
下限を割る試行は刻み幅を半減して再試行する。拒否点に勾配ゼロを返す処理は使わず、
無効な開始点はエラーにする。更新不能の場合も勾配が残る状態を収束扱いしない。
軌道更新で保持空間が変わり、スピン係数の解が更新前の状態より高いエネルギーになる場合は、
更新前の有効な状態を一つの方向として解く空間に追加する。
除外した全固有方向を復活させず、元の状態との変分的な比較を保つ。
スピン更新でも同じ下限と物理エネルギーの非増加を確認し、必要なら更新前の係数との間で
backtrackingする。採用可能な更新がなければ元の係数を保持し、全勾配による判定を続ける。
最終的には、除外方向も含めた全物理勾配が従来と同じ `1e-6` 以下、
波動関数ノルムの誤差が `epsilon` 未満であることを確認する。
エネルギーの停滞だけでは収束扱いにしない。

結果JSONの `alternating_optimization` に更新履歴、重なり固有値、保持したランク、
閾値と勾配を保存する。`spin_incoming_preserved` は更新前の状態を追加したか、
`spin_subspace_dimension` は追加後に実際に解いた空間の次元を示す。
`cycle_reference_norm` と `cycle_norm_floor` は両更新に共通のノルム保護を示す。
`orbital_max_accepted_step`、`orbital_rejected_trials`、`spin_step_fraction` に
更新幅とbacktrackingの結果を記録する。軌道状態コードは0=勾配許容値、
1=反復上限、2=下降ステップなし、3=ノルム保護によるステップ拒否を含む停止。
これは内側ブロックの状態であり、計算全体の収束には全勾配の判定が必要。
`verbose=2` の診断には `orbital_progress` を保存し、内側更新の進行も確認できる。
失敗時の診断snapshotには、最後の軌道更新後の状態を保存する。
閉殻緩和ありの場合は最終内側solveの履歴であり、
全solveの履歴が必要な場合は `output.verbose=2` の診断データも保存する。
NumPy/JAXの両backendと両エネルギー評価器に対応する。JAX determinant版は、
固定軌道の積分と小行列式を再利用し、スピン状態を一つずつ処理して大きな中間配列を避ける。
小さいスピン行列の固有値問題とBFGS軌道更新はCPUで実行する。

## パス・優先順位・出力

- 入力内のXYZと出力パスは、**入力ファイルの置かれたディレクトリが基準**。
- `output.path` を省略すると入力と同じ場所の同名 `.json` に保存する。
- 明示したCLIオプションが入力ファイルより優先する。CLIのパスは実行ディレクトリが基準。
- `--no-optimize-core` / `--no-verbose` で入力の `true` を無効化できる。
- 未知のセクション、未知のキー、型の不一致、不正な選択肢はエラーにする。
- JSONには入力ファイルの絶対パスと内容を `input_file` に保存する。
- TOMLはPython 3.11以降の標準ライブラリで読み込む。以前のPythonには
  `requirements-base.txt` の条件付き依存 `tomli` を使用する。

この入力方式は単一構造計算の `calc_molecule.py` が対象。
`calc_diatomic_mol.py` の距離走査設定は引き続きスクリプト内で編集する。

## 原子電荷の出力

計算終了時にMulliken・Löwdin・meta-Löwdin電荷とスピン分布を自動出力し、
JSONの `population_analysis.methods` に方式別に保存する。入力ファイルへの設定追加は不要。
既存JSONの後処理は [MOLECULE_INTERFACE.md](MOLECULE_INTERFACE.md) を参照。

## 原子軌道からのactive/inactive指定

`mode = "ao"` で、XYZの **1始まりの原子番号**、MINAO参照軌道名、役割を
空白で区切った行を `orbitals` に指定する。同一原子の複数軌道を1行で指定できる。

```toml
[active_space]
mode = "ao"
orbitals = [
    "1 2px 2py 2pz A", # 原子1の3つのp軌道をactiveの目標にする
    "1 1s F",         # 常に固定する二重占有空間
    "2 1s I",         # 二重占有空間（optimize_core=trueなら緩和可能）
]
ao_threshold = 0.2
# ao_electrons = 6    # 選択された活性電子数の検証。強制変更ではない。
optimize_core = true
```

これは書式例であり、軌道名と電子数は対象分子に合わせる。
実行可能なアリル入力は `examples/inputs/allyl_ao.toml`。

- `A`: active空間を選ぶ参照AO。
- `I`: 二重占有のinactive空間を選ぶ参照AO。`optimize_core=false`なら固定。
- `F`: 二重占有のfrozen空間を選ぶ参照AO。`optimize_core=true`でも固定。
- `A`の指定がある場合、未選択の二重占有HF軌道は `I` と同じ扱いになり、
  未選択の空軌道はexternalになる。`A`がない場合、`I/F`以外のHF軌道を全てactiveにする。
- 単占有HF軌道は全てactiveに含まれる必要がある。除外する指定はエラー。

AOそのものを占有させるのではなく、MINAO参照AOへの射影の固有値が
`ao_threshold`（0より大きく1より小さい値、既定0.2）以上のHF軌道の組合せを選ぶ。
`F`、`I`の順に二重占有空間から選び、残る空間で`A`を占有数ごとに選ぶ。
`I/F`の空軌道成分は選択しない。指定AO数と選択MO数は必ずしも同じにならない。
異なる役割が同じ選択空間を要求する場合は、射影の重なりを検出して終了する。
選択の固有値・AO指定・電子数・固定軌道数はJSONのactive-spaceメタデータに保存する。

`2pz`は **入力のCartesian z軸** の向き。分子の面法線を自動判定しない。
同じ元素でも原子ごとに指定できる。元素一括指定・軌道名のワイルドカードは扱わない。
軌道名はMINAOのラベル（例 `1s`, `2s`, `2px`, `2py`, `2pz`）に完全一致させる。
殻全体の省略指定も可能で、`"1 2p A"` は `"1 2px 2py 2pz A"` と同じ意味。
`3d`等も、その原子のMINAOに存在する全成分へ展開する。A/I/Fの全てで使用でき、
`"1 2s 2p A"` のように混在できる。`2p`と`2pz`の重複指定はエラーになる。
cc-pVDZ等の計算基底の追加の動径関数や分極関数を直接番号指定する機能ではない。
計算基底に関係なく参照MINAOに存在しない軌道、範囲外の原子、重複・役割矛盾は
HF開始前に理由を表示し、終了コード1で終了する。射影結果に関する矛盾はHF後に検出する。
`frozen_core`、`inactive_mos`、`active_mos`、πモードとは併用できない。

`optimize_core=true`ではF以外のinactive空間とactive空間を緩和するため、
最終軌道のAO組成は指定時から変わり得る。AO指定は初期空間の選択であり、
原子への局在やp軌道方向を最適化中に拘束するものではない。

## 保存済みJSONから初期軌道を引き継ぐ

```toml
[molecule]
xyz = "../thiophene.xyz"
basis = "cc-pvdz"
multiplicity = 1

[calculation]
backend = "jax"
device = "cpu"
energy_evaluation = "determinant"
initial_json = "thiophene_ao.json" # このTOMLからの相対パス

[active_space]
mode = "ao"
orbitals = ["1 2pz A", "2 2pz A", "3 2pz A", "4 2pz A", "5 3pz A"]
ao_electrons = 6
optimize_core = true

[output]
path = "thiophene_ccpvdz.json" # 読み込み元と別の名前にする
```

`initial_json`を指定すると、`initial_guess`と`initial_perturbation`に優先し、
保存されたSCGVB軌道・スピン結合係数・inactive軌道・active基底を読み込む。
同一原子順序、同一構造（Bohr座標差1e-7以内）、電荷、多重度、活性電子数、
inactive軌道数、F固定軌道数が必要。構造変更・原子並べ替えは未対応。

基底が異なる場合、AO間重なりから `S_new C_new = S_new,old C_old` を解き、
coreの直交規格化とcore–active直交性を復元する。F空間はI空間とは別に保持する。
active基底の次元が増えた場合は新しい選択空間から直交補空間を追加する。
小さい次元への縮小や射影でのランク欠落はエラー。Csは引き継ぎ、射影後の
波動関数ノルムを1に再規格化する。基底間射影自体は近似であり、元のエネルギーが
保存される保証も、最適化が必ず収束する保証もない。

読み込んだcore/active空間が初期空間となるため、新しい基底でHFから選び直した
空間とは異なる。`optimize_core=false`では射影後のcore/active空間を固定し、
その中でSCGVB係数のみを最適化する。空間そのものを緩和する場合はtrueを指定する。
CLIでも `--initial-json result.json` を利用でき、`--check` で入力と射影を検証できる。

## 結合次数の自動出力

通常計算と保存JSONの `PopulationAnalysis.py` による解析の両方で、電荷・スピン
解析に続いてMayer結合次数とWiberg–Löwdin結合指数を出力する。追加指定は不要。
正規化SCGVBの全一電子密度（inactive電子を含む）を共有し、再最適化しない。

`P=P_alpha+P_beta`, `R=P_alpha-P_beta`, `S=AO重なり行列`として、
原子A,BのAOブロック間の和から次を計算する。

- Mayer: `sum [(PS)_mu,nu (PS)_nu,mu + (RS)_mu,nu (RS)_nu,mu]`。
  開殻ではスピン密度項を含む。
- Wiberg–Löwdin: `sum (P_L)_mu,nu (P_L)_nu,mu`, `P_L=S^(1/2) P S^(1/2)`。
  全電子密度を用いる定義で、スピン密度項は追加しない。NBO/NAO版ではない。

定義は https://www.faccts.de/docs/orca/6.1/manual/contents/spectroscopyproperties/population.html
の式(5.12),(5.13)に対応する。相関波動関数の一電子密度に適用した解析指標であり、
二電子密度から得る厳密な電子対共有数ではない。基底・直交化・スピン状態に依存し、
整数の形式結合次数や結合エネルギーとは一致を要求しない。負値も切り捨てない。

JSONの `population_analysis.bond_orders` に全原子対の `pairs` と対称な `matrices`
（キー `mayer`, `wiberg_lowdin`）、定義を保存する。独立解析の出力では
`bond_orders` がトップレベルにある。原子番号は1始まりで、行列の行列順は入力原子順。
対角は0（原子内指数は未算出）。画面表示のみ、どちらかの指数の絶対値が0.05以上の
原子対に絞る。全原子対の値はJSONに残る。

## 最適化診断の出力レベル

`[output] verbose` は `false` / `true` または整数 `0` / `1` / `2` を指定できます。
`false` / `0` は進捗表示なし、`true` / `1` は従来の進捗表示、`2` は進捗表示に加えて
係数・積分・収束指標を保存します。CLIは `--verbose 2`（または `--verbose=2`）、
従来の `--verbose` はレベル1、`--no-verbose` はレベル0です。

```toml
[output]
path = "thiophene_ccpvdz_verbose.json"
verbose = 2
```

診断データは出力JSONと同じ場所の `<出力名の拡張子を除いた部分>_diagnostics/run-*/`
に保存します。実行ごとに新しいディレクトリを作り、過去の診断を上書きしません。
Python APIは `calculate(..., verbose=2, diagnostics_dir="diagnostics")` で指定でき、
省略時は作業ディレクトリの `scwf_diagnostics/run-*/` を使います。
レベル0/1では診断ファイルを作りません。

- `metadata.json`：PySCF分子情報（`molecule` は `pyscf.gto.loads` で復元）、
  AOラベル、活性電子数、スピン、評価器、実行デバイス。
- `reference.npz`：AO重なり行列とスピン結合のbranch diagrams。
- `solve-NNNN.npz`：各内側SCGVB計算の重なり行列 `S`、有効一電子積分 `h1`、
  二電子積分 `eri`（chemist順序）、AO上の活性基底 `basis`、非活性軌道 `inactive`、
  核反発と非活性エネルギーの和 `ecore`、外側の軌道最適化反復番号。
- `events.jsonl`：各行が一つのイベント。内側計算番号 `solve`、外側反復番号
  `macro_iteration`、内側拡張Lagrangian反復番号 `iteration`（該当時）、
  エネルギー分子、波動関数ノルム、全エネルギー、単位軌道ノルム、最大勾配、
  ペナルティ係数と乗数、エラーを記録します。
- `event-*.npz`：開始時、ペナルティ最適化後、物理エネルギー精密化後の候補、
  採用時、失敗時の最後に保存可能だった状態を記録します。
  生の変数 `x`、最適化基底の係数 `C`、規格化したスピン係数 `Cs`、
  AO係数 `ao_C`、軌道重なり行列、単位軌道重なり行列の固有値、物理勾配を保存します。
  軌道更新イベントは更新後の `basis` と `inactive` を保存します。
  更新前の状態は直前のsolve・acceptedイベントに対応します。

NPZは `numpy.load(path, allow_pickle=False)` で読み取れます。
全エネルギーは `numerator / norm + ecore` です。
失敗して結果JSONが作成されなくても、そこまでの診断は残ります。
診断は解析用で、`initial_json` に直接指定する再開ファイルではありません。
評価できなかった指標は `evaluation_error` を記録し、生の変数は保存します。
レベル2は追加のエネルギー・勾配評価とファイル保存を行うため、実行時間と容量が増えます。
最適化アルゴリズムと収束条件は変更しません。

内側最適化は、各固定Hamiltonianについて規格化可能な最良の物理状態を保持します。
比較するのはペナルティ目的関数ではなく `numerator / norm` です。
最良値より `1e-10 Eh` を超えて高い候補からは再開せず、保持した状態へ戻します。
改善のない復帰が5回続いた場合は未収束として停止し、収束条件は緩めません。
レベル2の `best` イベントは最良状態の更新、`best_restart` はその状態からの再開です。
失敗イベントにも保持した最良状態を保存します（未取得の場合は最後の状態）。
各内側計算で最良値は初期化され、異なる積分のエネルギーを比較しません。

Optional `[optimizer] joint_correction = true` adds a bounded correction of
orbital and spin coefficients together after each alternating cycle from
`joint_correction_start` onward (default 1). It requires
`method = "alternating"`, `spin_solver = "projected"`, and determinant energy
evaluation. The default is `false`. See `JAX_BACKEND.md` for the experimental
six-direction approximation, physical bounds and unchanged convergence tests.

## スピン初期値と同時補正の開始時期を独立に指定する

```toml
[calculation]
initial_guess = "hf"
initial_spin_guess = "eigen"
initial_spin_overlap_cutoff = 1e-12

[optimizer]
method = "alternating"
orbital_update = "single_orbital"
spin_solver = "projected"
spin_overlap_cutoff = 1e-12
orbital_micro_iterations = 3
max_inner_iterations = 12
max_outer_iterations = 50
joint_correction = true
joint_correction_start = 21
```

`initial_spin_guess = "eigen"` は、初期軌道を固定してスピンの一般化固有値問題を
解き、最適化へ渡すCsを決めます。軌道初期値の `initial_guess` と最適化方式の
`method` から独立です。HF・core・randomの軌道、simultaneous・alternatingの
最適化、一軌道更新、wavefunction更新、不活性軌道緩和に使用できます。
初期化時には軌道を変更しません。determinant評価では中心化したprojected求解、
permutation評価では既存のスピン行列による求解を使います。
初期化用の `initial_spin_overlap_cutoff` は最適化用の `spin_overlap_cutoff` と別で、
既定値は1e-10です。数値的に保持できる計量部分空間と入力状態を用いて求解し、
実際のエネルギーを上げる更新は採用しません。小さい計量固有値を無条件に逆数化しません。

`initial_spin_guess` の既定値は従来互換の `"uniform"` です。新規計算では
Csを等係数1/sqrt(fs)で開始し、`initial_json` を指定した再開では保存Csを保持します。
再開に `"eigen"` を明示した場合だけ、読み込んだ軌道を固定してCsを解き直します。
初期化結果、Cs、計量ランク、更新前後の活性エネルギーと採用状況は結果JSONの
`calculation.spin_initialization` に記録します。詳細診断では `initial_spin` イベントも保存します。
交互最適化の初回スピン求解は引き続き実施し、初期化と最適化で異なるカットオフも使えます。

上の例は最初の20スイープを交互最適化だけで進め、21スイープ目の交互更新後から
各スイープ末に同時補正を追加します。補正だけの独立した最適化へ置き換える指定ではありません。
切り替え時に軌道・Cs・ノルムをリセットしません。`joint_correction_start` は1以上の整数で、
1なら従来の「初回から補正」です。1以外の指定には `joint_correction = true` が必要です。
各スイープの診断に `optimization_phase` と `joint_correction_active` を記録します。

開始番号は各 `optimize` 呼び出し内の1始まりです。再開・不活性軌道緩和の各内側求解でも
番号は1に戻ります。上限より後の開始番号では、その呼び出し中には補正しません。
切り替え前に全体勾配・規格化の収束条件を満たせば、補正せず収束終了します。
補正を待つ有限の前処理期間は、エネルギー停滞だけでは停止せず、指定開始まで進めます。
切り替え時に停滞カウンタをリセットし、その後は従来と同じ5回の停滞で未収束停止します。
エネルギー増加の検出、反復上限、全体勾配1e-6以下と規格化の収束判定は全期間で維持します。

対応するCLIは `--initial-spin-guess eigen`、`--initial-spin-overlap-cutoff 1e-12`、
`--joint-correction --joint-correction-start 21` です。
例入力は `examples/inputs/fe_co5_staged_joint.toml` にあります。
20スイープは設定例で、Fe(CO)5について検証済みの最適値ではありません。

### 実験用の補正APIとの違い

通常の `joint_correction = true` が呼ぶのは `CoupledOptimization.correction` の
最大6方向の補正です。残差方向を追加する `adaptive_correction` や、軌道側だけを減衰して
スピン応答を戻す `SpinResponseCorrection.response_correction` を選ぶTOMLキーはありません。
これらは [HPC_README.md](HPC_README.md#保存状態から同時補正だけを比較する)に記載した
診断スクリプトまたは低水準APIから明示的に呼びます。
診断スクリプトは同じ保存状態から同時補正だけを試し、交互最適化を挟みません。
これは上の `joint_correction_start` による、交互更新への補正追加とは異なる実行です。
Fe(CO)5でのこれらの試行では収束は得られておらず、収束済みの設定例ではありません。
