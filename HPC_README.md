# SCGVB 12π電子版：スパコン持ち込み手順

この配布物は **1プロセス・1 GPU・FP64** を対象とする。従来のfull SCGVBを維持し、
固定閉殻と活性空間も同じ評価器で扱う。複数GPU・複数ノードへの自動分散は行わない。
平面ポリエン12π電子はrikyuのGB200で収束を確認済み（ジョブ125026、最適化ステップ2分46秒）。
局所法線による非平面π空間選択はCPUでエネルギー・勾配を検証済みだが、
今回のビフェニル例の最適化収束とGPU実行は未検証。性能・ピークメモリは対象ごとに測定する。

## 1. コピーするもの

`dist/scgvb-12pi.tar.gz` と `dist/scgvb-12pi.tar.gz.sha256` の2ファイルを転送する。
ソース、CLI、full版、テスト・参照データ、XYZ例、実行スクリプト、依存関係リスト、
CPU検証レポートを同梱している。Macの仮想環境、キャッシュ、大きなCubeファイル、
計算中の出力、プライベート設定は含まない。バイナリ依存パッケージは別途インストールする。

```bash
sha256sum -c scgvb-12pi.tar.gz.sha256
tar -xzf scgvb-12pi.tar.gz
cd scgvb-12pi
python hpc/verify_manifest.py
```

変更後に再配布する場合は `python build_hpc_bundle.py` を実行する。
MANIFEST.jsonは配布時点の各ファイルのSHA256を記録する。

## 2. GB200向けのPython環境

Grace CPUはLinux aarch64なので、**Macの環境をコピーしない**。
まず施設指定のPython/CUDAモジュールやコンテナ利用規則を確認する。
以下はPython 3.12とpipで独立環境を作る例。施設提供JAXを使う場合はそれを優先し、
バージョンを記録して同梱テストを実行する。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu-cuda13.txt
```

このリストは開発時のJAX 0.11.2と同じ版のCUDA 13パッケージを指定する。
GPU依存パッケージのインストール自体は開発機で検証していない。
JAX公式手順ではLinux aarch64/x86_64のGPU wheelが提供され、CUDA 13版には
Linux NVIDIAドライバー580以上が必要。施設の対応状況に従って環境を選ぶ。
CUDA 12等の施設環境を使う場合は、GB200との互換性も管理者に確認し、依存関係を
調整した環境でテストする。ドライバーをこのスクリプトが変更することはない。
[JAX公式インストール手順](https://docs.jax.dev/en/latest/installation.html)

PySCFはpipまたはconda-forgeのARM64版を利用できる。wheelのない環境では
ビルドが必要な場合があるため、施設提供版も選択肢になる。
[PySCF公式インストール手順](https://pyscf.org/user/install.html)

オフライン計算機では、同じLinux aarch64/Python版のネット接続環境で依存wheelを
取得し、まとめて転送する。CPUアーキテクチャが違うログインノードでそのまま
`pip download` すると異なるwheelを取得するため注意する。

```bash
# 対象と同じLinux aarch64/Python環境で実行
python -m pip download --only-binary=:all: -r requirements-gpu-cuda13.txt -d wheelhouse
# wheelhouseも転送後、対象環境で実行
python -m pip install --no-index --find-links wheelhouse -r requirements-gpu-cuda13.txt
```

GPUなしの検証環境には `requirements-cpu.txt` を用いる。

## 3. 割り当て済み計算ノードで確認

まず1 GPUを確保する。GPUの可視性はスケジューラが設定するものを尊重し、
未割り当てGPUを使うために `CUDA_VISIBLE_DEVICES` を上書きしない。
CPUのスレッド数も割り当てに合わせる。以下は1スレッドでの確認例。

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python hpc/check_environment.py --device gpu
```

デバイス名、アーキテクチャ、パッケージ版、FP64の自動微分テスト結果を表示する。
GPUが利用できなければ終了コード1で停止し、CPUへの暗黙フォールバックはしない。
この確認はログインノードではなく、GPUの割り当てられた計算ノードで実行する。

## 4. 検証・測定・計算

```bash
# 短いGPU検証（12eのエネルギーと方向微分を独立なNumPy/PySCFと比較）
bash hpc/run_12pi.sh verify

# 合成12e/12o、および実分子ポリエン12π電子のベンチマーク
bash hpc/run_12pi.sh benchmark

# ポリエン12π電子のSCGVB最適化
bash hpc/run_12pi.sh optimize
```

実行ごとに `results/12pi-日時-PID/` を作る。保存先は `SCGVB_OUTPUT_DIR` で指定できる。
`environment.json` と `pip-freeze.txt`、ログ、結果JSONを保存する。
例示ポリエンは平面C12H14の理想化構造であり、構造最適化済み分子ではない。
開発機ではこの構造のエネルギー・勾配を検証し、12π電子の軌道最適化収束までは
確認していない。最適化には多数の評価が必要で、ベンチマーク時間からジョブ時間を見積もる。

全回帰テスト（主にCPUで実施、一部JAXテストは環境依存）も実行可能：

```bash
SCGVB_RUN_12E=1 SCGVB_TEST_DEVICE=gpu python -m unittest discover -s tests -t . -p 'test_*.py' -v
```

Slurm利用施設向けの雛形は `hpc/slurm_1gpu.example.sh`。
アカウント、パーティション、GPU要求構文、環境有効化、時間・メモリ指定は
施設に合わせて編集する。実際のスケジューラが未確認なので、自動投入は行わない。
4 GPU割り当て時も現実装は1 GPUだけを使う。複数の独立した初期値・構造の計算は、
スケジューラで1 GPUずつ割り当てた別プロセスとして実行する。

### 別のπ系

```bash
python calc_molecule.py --xyz examples/COT.xyz --active-space pi --pi-electrons 8 \
  --energy-evaluation determinant --backend jax --device gpu --initial-guess hf \
  --output cot_pi.json
```

自動π空間選択は炭化水素が対象。既定は平面分子で、非平面sp2系には
`--pi-normal-mode local` を追加する。ビフェニルの例は `ACTIVE_SPACE.md` を参照。
電子数とスピン、サイズ上限を検査する。
占有・非占有の両方を含めてπ部分空間を選び、σ電子・内殻電子は固定閉殻にまとめる。
詳しい選択方法と手動MO指定は `ACTIVE_SPACE.md` を参照。

## 5. 今回の評価器の変更

- 行列式展開の小行列式は、次数k-1からkへのLaplace再帰で構成する。
  k!個の置換の列挙や逆行列を使わず、特異な部分行列でも多項式として微分できる。
  各次数をJAX checkpointで再計算し、逆伝播の保存量を減らす。
- 化学者表記ERIの対称な軌道ペア空間を対角化し、Hamiltonianを一体演算子の二乗和で表す。
  固有値は切り捨てず、理論上の近似は追加しない。丸め誤差は通常どおり存在する。
- 一体演算子の作用は32行以下のタイルでgather/縮約する。
  `n_orbital**2 × n_det_alpha × n_det_beta` の中間配列を作らない。
- 固定した実対称Hamiltonianに対し、`(c.T H c, c.T c)` の専用VJPは
  `2 * (energy_cotangent * Hc + norm_cotangent * c)`。
  Hamiltonian内部の処理を逆伝播用に保存せず、cとHcから勾配を返す。
  その後、非直交軌道・スピン係数へ自動微分する。
- 積分・演算子・結合表は初期化時に選択デバイスへ転送する。SciPy最適化はCPUで動くが、
  各評価でやり取りするのはパラメータ、エネルギー、勾配であり、巨大な展開配列は戻さない。

閉殻・活性空間基底・積分は最適化中に固定。実軌道と実スピン係数の一階微分を対象とし、
核座標の微分や、非活性軌道の同時最適化を実装したものではない。
各スピン6電子以下、行列式係数・各次数の小行列式・primitiveスピン係数・変換後の二電子積分配列それぞれ
既定の `standard` では200万要素以下という制限を置く。これはGPUの空きメモリ検査でもピークメモリの保証でもない。

## 6. ベンチマークの読み方

`calc_molecule.py --optimize-core` で二重占有を保った非活性軌道の緩和も可能。
これは固定coreの12πベンチマークとは別の計算で、外側の積分変換・密度行列・軌道回転は
現在CPUで行い、指定GPUは内側のSCGVB評価に使用する。マクロ反復ごとに評価器の
再構築が発生する。12πでの緩和計算の速度・収束を検証済みとはしていない。
設定と小分子での使用例は `ACTIVE_SPACE.md` を参照。

`scgvb/benchmarks/active_space.py` は最適化をせず、同じパラメータでエネルギーと勾配を評価する。
Pythonの最終評価キャッシュを通さず、各回GPU実行完了を待って測定する。

- `setup_seconds`：HF・積分・評価器準備を含む初期化。
- `compile_seconds`：JAX/XLAのコンパイル。
- `first_execution_seconds`：コンパイル後の最初の実行。
- `warm_execution_seconds`：ウォーム実行の各回の秒数。
- `compiled_memory_analysis`：XLAが報告するコンパイル済み実行のバッファ見積もり。
  デバイスのピーク使用量全体ではない。
- `device_allocator_memory_stats`：バックエンドが提供する場合のallocator統計。
  CPUでは通常null。GPUの予約領域と実使用領域は区別する。
- `process_peak_rss_bytes`：ホストプロセスの最大RSS。GPUメモリではない。
- `verification`：独立NumPy/PySCFとのエネルギー・方向微分の誤差。

CPU検証結果は開発環境の `hpc/validation/` に保存しており、GitHub・配布アーカイブには含めない。コンパイルの一時領域見積もりは
12e/12oで約389 MiBだったが、GPUでのコンパイラの選択・一時領域は別途測定が必要。
測定値からGB200の速度を外挿した値は記載していない。

## 3/4/6活性電子：permutationとdeterminantの速度比較

```bash
python -m scgvb.benchmarks.evaluators --case all --basis cc-pvdz --device gpu \
  --repeats 30 --output results/evaluator-comparison
```

allyl (3e, doublet)、cyclobutadiene (4e, singlet)、benzene (6e, singlet)の
π活性空間を同一HF参照から作る。各分子の同じHF摂動軌道・スピン係数を
`fixture.npz`に保存し、評価器ごとに独立した子プロセスへ渡す。
coreと活性空間は固定し、軌道最適化は実施しない。

測定は初期化、最初のエネルギー＋ノルム評価、続く最初のRayleighエネルギー＋
勾配評価、3回のウォームアップ後の30回評価を分離する。最初の呼出しはJITを含み、
純粋なコンパイル時間ではない。中央値・最小値・最大値・全サンプルを保存する。
ホストからの引数転送と結果取得を含め、NumPy値へ変換してGPU同期する。
評価キャッシュは毎回無効化する。全波動関数の規格化と定数項に依存しない
`E_active=E_numerator/norm`の勾配を比較する。

`summary.json`に評価器間のエネルギー・ノルム・勾配一致判定と速度比を保存する。
各子プロセスのCPUピークRSSとJAX device memory statsも記録する。
メモリピークは初期化・コンパイルを含むプロセス全体で、評価1回の増分ではない。
各方式の結果とログは分子別フォルダに保存する。既存出力ディレクトリは上書きしない。

## 保存状態から同時補正だけを比較する

`hpc/probe_joint_correction.py` は同じ保存されたsinglet状態から、交互最適化を挟まず、
既存の6方向の同時補正の更新幅・方向数・方向採用下限を比較する診断用スクリプト。
`hpc/probe_adaptive_joint.py` は、その補正候補で生じたスピン残差方向を最大4方向追加し、
最大10方向で同じ補正前状態から解き直す方式と既存方式を比較する。
どちらも `solve-*.npz` の固定積分（`S/h1/eri/ecore`）と保存状態の `C/Cs` を読み込む。
HF初期化・活性空間再構築・単独の軌道/スピン更新は実施しない。
既存の方向構築内の固定軌道スピン求解は、補正方向の候補を作るためにだけ使用する。

```bash
python hpc/probe_adaptive_joint.py --problem /path/to/solve-0001.npz \
  --state /path/to/joint_before.npz --problem-sha256 <SHA256> --state-sha256 <SHA256> \
  --expected-energy=<total-energy-hartree> --norm-floor=<fixed-unit-column-norm-floor> \
  --device gpu --steps 10 --output /path/to/new-diagnostic-directory
```

実行は計算ノードで行う。データのハッシュと出発エネルギーを照合する。
最初の6方向を共通に保存し、同じ計量配列形状で旧方式の最初の補正と適応方式の
初回候補の一致を検証する。比較はtrust radius 0.02と0.001、各方式最大10回。
適応補正は、同じノルム下限とtrust radiusを保持して元の状態で縮約計量・曲率を
再評価する。候補状態へ移った上で次の補正を重ねる方式ではない。

候補は従来の実エネルギー低下・ノルム下限・軌道変位/波動関数距離の条件に加えて、
規格化した全物理勾配のL2ノルムが補正前から1e-8を超えて増えず、スピン勾配L2が
`max(1e-6, 2 * incoming_spin_gradient_l2)`以下の場合に採用する。
これは実験的な候補採否条件であり、最大物理勾配1e-6・規格化誤差1e-8という
既存の収束判定は変えていない。候補の全てが条件を満たさなければ元の状態を保持して停止する。
追加方向は元のCsの接線へ投影し、既存基底へ二度直交化する。相対独立性が1e-10以下の
重複方向は追加しない。計量の相対カットオフは1e-12のまま。

各追加段階の状態・勾配・縮約計量/曲率・残差・採否を保存する。
`summary.json` の `diagnostic_completed` と `scientific_convergence` は別に読む。
低水準APIの `CoupledOptimization.adaptive_correction` は明示的に呼ぶ実験的な追加方式であり、
通常の `joint_correction` 指定・交互最適化・開始点指定の既定動作は変更していない。

## 軌道だけを減衰するスピン応答付き同時補正

`SpinResponseCorrection.response_correction` は実験用の独立した低水準API。
従来の `correction` / `adaptive_correction` と通常の交互最適化の既定動作は保持する。
縮約Hessianのブロックから
`ds = -Hss^{-1} gs - Hss^{-1} Hso do` を求め、これを軌道方程式へ代入する。
軌道側のSchur補行列だけにlevel shiftを加える。スピン側のlevel shiftは0。
スピン計量は相対1e-12で解像するが、ランク欠損や正でないスピン曲率があれば
補正を保持して停止する。スピン曲率の採用下限は `1e-12 * max(1, max(abs(curvature)))`。
固定軌道で得たスピン定常点の近傍を想定した方式である。

混合した前回の同時変位も、軌道成分とスピン成分の各張る空間へSVDで分ける。
各成分の特異値は `max(1e-10, 1e-10 * largest_singular_value)` を超えるものを保持する。
分割により初期方向数が増える場合がある。追加する補正後のスピン残差方向は最大4、
総方向数は最大10。追加モデルはいずれも補正前の元の状態を基準に解く。

trust radiusは、軌道変化とそれに誘起されるスピン応答の物理距離を制限する。
`do=0`でも存在するスピン補正はこの半径に含めない。実際の全波動関数距離0.05、
各軌道のS距離0.1、実エネルギー低下、固定ノルム下限は全候補に課す。
全物理勾配L2とスピン勾配L2の採否条件は前節の適応方式と同じである。
追加方向でも採否条件を満たさない場合は、軌道座標だけを半減し、その都度スピン応答を
上式から求め直す（最大30候補）。同時変位全体を半減する方式ではない。
局所モデルのスピン残差を消しても、有限更新後の全スピン残差が消える保証はない。
各候補・局所モデル・予測スピン残差・軌道の半減履歴を保存して確認する。

```bash
python hpc/probe_spin_response.py --problem /path/to/solve-0001.npz \
  --state /path/to/joint_before.npz --problem-sha256 <SHA256> --state-sha256 <SHA256> \
  --expected-energy=<total-energy-hartree> --norm-floor=<fixed-unit-column-norm-floor> \
  --device gpu --steps 10 --output /path/to/new-diagnostic-directory
```

同じ保存状態・共通の初期6方向から、共通減衰の適応方式とスピン応答付き方式を
trust radius 0.02 / 0.001で比較する。新方式の半径は上記の応答後の計量で測るので、
旧方式の全同時変位の半径とは定義が異なる。HF再初期化や交互最適化は実施しない。
`diagnostic_completed` と `scientific_convergence` の区別と収束判定は前節と同じ。
内部の勾配評価回数には、診断出力を保存するcallbackでの追加評価は含めない。

## 14電子・14軌道の実行

通常コードで `--determinant-size-profile large14` を明示的に指定する。
既定は従来の `standard`。TOML・Python API・再開・コア緩和・電子密度解析の
設定方法と範囲は `ACTIVE_SPACE.md` の「行列式サイズ設定：14電子対応」を参照。

```bash
python -m scgvb.benchmarks.active_space --device gpu --synthetic-orbitals 14 \
  --determinant-size-profile large14 --verify --output benchmark14.json
```

フェナジン14e/14o、cc-pVDZの固定非活性軌道（ジョブ165725）と
全40本の非活性軌道緩和（ジョブ166080）の収束を確認済み。
結果は `runs/rikyu_phenazine14_20261004/` と
`runs/rikyu_phenazine14_relaxed_20261004/` に保存している。
