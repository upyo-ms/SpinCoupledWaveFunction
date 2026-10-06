# 非直交軌道の表示

`generate_orbitals.py` は `calc_molecule.py` が保存した結果JSONを読み込み、
各非直交軌道をGaussian Cubeファイルとブラウザ用HTMLへ出力する。
既存の `generate_cube.py` はそのまま残している。
新しいプログラムでは分子や係数をソースコードへ書き込む必要はない。
SCF・Full CI・軌道最適化の再実行やJAXの導入も不要。

## 基本的な使い方

```bash
python calc_molecule.py --xyz examples/h3_plus.xyz --charge 1 --output result.json
python generate_orbitals.py result.json --output-dir orbitals
```

生成された `orbitals/index.html` をブラウザで開く。
軌道の選択、マウスでの回転・拡大、等値面値、不透明度の調整ができる。
正の振幅は赤、負の振幅は青。入力値を変更後に「表示を更新」を押す。
HTMLにはCubeデータを埋め込んでいるため、表示用のPythonサーバーは不要。

HTMLは既定で [3Dmol.js](https://3dmol.org/doc/GLViewer.html#addVolumetricData) を
公式サイトから読み込むため、インターネット接続が必要。
ローカルに保存したライブラリを使う場合は次のように指定する。

```bash
python generate_orbitals.py result.json --viewer-js /path/to/3Dmol-min.js
```

ライブラリを出力フォルダへコピーし、そのローカルファイルを参照する。
オフラインではHTMLとライブラリを同じフォルダに置いて使う。
HTMLを作らず、Cubeを外部の分子ビューアで表示する場合は `--no-html` を指定する。

## 軌道・格子の指定

```bash
# 1番目と3番目の軌道のみ（番号は1始まり）
python generate_orbitals.py result.json --orbitals 1 3 --output-dir selected

# 格子を細かくし、箱の余白を広げる
python generate_orbitals.py result.json --grid 80 80 80 --margin 4 --isovalue 0.03

# 各軌道を個別に規格化して表示する（直交化しない）
python generate_orbitals.py result.json --normalize
```

既定は全軌道、64×64×64点、余白3 Bohr、等値面の絶対値0.04。
格子箱は分子を囲む直方体で、座標と余白の単位はBohr。
等値面は軌道振幅（Bohr^(-3/2)）を表し、電子密度ではない。
同じ格子数でも分子が大きいと格子間隔が広くなる。
既定の上限は200万格子点。必要なら `--max-grid-points` で変更する。

## 係数の扱い

`phi_i(r) = sum_mu C[mu,i] chi_mu(r)` を計算する。
既定では保存された係数の符号・大きさを変更せず、軌道を直交化しない。
`--normalize` は各列を `sqrt(C_i.T @ S @ C_i)` で割るだけで、
軌道形状・位相・非直交性を維持する。軌道の全体スケールには自由度があるため、
同じ等値面値で軌道間の形状を比較したい場合に利用できる。
スピン係数から多電子密度や占有数を計算するプログラムではない。

結果JSONに保存された分子、電荷、多重度、基底、座標単位からAOを再構成する。
AOラベルの順序と重なり行列を照合し、不一致なら停止する。
異なる基底・原子順序の係数を誤って描画しないための検査である。
入力JSONは変更しない。同名の出力ファイルは上書きする。

## 出力ファイル

- `orbital_001.cube` 等：軌道番号ごとの符号付き振幅。
- `molecule.xyz`：Angstrom単位の分子構造。
- `index.html`：選択・回転できる軌道ビューア。
- `manifest.json`：格子情報、出力軌道、元の軌道ノルム、係数の倍率、
  表示に用いた軌道間重なり行列。

## 検証

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_generate_orbitals -v
```

5テストで、PySCFの `cubegen.orbital` との格子値一致（Cube出力の丸め精度内）、
選択軌道の順序、個別規格化、非直交性の保持、不正入力の検出、
ローカルライブラリ指定を確認した。H3+の結果ではブラウザで等値面表示も確認。

## 活性空間の結果

活性SCGVB軌道を先に、続いて固定された非活性空間軌道を1本ずつ表示する。
非活性軌道はHTMLの選択欄で「非活性・二重占有」と表示し、manifestにもroleを保存する。
読み込み時に電子数、閉殻の直交規格化、活性・非活性間の直交性を検査する。
旧full版JSONにも対応する。詳しくは [ACTIVE_SPACE.md](ACTIVE_SPACE.md) を参照。
