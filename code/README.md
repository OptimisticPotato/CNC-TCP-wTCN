# CNC → TCP TCN

`TCN_CNC2TCP_SPEC.md` 구현. CNC 지령 궤적에서 `e = TCP − CNC` [µm]를 causal TCN으로
end-to-end 예측하고 `TCP = CNC + e`로 복원한다.

```
project/
├── Runs/                       # 공정 단위 폴더, 각각 *_out.csv 하나
│   ├── DOE_0001_.../
│   └── ...
└── code/                       # 이 폴더
```

`config.py`가 위 구조와 "code가 Runs 안에 있는" 구조를 모두 자동 인식한다.
경로가 다르면 `--set paths.runs_dir=...` 또는 환경변수 `TCN_RUNS_DIR`로 덮어쓴다.

### VS Code

**`project/code`를 열면 된다.** 인터프리터(`code/.venv`)가 자동으로 잡히고,
터미널 작업 디렉토리가 `code/`라서 아래 명령들이 문서 그대로 돌아간다.
`.vscode/settings.json`에 인터프리터 경로와 검색/파일감시 제외가 들어 있다.

데이터 폴더도 같이 보고 싶으면 `code/tcn.code-workspace`를 열면 `code`와
`Runs`가 멀티루트로 붙는다. `project` 폴더를 통째로 여는 것은 권하지 않는다 —
`Runs`가 968개 폴더 / 2.7 GB라 검색과 파일 감시가 느려진다.

작업 디렉토리는 사실 아무 곳이어도 된다. 경로 해석은 전부 `config.py`의 위치
기준(`__file__`)이고 스크립트는 `scripts/_bootstrap.py`로 import 경로를 스스로
세우므로, `python D:\project\code\scripts\05_train.py`처럼 절대경로로 불러도 동작한다.
문서의 명령줄이 `scripts\...`로 시작할 뿐이다.

---

## 1. 설치 (학습 PC에서 한 번)

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

`.venv` 생성 → CUDA torch(cu121) → 나머지 패키지 순으로 설치한다.
드라이버가 최신이면 `-CudaTag cu124`로 바꿔도 된다. Python 3.10 기준.

```powershell
python scripts\00_check_env.py --selftest
```

`--selftest`는 합성 데이터로 **torch 경로 전체**(모델 생성 → 구조 보장 2종 검증 →
forward → 3항 손실 → backward → optimizer step → 청크 추론)를 돌린다.
이 코드는 GPU가 없는 PC에서 작성돼 torch 부분이 실행 검증되지 않았으므로,
**학습 전에 이것부터 통과시킬 것.**

---

## 2. 실행 순서

```powershell
python scripts\01_scan_runs.py                      # 전 런 스캔 → work/meta/index.csv
python scripts\02_select_doe.py                     # 32종 선별 → selection_r1.txt
python scripts\03_validate_loader.py --selection r1 # SPEC 7-1 로더 검증
python scripts\04_overfit_test.py                   # SPEC 7-2 과적합 테스트
python scripts\05_train.py --selection r1           # SPEC 7-3 본 학습
python scripts\06_evaluate.py --run tcn_r1          # SPEC 8 지표 1~5 + 그림 3종
```

포화점·수용영역·리샘플 강건성 (지표 6/7/8):

```powershell
python scripts\02_select_doe.py --set select.n_programs=64 --set select.tag=r2
python scripts\07_sweep_ndoe.py --selection r2      # 지표 6: DOE 종수 대비 오차
python scripts\08_sweep_rf.py  --selection r1       # 지표 7: 100/200/300 ms
python scripts\09_resample_robustness.py --run tcn_r1   # 지표 8
```

추론 (임의 타임스탬프 → 표준 격자 → TCN → 원 타임스탬프 역보간):

```powershell
python infer.py --run tcn_r1 --run-name DOE_0001_RANDOMWALK3D_FINE_F500_S01
python infer.py --run tcn_r1 --csv D:\어딘가\1234_op1_out.csv --out pred.csv
```

라이브러리로도 쓸 수 있다.

```python
from infer import Predictor
p = Predictor("tcn_r1")
tcp = p.predict(t, cnc, feed)      # 어떤 dt, 어떤 타임스탬프든 가능
```

### 소요 시간 감각 (RTX 3060 12GB 기준 추정)

| 단계 | 대략 |
|---|---|
| 01 전 런 스캔 (844종, 8 worker) | 10초 내외 |
| 03 로더 검증 32종 | 1분 내외 |
| 04 과적합 테스트 30 epoch | 5~10분 |
| 05 본 학습 60 epoch × 300 step × batch 16 | 1~2시간 |
| 07 종수 스윕 4점 | 05 × 4 |
| 08 수용영역 스윕 3점 | 05 × 3 |

VRAM은 기본 설정(window 6144 × batch 16 × 48ch, RF 509)에서 2 GB 미만이다.
여유가 있으니 `--set train.batch_size=32`나 `--set model.channels=64`로 올려도 된다.

---

## 3. config.py

모든 경로·하이퍼파라미터가 여기 한 곳에 있다. 세 가지 방법으로 덮어쓴다.

```powershell
# 1) 커맨드라인 — 모든 스크립트 공통
python scripts\05_train.py --set train.epochs=40 --set model.channels=64 `
                           --set model.dilations=1,2,4,8,16,32,64,128

# 2) 환경변수
$env:TCN_RUNS_DIR = "E:\data\Runs"

# 3) config_local.py — 이 파일을 만들면 config.py 다음에 import 된다 (git 무시 대상)
```

자주 건드릴 값:

| 키 | 기본 | 뜻 |
|---|---|---|
| `paths.runs_dir` | 자동 탐지 | 공정 폴더들이 있는 위치 |
| `paths.work_dir` | `code/work` | 캐시·체크포인트·리포트 출력 |
| `data.dt` | `0.0005` | 표준 격자. **바꾸면 학습된 커널의 물리 시간축이 통째로 스케일된다** |
| `data.g00_source` | `"feed"` | `feed` 컬럼 사용. `"speed"`는 폴백, `"none"`은 3채널 |
| `data.rapid_feed_auto` | `True` | 급속 feed를 데이터에서 측정 (아래 참조) |
| `data.rapid_feed_mm_min` | `40000` | 측정 실패 시 폴백, `auto=False`면 강제값 |
| `model.dilations` | `1..64` | RF = 1+4·Σd. `1..64`→509샘플(254 ms), `1..128`→1021(510 ms) |
| `model.linear_skip` | `False` | SPEC 5.2의 선형 skip. **학습이 안 될 때 제일 먼저 켤 것** |
| `loss.lambda_diff` / `lambda_stft` | 0.5 / 0.3 | 진동이 뭉개지면 올린다 |
| `loss.g00_weight` | 0.1 | G00 구간 손실 가중치 (입력에서는 절대 제거하지 않음) |
| `train.window` | 6144 | RF보다 충분히 길어야 함 (앞 RF 샘플은 손실에서 제외) |
| `augment.enabled` | `True` | 리샘플 증강 (SPEC 7-3) |

---

## 4. 구조

| 파일 | 역할 |
|---|---|
| `tcn_cnc2tcp/runs_io.py` | `*_out.csv` 파싱(float64, 단위행 skip, 후행 콤마 처리), dt 검사, 런 통계 |
| `tcn_cnc2tcp/resample.py` | cubic spline 표준격자 왕복, 증강용 손상·복원 |
| `tcn_cnc2tcp/features.py` | 증분, G00/G01 6채널 마스킹, 정규화, 손실 가중치 |
| `tcn_cnc2tcp/dataset.py` | 캐시, **프로그램 단위 분할**, 랜덤 크롭, 전체 시퀀스 추론 |
| `tcn_cnc2tcp/model.py` | causal·bias-free·dilated TCN, 구조 보장 자체 검증 |
| `tcn_cnc2tcp/losses.py` | MSE + diff-MSE + multi-resolution STFT (전부 마스킹) |
| `tcn_cnc2tcp/trainer.py` | 학습 루프, 검증, 체크포인트 |
| `tcn_cnc2tcp/metrics.py` | SPEC 8 지표 1~5 |
| `tcn_cnc2tcp/plots.py` | SPEC 8 그림 3종 |
| `infer.py` | 추론 파이프라인 |
| `excluded_runs.txt` | 학습에서 제외하는 런 목록 (이유 주석 포함) |

출력은 전부 `work/` 아래로 간다.

```
work/
├── meta/     index.csv, ranking.csv, selection_*.txt
├── cache/    런별 .npz (CSV 재파싱 회피, 런당 ~1.5 MB)
├── runs/     <train.name>/{best.pt, last.pt, split.json, config.json, train.log}
└── reports/  <run>/<split>/{metric*.csv, fig*.png, tcp/*.csv, summary.json}
```

---

## 5. 스펙 대응표

| SPEC | 구현 위치 |
|---|---|
| 4.1 출력 `e`[µm], 좌표 float64 | `features.build_target`, `runs_io` 전체 float64 |
| 4.2 증분 입력, 6채널 | `features.build_inputs`, 항등식은 `check_identity`가 매 로드마다 검사 |
| 4.3 저크 채널 금지 | 파생 채널 자체가 없음. 미분은 conv 커널이 내부적으로 만든다 |
| 2 인과성 / 영입력→영출력 | `model.CausalConv1d`(좌측 패딩), 전 층 `bias=False`, `f(0)=0` 활성함수 |
| 2 BatchNorm 금지 | `norm="none"` 기본, `"batchnorm"`은 생성자가 거부 |
| 2 G00/G01 채널 분리 | `features.g00_mask` — 시퀀스는 자르지 않고 채널만 나눔 |
| 3.1 feed로 G00 판별 | 급속 feed는 상수가 아니라 `runs_io.detect_rapid_feed`가 측정 |
| 5.3 3항 손실 | `losses.TCNLoss` |
| 5.4 마스킹 | `features.loss_weight` (앞 RF 샘플, NaN 행, G00 가중치) |
| 6.2 표준 격자 | `resample.run_to_grid`, `infer.Predictor` |
| 6.4 cubic spline | `resample`, PCHIP은 §6.4 재측정용으로만 남김 |
| 7 프로그램 단위 분할 | `dataset.split_programs` — 윈도우 분할 경로는 존재하지 않음 |
| 8 지표 1~8 | `metrics` + 스크립트 06/07/08/09 |
| 3.3 DOE 선별 | `scripts/02_select_doe.py` |

---

## 6. 이 데이터셋에서 실측한 값

스펙 부록은 다른 장비(`322_short` / Wia_KF5)의 값이므로 그대로 쓰면 안 된다.
아래는 이 `Runs` 데이터에서 직접 측정한 것이다.

| 항목 | 값 |
|---|---|
| 사용 가능 런 | 844 (968 폴더 중 124 제외) |
| 총 샘플 | 24,307,818 행 = 202.6 분 |
| dt | 전 런 0.5 ms 균일 (비균일 2런은 제외 목록에 있음) |
| `e = TCP − CNC` RMS | X 16.0 / Y 6.2 / Z 13.5 µm (중앙값) |
| `e` 최대 | 1150 µm (F15000 TWO_AXIS_CIRCULAR), 중앙값 108 µm |
| `e` 에너지 | 99 %가 ~15 Hz 이하, 100 Hz 초과는 1.5e-4 |
| 전달함수 공진 | ~40 Hz, ~116 Hz (\|G\| 최대 7.2, Y축) |
| G00 비중 | 0.06 ~ 0.75 (중앙값 0.19) |
| 절삭 feed 수준 | 500, 1000, 1500, 3000, 5000, 6000, 7500, 10000, 15000 mm/min |
| 급속 feed | 40000 mm/min |
| 빈 행 | 런당 1행(마지막). 타깃에서 제외됨 |
| float32 좌표 양자화 | 최대 0.015 µm — `e` RMS 6 µm에 대해 무시 못 할 크기 |

**표준 격자를 0.5 ms로 고정한 이유:** 전달함수에 116 Hz 공진이 있어
"지배 진동의 10배" 규칙이 1160 Hz 이상을 요구한다. 스펙이 허용한 1 ms(1000 Hz)는
이 데이터에서 미달이고, 원본이 이미 0.5 ms이므로 굳이 거칠게 만들 이유도 없다.
다만 `e`의 에너지 자체는 15 Hz 이하에 몰려 있으므로, 실제 영향은
공진 대역을 얼마나 정확히 재현하려는지에 달려 있다.

**cubic spline 왕복 오차** (`DOE_0268`, 40000샘플, 이 코드로 측정):

| 원본 샘플링 | 위치 최대 | 위치 RMS | 증분 최대 |
|---|---|---|---|
| 0.5 ms 지터 ±50 % | 0.023 µm | 0.0006 µm | 0.040 µm |
| 0.5 ms, 5 % 누락 | 0.090 µm | 0.0010 µm | 0.090 µm |
| 1 ms | 0.085 µm | 0.0029 µm | 0.085 µm |
| 2 ms | 0.153 µm | 0.0067 µm | 0.127 µm |
| 4 ms | 0.742 µm | 0.0334 µm | 0.316 µm |
| 8 ms | 2.672 µm | 0.1597 µm | 0.751 µm |

스펙 §6.3(선형 커널, 다른 프로그램)보다 크지만 같은 결론이다 — 4 ms까지는
`e` 크기 대비 무시 가능. 8 ms부터는 아니다. 그래서 증강(§7-3)을 켜 둔다.

---

## 7. G00 / G01 판별

데이터셋을 쪼개지 않는다. 시퀀스는 연속으로 두고 증분만 채널 두 그룹으로 나눈다
(스펙 §2) — G00 감속의 링잉이 다음 G01 절삭 초반으로 흘러드는데, 경계에서 자르면
그 원인이 입력에서 사라진다.

판별은 `feed` 컬럼으로 한다(스펙 §3.1). 다만 **급속 feed 값은 코드에 박지 않는다.**
`01_scan_runs.py`가 데이터에서 측정해 `work/meta/machine.json`에 쓰고,
나머지 전부가 그 값을 읽는다.

측정 방식은 "런별 최대 feed의 **최빈값**"이다. 급속은 장비 상수라 프로그램 수백 개에
걸쳐 같은 값으로 반복되므로, 이상치 런 하나가 값을 끌고 가지 못한다. 이 데이터셋에서는
844런 전부가 40000 mm/min으로 일치했다.

판단하지 못하는 두 경우는 조용히 넘어가지 않는다.

| 상황 | 동작 |
|---|---|
| 최상위 feed가 그 아래와 1.5배 미만 차이 (예: 급속 20000, 절삭 15000) | `detected: false`, 이유를 남기고 설정값으로 폴백 |
| 급속이 아예 없는 데이터셋 | 최고 절삭 feed를 급속으로 오인함 — **feed 값만으로는 구분 불가.** `01`이 찍는 G00 비율이 검산이다 (0.00이나 0.9 이상이면 틀린 것) |

강제 지정:

```powershell
python code\scripts\01_scan_runs.py --set data.rapid_feed_auto=false --set data.rapid_feed_mm_min=24000
```

입력 정규화 상수(`increment_scale`)는 자동 측정값이 아니라 **설정값**을 쓴다.
학습된 망은 학습 때의 입력 스케일을 기대하므로 데이터가 바뀌어도 움직이면 안 된다.
체크포인트가 `dt`, `rapid_feed_mm_min`, `rapid_feed_tol`, `g00_source`를 기록하고
로더가 복원하며, 현재 설정과 다르면 알려준다.

NC 원문 역참조(스펙 §3.1의 "필요하면 block#으로")는 구현하지 않았다. `.nc`에 G0/G1이
명시돼 있지만 CSV의 `block#`이 NC 줄 번호와 깔끔하게 대응하지 않는다 — N번호가
10, 12, …, 332 뒤에 4416, 5326으로 건너뛰고, `block#` 6·7은 6·7번째 N줄과 맞는데
8부터 어긋난다. 시뮬레이터의 블록 번호 규칙을 확정하기 전에는 믿고 쓸 수 없다.
필요해지면 그 규칙부터 확인해야 한다.

---

## 8. 제외한 런 (`excluded_runs.txt`, 124종)

| 그룹 | 수 | 이유 |
|---|---|---|
| CSV 없음 | 2 | `.nc`만 존재 (시뮬레이션 실패) |
| 다른 샘플링 | 2 | `DOE_9000`(7 ms), `DOE_9999`(2.5 ms), 비균일 |
| **다른 서보 동특성** | 120 | 아래 |

120종은 나머지 844종과 CNC→TCP 관계 자체가 다르다. 두 방법이 같은 집합을 가리켰다:
저주파 FRF가 빈 구간을 사이에 두고 이봉이고(`|G_x(9.8Hz)|` 1.118 vs 1.178),
FIR 모델 상호전이 테스트에서 같은 그룹끼리는 잔차 비율 1.0배, 다른 그룹 간에는
4.2배였다. 경로·이송이 같고 시드만 다른 형제 런이 갈라지므로 경로 특성이 아니라
런별 설정 차이다. LTI 가정이 전제인 이 모델에 섞으면 안 된다.

다시 넣고 싶으면 해당 줄을 지우거나 `--set paths.exclude_file=없는파일` 하면 된다.

---

## 9. 알려진 한계

* **torch 경로 미실행 검증.** 작성 PC에 NVIDIA GPU가 없어 모델·학습·평가·추론은
  정적 검토만 거쳤다. 로더·선별·리샘플·지표·그림은 실제 데이터로 돌려 확인했다.
  `00_check_env.py --selftest`가 그 공백을 메우도록 만들어져 있다.
* **SPEC 8 지표 5(최종 형상오차)는 여기서 끝나지 않는다.** 형상오차 파이프라인이
  이 저장소 밖에 있으므로, `06_evaluate.py`는 예측 TCP를 프로그램별 CSV로 내보내고
  (`work/reports/<run>/<split>/tcp/*.csv`) 대신 **법선 방향 편차**를 근사 지표로
  보고한다. 근사라는 사실은 코드와 출력 양쪽에 명시돼 있다.
* **스펙 §2의 시간창 기대치(100 ms → 30 nm 등)는 다른 장비 값이다.** 이 데이터의
  실제 개선폭은 `08_sweep_rf.py`로 측정해야 한다.
* TCN이 FIR보다 나을 것이라는 보장은 없다. 스펙도 합격 기준을 두지 않는다 —
  지표를 산출해 보고할 뿐이다.
