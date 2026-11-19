# Telephone Game

문서 편집 순서에 따른 사실 보존을 비교하는 **실제 4-role 파일럿 실험 코드**다. 동일한 Qwen 모델을 단순화·압축·교정·구조화 역할로 호출하며, 일곱 순서 선택 방법과 최종 성능·수명·ablation·외부 데이터 이전을 실행한다.

현재 구현 범위는 **영어 역할 4개와 최대 16-hop**이다. 전수조사가 없는 확장형 최종 알고리즘, 더 많은 역할, 번역 역할은 검토 단계이며 이 코드에 구현됐다고 해석하지 않는다. 기존 프로토콜 설명은 [실험 프로토콜](docs/EXPERIMENT_PROTOCOL.md)에 보존했다.

## 코드 구성

```text
pipeline/                 # 배치 편집, 데이터·평가, 실행·재개, 분석·그림
reference/                # A1–A7 스케줄러, 역할 프롬프트, 공개 데이터 다운로드
configs/datasets.json     # 데이터셋 revision과 표본 추출 조건
model_lock.json           # 편집 모델 checkpoint와 고정 revision
scripts/                  # 설치와 GPU 없이 실행하는 기능 검사
tests/                    # prefix/cache/분할/정보 접근 경계/재학습 검증
start.sh                  # SSH와 분리된 전체 실험 시작·재개
status.sh                 # 현재 단계와 진행 상태
docs/                     # 프로토콜과 실행 구조 설명
```

모델 가중치, 원시 데이터, 생성 문서, 기존 측정 결과·그림, 로컬 가상환경과 인증 정보는 업로드하지 않는다. 실행 시 생성되는 `data/`, `runs/`, `artifacts/`는 Git에서 제외한다.

## 환경 설치

Linux, Python 3.10–3.12, `python3-venv`와 CUDA 지원 NVIDIA GPU를 기준으로 한다. 실제 편집은 CUDA GPU를 요구한다. CPU 설치는 아래 기능 검사와 데이터 준비용이다.

```bash
git clone https://github.com/Doyoung619/Telephone-Game.git
cd Telephone-Game
bash scripts/setup.sh
source .venv/bin/activate
```

GPU 없이 코드부터 확인하려면 `bash scripts/setup.sh --cpu`로 설치한다. 설치 스크립트는 실험을 시작하지 않는다. 다른 가상환경을 사용한다면 shell 스크립트에 `TELEPHONE_PYTHON=/path/to/python`을 지정할 수 있다.

## 실험 없이 기능 검사

```bash
bash scripts/check.sh
```

이 명령은 통제된 synthetic fixture와 mock editor로 분할·정답/프롬프트 분리·64개 prefix·중단 후 재개·모델 호출 재사용·test fitting 차단을 검사한다. 스케줄러의 수학적 검증과 Python/shell 구문 검사도 포함한다. **모델 다운로드나 실제 LLM/GPU 추론은 하지 않는다.**

## 데이터와 모델 준비

```bash
# 다운로드 계획과 고정된 revision만 확인
python pipeline/prepare.py plan

# Synthetic 데이터, 공개 데이터 3종, 모델 캐시 및 CPU QA 감사 준비
python pipeline/prepare.py all --audit-qa
python pipeline/prepare.py check
```

준비 단계는 편집 실험을 실행하지 않는다. `--audit-qa`는 SQuAD calibration 자료에 한해 독립 QA 모델을 CPU에서 실행한다. 인터넷은 공개 가중치·데이터를 받는 데 사용하며 유료 inference API는 사용하지 않는다.

부분 준비도 가능하다.

```bash
python pipeline/prepare.py synthetic
python pipeline/prepare.py model
python pipeline/prepare.py external --audit-qa
# 필요한 Hugging Face 파일이 모두 캐시된 환경
python pipeline/prepare.py all --offline --audit-qa
```

기존 파일이나 evaluator가 바뀌었으면 덮어쓰지 않고 오류를 낸다. 다른 프로토콜/코드로 재실행할 때는 별도 체크아웃과 출력 경로를 사용한다.

## GPU 실험 실행·재개

아래 명령부터 실제 편집 모델을 실행한다.

```bash
# 작은 12문서 smoke만 확인하고 종료
python pipeline/run_all.py --run runs/smoke --smoke-only

# 전체 파일럿을 독립 세션에서 실행; SSH 종료 후에도 계속 진행
bash start.sh
bash status.sh
tail -f runs/main/runner.log
tail -f runs/main/finalizer.log
```

전체 실행은 평가기 검증 → smoke → 후보 비교·동결 → test → 16-hop 수명 → 이론 진단 → ablation → 외부 zero-shot → 비용 분석 → 외부 A1 재학습 → 보고서 생성 순서다. `start.sh`를 다시 호출하면 잠금과 체크포인트를 사용해 재개하며, 완료된 실험은 다시 시작하지 않는다.

외부 재학습은 기존 **A1 고정 프로파일**이다. `prepare.py external`이 외부 test 생성 전에 protocol과 fit ID를 등록한다. 메인 선택이 A1이 아닌 경우 이 확장은 자동으로 다른 알고리즘으로 바꾸지 않고 중단한다. 재학습을 생략하려면 다음 명령을 사용한다.

```bash
# 처음부터 외부 재학습을 제외하고 실행
bash start.sh --skip-external-refit
# 이미 주 실험이 완료됐다면 보고서만 생성
python pipeline/finalize.py --skip-external-refit
```

보고서 생성은 주 실험 `status=complete` 상태를 요구한다. 실행 단계와 코드별 역할은 [실행 구조](docs/CODE_GUIDE.md)를 참고한다.

## 생성되는 파일

| 경로 | 내용 |
|---|---|
| `data/paper/` | Synthetic public/private 데이터와 감사·분할 manifest |
| `data/external/` | 외부 데이터의 그룹 분할과 평가 범위 |
| `runs/main/journal.sqlite` | 생성 캐시, 단계별 문서/점수, 누적 편집 카운터 |
| `runs/main/selection/` | 후보 비교와 동결한 정책 |
| `runs/main/final/`, `lifetime/`, `theory/`, `ablation/` | 개별 문서 결과와 집계 |
| `runs/main/external/` | 데이터별 zero-shot·재학습 평가 및 동결 기록 |
| `runs/main/cost/` | 편집·평가·검색 비용과 실패 내역 |
| `artifacts/Review/` | PNG/PDF 핵심 그림과 전체 모음 |
| `runs/main/completion_manifest.json` | 전체 보고 완료 기록 |

## 측정과 해석

- Synthetic의 F는 규칙 평가기가 판정한 **11개 사실의 보존 비율**이다. bit 단위 정보량이나 상호정보량이 아니다.
- Q는 편집 요구 충족률이고 S는 F/Q의 조화평균이다. 평가기는 재서술·시제·사건 결합에 한계가 있으며, 알려진 fixture 통과가 일반적인 자연어 정확성을 보장하지 않는다.
- E2E/WebNLG는 제한된 관계/alias 보존 proxy, SQuAD는 독립 MiniLM의 추출형 QA 점수다. 외부 점수를 Synthetic S와 합산하지 않는다.
- 모든 후보가 공유한 전체 순열 은행은 기존 4-role 비교를 재현하기 위한 것이다. A2/A3의 DP와 A4/A5의 현재 상한 계산에도 지수적 비용이 남아 있다.
- 실행 환경에 맞춘 경로/준비/재개 코드 정리 내역은 [이식 내역](docs/PORTING_NOTES.md)에 기록했다. 새로운 GPU 벤치마크 결과는 이 저장소에 포함하지 않는다.
