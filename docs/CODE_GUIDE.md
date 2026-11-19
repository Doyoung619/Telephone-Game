# 실행 코드 안내

## 입력 준비

`pipeline/prepare.py`가 고정된 revision의 자산과 데이터 분할을 준비한다.

- `synthetic`: `data_eval.prepare_data()`로 200문서와 private facts를 생성하고 source/witness와 통제된 evaluator 사례를 검사한다.
- `model`: 루트 `model_lock.json`의 편집 모델을 Hugging Face 캐시에 다운로드한다. GPU 모델을 로딩하지 않는다.
- `external`: `configs/datasets.json`의 Parquet 파일을 `reference/download_public_datasets.py`로 받아 원문 그룹 단위 자료를 구성한다. 이후 `external_data.prepare_external()`로 정규화한다.
- 외부 A1 프로파일은 fit ID, 목적함수, tie break를 외부 test 실행 전에 `runs/main/external/refit_preregistration.json`에 기록한다. 이미 외부 결과가 있으면 사후 등록을 거부한다.
- `--audit-qa`: SQuAD calibration 원문과 no-answer fixture를 CPU QA로 감사한다.

준비 파일은 저장소에 포함되지 않는다. 기존 자료의 hash/revision이 다르면 별도 실험 디렉터리를 사용해야 한다.

## 편집과 저장

`pipeline/engine.py`는 `reference/local_editor.py`의 공통·역할별 프롬프트를 사용한다. 입력에는 현재 문서, 역할, 고정 단어 수 조건만 들어간다. 실제 생성은 FP16 Qwen의 batch greedy decoding이다.

`Journal`은 SQLite WAL 모드로 모델 응답 캐시와 상태를 저장한다. 캐시 키는 prompt와 모델/생성 조건으로 구성하며 모델 로딩 시간처럼 매 실행 바뀌는 값은 제외한다. 실패한 출력은 정상 응답 캐시에 저장하지 않는다. 편집 카운터도 캐시와 함께 체크포인트한다.

`pipeline/run_all.py`의 `Study`는 public 문서와 private 정답을 전달 경계에서 분리하고, prefix/chain을 실행해 매 단계 점수를 저장한다. 재개할 때 완료 상태를 읽고, 원래 카운터와 단계 시간을 유지한다. 모델·editor·evaluator 변경은 기존 run과 혼합하지 않는다.

## 평가와 순서 선택

| 코드 | 역할 |
|---|---|
| `pipeline/data_eval.py` | Synthetic 데이터 생성, 규칙 기반 사실/편집 평가 및 audit |
| `pipeline/analysis.py` | 후보 fitting, 독립 selection, 정책 동결, 통계와 그림 |
| `reference/schedulers.py` | NumPy A1–A7 순서 선택과 제한된 검색 certificate |
| `pipeline/external_data.py` | E2E/WebNLG proxy 및 pinned CPU MiniLM QA |
| `pipeline/refit_external.py` | 기존 A1의 외부 fit-only 재학습 및 별도 test 비교 |

`data_eval.py` 안의 일반 외부 점수는 생성 journal용 임시 진단이다. 외부 최종 결과에는 `external_data.analyze_outputs()`가 다시 계산한 데이터별 지표를 사용한다. 외부 refit의 중간 prefix는 평가하지 않고 terminal bank를 별도 평가한다.

원자 사실의 `standalone/embedded/LOST` 상태는 A4/A5의 작은 전이모형에 사용한다. 모델의 검색 gap이 실제 언어의 정보량·공동 점수 S를 보장하는 것은 아니다.

## 실행 프로세스

`start.sh`는 준비 상태를 검사하고 supervisor와 finalizer를 terminal 없이 독립 세션으로 띄운다.

1. `pipeline/supervise.py`: 독점 잠금을 잡고 주 실험을 실행한다. 예상치 못한 실패에 한해 최대 세 번 재시도한다.
2. `pipeline/run_all.py`: smoke부터 외부 zero-shot과 주 비용 분석까지 수행한다.
3. `pipeline/finalize.py --wait`: 주 실험 완료 후 지원되는 외부 A1 프로파일을 재학습하고 보고서를 생성한다.
4. `pipeline/refit_external.py`: 같은 GPU 잠금을 획득한 후 실행한다. Calibration-only 순서를 동결한 뒤 test를 실행한다.

호스트 재부팅 뒤에는 `bash start.sh`를 다시 실행한다. GPU 실험이 도는 동안 코드, 데이터, 모델 revision을 수정하지 않는다.

## 보고서

- `pipeline/progress.py`: 현재 상태와 evaluator audit 그림.
- `pipeline/watch_progress.py`: 직접 실행하면 진행 그림을 주기적으로 갱신하는 선택적 보조 프로세스.
- `pipeline/external_summary_plot.py`: 서로 다른 외부 지표를 분리한 비교 그림.
- `pipeline/finalize.py`: 저장된 관측치로 최종 보고서·비용·그림 모음을 생성한다.

기존 연구자가 별도로 작성한 정성 검토의 실제 문장 사례와 그 전용 렌더러는 결과 자료에 해당하므로 업로드하지 않았다. 자동 생성된 그림을 인간 정답 검토로 해석하지 않는다.

## 제한

현재 runner는 네 역할과 해당 프로파일에 맞춘 구현이다. YAML 설정만 바꾸어 임의 역할 수·번역·긴 체인을 지원하는 범용 프레임워크가 아니다. 그 확장은 프로토콜에 기록한 미실행 요건을 따른 별도 설계/검증이 필요하다.
