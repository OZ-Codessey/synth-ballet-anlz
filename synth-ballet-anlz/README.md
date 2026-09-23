# 🩰 발레 훈련 일기 시계열 분석 

# Purpose & Data Nature (목적 및 데이터 성격)

이 데이터는 실제 관측 데이터가 아니라, 본 과정의 "AI 응용1(시계열 분석)"과 "AI 응용2(AI 비서 구축)" 두 미션에 연속으로 활용할 목적으로 설계·생성한 합성 데이터입니다. 서로 다른 생성형 AI 여러 개를 오가며 교차 검증(생성 → 구조·수치 검토 → 보완 → 재검증)한 뒤 확정했으며, 응용1에서는 이 데이터의 시계열 패턴(추세·분해·예측)을 분석하고, 응용2에서는 동일 데이터를 Firestore에 저장해 AI 비서가 참조하는 컨텍스트로 재사용합니다.

따라서 본 리포트의 목적은 "실제 발레 생리학을 규명"하는 것이 아니라, "시계열 데이터에서 패턴을 발견하고, 관찰(사실)과 해석(가설)을 구분해 서술하는 분석 방법론"을 시연하는 데 있습니다.

[**합성 데이터를 사용한 이유**] :  
시계열 분석을 위한 실제 발레 일기 데이터를 구할 수 없어 개인의 발레 루틴을 참조하여 직접 데이터를 생성함.  
합성 데이터는 검증 가능한 "정답"이 있어 생성형 AI의 잘못된 분석이나 주장을 반증하기 쉽고, 1년치 데이터를 즉시 확보 가능하며 민감정보 제약 없어 공개 배포 가능하다는 등 장점이 있습니다. 상세 근거는 REPORT.md 1장 참고.

## Research Basis (연구 근거)

데이터 설계 시 아래 두 편의 실제 스포츠과학 논문을 참고했습니다(정확한 계수·사건 날짜·회복 기간까지 논문 그대로는 아니며, 시뮬레이션 가정임을 명시합니다).

- Shaw, J. W. et al. (2020). *The Validity of the Session Rating of Perceived Exertion Method for Measuring Internal Training Load in Professional Classical Ballet Dancers.* Frontiers in Physiology, 11:480. https://doi.org/10.3389/fphys.2020.00480
- Lee, L. et al. (2017). *Injury Incidence, Dance Exposure and the Use of the Movement Competency Screen (MCS)...* International Journal of Sports Physical Therapy, 12(3), 352–370. https://pmc.ncbi.nlm.nih.gov/articles/PMC5455185/



## Project Overview (프로젝트 개요)

- **데이터 기간**: 2025-01-01 ~ 2025-12-31 (365일, 결측 없이 전수)
- **핵심 지표**: 를르베(Relevé) 균형 유지 시간(초), 피루엣(Pirouette) 회전 수, 그랑 제떼(Grand Jeté) 체공 높이(cm), 턴아웃(Turnout) 각도(도), 연습 시간(분), RPE, 세션 부하(AU), 수면 시간/질, 피로도, 통증 점수
- **특수 상황 태그(event_type)**: 평상시(normal) 외에 발목 부상(급성/완화/회복), 공연 주간, 여행, 경미한 질병, 집중 워크숍 등 8종
- **기초통계량**: 10개 변수의 n/평균/중앙값/표준편차/사분위수/최댓값을 정리했습니다(REPORT.md 4장). 연습시간·RPE·세션부하는 완전 휴식일(98일)이 섞여 있어, "전체 평균"과 "훈련일만의 평균"을 구분해서 봐야 합니다(전체 49.11분 vs 훈련일 67.13분).
- **결측치 처리**: `measurement_status`(full/partial/not_measured)로 결측 사유를 구분, 임의 보간 없이 pairwise deletion 원칙 적용(STL/ARIMA만 예외적으로 선형 보간).
- **이상치 처리**: IQR 기준으로 탐지한 이상치는 전부 `event_type`(부상·워크숍·공연주간)으로 설명되어 삭제하지 않았습니다. "통계적으로 이상하다 ≠ 잘못된 데이터"라는 원칙을 일관 적용했습니다(REPORT.md 4장).
- **분석 기법**: 7일 이동평균(SMA), 구간별 통계(부상 전/중/회복) / STL 분해, ARIMA(1,1,1) 예측 / ACWR(급성:만성 부하비율), 지표별 예측모델(5-fold 교차검증), 이월효과 분석 

## Analysis Questions (5) → Answer Mapping (분석 질문 → 답변 매핑)

REPORT.md 3장에 5개 질문과, 각 질문에 답한 시각화·인사이트를 매핑한 표가 있습니다(장기추세 / 요일패턴·지연효과 / 부상영향 / 변수간상관 / 훈련부하 예측가능성).

---

###  — Dashboard (URL 형식)- Bonus 1

> **대시보드 URL**: https://claude.ai/artifact/U2p7EsxkYFcG6tNNbemUfd
>
> * **실시간 탐색**: 기간(시작일/종료일)·훈련 상황(`event_type`) 필터와 지표 선택 드롭다운으로 트렌드·요일별 패턴·상관관계 히트맵·훈련부하(AU) 추이를 실시간 탐색할 수 있습니다.
> * **예측 모델 신뢰도 카드**: 4개 지표 중 원하는 것을 선택해 MAE·베이스라인·R²를 비교할 수 있습니다.
> * **부상 단계 비교 차트**: 부상 시점이 고정된 사건이므로 항상 전체 기간 기준으로 표시됩니다.

### — Time Series Deep Dive (STL+ARIMA)- Bonus 2

> * **STL 시계열 분해**: 추세 강도($F_t$)=0.608, 계절성 강도($F_s$)=0.204. 부상의 영향은 잔차가 아니라 추세 성분의 이동으로 포착됨.
> * **ARIMA(1,1,1) 예측**: $AR(1)=0.28$, $MA(1)=-0.85$. 백테스트 MAE 2.12cm (MAPE 7.13%).
> 
>>> *※ 상세 설명과 그래프 주석은 REPORT.md 시각화 ⑤, ⑥ 참고.*

###  — Workload-Based Injury & Prediction Analysis (추가 확장)- Extended

> * **ACWR(급성:만성 부하비율)**: 부상 전날 0.95로 과부하 기준(1.5) 미만, 시간적 선후관계와 인과관계를 명확히 구분.
> * **지표별 예측모델 R² 비교**: 를르베 0.499 vs 턴아웃 0.127 (지표 생리학적 특성에 따른 예측력 차이 확인).
> * **훈련부하 지연 효과**: 당일 $r=0.387$ → 익일 $0.325$ → 이틀 뒤 $0.144$ (부하의 영향이 1–2일간 이월됨).
> * **구조 변화(부상) 시 단순예측 실패 검증**: 정상 구간 MAE 2.19cm → 부상 구간 4.46cm (약 2배 폭증).
> 
>>>*※ 상세 내용은 REPORT.md 시각화 ⑦, ⑧, ⑨ 및 8장 참고.*


<bf>
<bf>
<bf>


# How to Run

`data/ballet_training_cleaned_365.csv`로부터 `graphs/`의 9개 시각화를 재현하는 `analysis.py`를 포함합니다( 직접 실행하지 않아도 리포트와 이미지만으로 결과 확인이 가능합니다).

```bash
pip3 install -r requirements.txt
python3 analysis.py
```



## Verification & Reproducibility Principles (검증·재현 원칙)

이 프로젝트는 이전 세션(180일 시범 데이터)에서 생성형 AI가 생성한 분석 주장 중 실제 데이터와 어긋나는 내용(허위 상관관계 방향, 존재하지 않는 자기상관계수 값, 근거 없는 회귀모델 정밀도 주장 등)을 재현 검증으로 발견하고 수정하였습니다. 외부 생성형 AI가 별도로 작성한 참고 리포트의 인용 논문 3건 중 1건도 동일한 방식으로 검증해 오류를 발견. 대시보드 개발 중에도 JSON 직렬화 버그(결측치가 `null`이 아닌 `NaN` 리터럴로 출력되어 히트맵이 전부 깨지는 문제)의 원인을 찾고 수정. **특정 AI 하나가 항상 옳았던 것은 아니며, 여러 생성형 AI의 산출물을 서로 교차 대조하며 오류를 찾아나가는 과정을 반복.** 상세 기록은 REPORT.md의 "AI 사용 로그" 섹션을 참고하세요.

## Limitations

- 장기 성장 추세가 데이터 설계에 명시적으로 반영되지 않아 1년 트렌드가 평평하게 나타남.혹은 숙련된 경력 무용수의 경우 1년간의 연습기간 동안 크게 실력이 늘거나 주는 등의 변동성이 없다는 것으로 설계했을 가능성도 있음.
- 7개 변수 중 6개(연습시간·피로도·4개 기술지표)는 생리학적으로 타당한 상관관계를 보였으나, 수면시간만 예외적으로 다른 변수와 거의 상관이 없었음. 이는 실제 스포츠과학에서도 보고되는 현상일 수 있으나, 변수 간 인과관계 해석은 참고용으로만 활용해야 함.
- 부상 회복은 완전한 원상복귀가 아닌 근접 회복 수준으로 설계됨. 이것이 장기 성장 추세에 영향을 미친 원인일 수도 있음.
- 훈련부하 기반 예측 모델의 설명력(R²)은 지표별로 0.13~ 약 0.50 까지 편차가 크므로, 모델 신뢰도를 지표 구분 없이 일반화해서는 안 됨.
- 합성 데이터이므로 실제 발레 생리학을 일반화하는 근거로 사용하기 어려움.

## Directory Structure 
```
synth-ballet-anlz/
|-- data/
|   |-- ballet_training_cleaned_365.csv      # 365일 정제 데이터셋 (18개 변수)
|-- graphs/                                  # analysis.py 실행 시 자동 생성되는 분석 차트 (01~11)
|   |-- 01_jete_trend_sma.png                # (필수 1) 그랑 제떼 트렌드 및 7일 이동평균선
|   |-- 02_weekly_seasonality.png            # (필수 2) 요일별 평균 훈련시간 vs 피로도
|   |-- 03_correlation_heatmap.png           # (필수 3) 지표 간 상관관계 히트맵
|   |-- 04_injury_phase_comparison.png       # (필수 4) 부상 단계별 성과 비교 (지수화: 부상 전=100)
|   |-- 05_stl_decomposition.png             # (보너스 1) STL 시계열 분해 (추세/계절성/잔차)
|   |-- 06_arima_forecast.png                # (보너스 2) ARIMA(1,1,1) 14일 단기 예측
|   |-- 07_workload_injury_precursor.png     # (추가 1) 훈련 부하(AU) & 부상 전조 신호 (ACWR 검증)
|   |-- 08_prediction_r2_comparison.png      # (추가 2) 지표별 예측 설명력(R²) 비교
|   |-- 09_lag_effect.png                    # (추가 3) 훈련 부하의 지연효과 (Lag Effect)
|   |-- 10_scatter_au_fatigue.png            # (추가 4) 훈련 부하(AU) vs 피로도 상황별 산점도
|   |-- 11_boxplot_fatigue_by_event.png      # (추가 5) 훈련 상황(event_type)별 피로도 박스플롯
|-- images/                                  # 시각화 이미지 설명용 파일 11개 (Imagen 생성본)
|-- analysis.py                              # 11종 시각화 파이프라인 및 통계 검증 전체 실행 스크립트
|-- VISUALIZATION_GUIDE.md                   # 시각화 상세 설명서 (11종 차트별 관찰-해석-한계-행동 매뉴얼)
|-- REPORT.md                                # 최종 데이터 분석 및 생체역학 인사이트 종합 보고서
|-- README.md                                # 프로젝트 소개, 실행 방법 및 대시보드 배포 링크
|-- requirements.txt                         # 실행 의존성 패키지 목록  
```
