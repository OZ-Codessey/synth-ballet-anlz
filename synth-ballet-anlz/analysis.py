# -*- coding: utf-8 -*-
"""
발레 훈련 365일 시계열 분석 파이프라인
- 데이터 정제 요약, 시각화 11종(필수4+보너스7), STL/ARIMA(보너스), 인사이트용 통계 출력
- REPORT.md에 실린 그래프와 동일하게 재현되도록 스타일을 고정한다.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_prop = font_manager.FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

def kfont(size=11, weight='normal'):
    return font_manager.FontProperties(fname=FONT_PATH, size=size, weight=weight)

# ---------- 공통 색상 ----------
BG = '#FAFAF8'          # 거의 흰색에 가까운 밝은 배경(①②③④⑤⑥⑦⑧에 사용)
PANEL_BG = '#FBF7FA'    # 연한 라벤더-핑크 배경(⑨ 박스플롯에 사용)
BLUE = '#1f77b4'
BLUE_DARK = '#2059A6'
BLUE_LIGHT = '#AFC9EC'
GREY = '#8C8C8C'
RED = '#D62728'
ROSE = '#D88CA3'
LAVENDER = '#9B84C4'
SAGE = '#8FB9A8'
GOLD = '#E3B77D'
NEUTRAL_GREY = '#C4C2B4'   # R² 비교 막대 등 비강조 회색
EVENT_COLORS = {
    'normal': '#7B5EA7', 'ankle_injury_acute': '#E38294', 'ankle_injury_modified': '#F0A8BA',
    'ankle_injury_recovery': '#C9C2DC', 'performance_week': '#D9A441', 'travel': '#EEA8BB',
    'mild_illness': '#8FB9A8', 'intensive_workshop': '#EEDDC0'
}
EVENT_KLABEL = {'normal':'평상시','ankle_injury_acute':'부상(급성)','ankle_injury_modified':'부상(완화)',
    'ankle_injury_recovery':'부상(회복)','performance_week':'공연주간','travel':'여행',
    'mild_illness':'경미한질병','intensive_workshop':'집중워크숍'}
sns.set_theme(style='whitegrid', rc={
    'axes.facecolor': BG, 'figure.facecolor': BG,
    'grid.color': '#E7E7E3', 'axes.edgecolor': '#E7E7E3',
    'axes.labelcolor': '#333333', 'text.color': '#222222',
    'xtick.color': '#555555', 'ytick.color': '#555555',
})

def style_axes(ax, title=None, title_size=15):
    if title: ax.set_title(title, fontproperties=kfont(title_size,'bold'), pad=14, color='#111111')
    for lbl in ax.get_xticklabels()+ax.get_yticklabels():
        lbl.set_fontproperties(kfont(10))
    sns.despine(ax=ax)

df = pd.read_csv('data/ballet_training_cleaned_365.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df['event_kr'] = df['event_type'].map(EVENT_KLABEL)

# ---------- 1) 데이터 정제 요약 ----------
print("=== 데이터 정제 요약 ===")
print("총 일수:", len(df))
print("결측치(핵심 기술지표):")
for c in ['releve_balance_sec','pirouette_turns','jete_height_cm','turnout_angle_deg']:
    print(f"  {c}: {df[c].isna().sum()}일 결측 ({df[c].isna().mean()*100:.1f}%)")
print("measurement_status 분포:")
print(df['measurement_status'].value_counts())
print()

# ---------- 2) 시각화 ① 그랑 제떼 트렌드 + 7일 이동평균 (필수) ----------
df['jete_sma7'] = df['jete_height_cm'].rolling(7, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(11,4.5))
ax.plot(df['date'], df['jete_height_cm'], color=GREY, linewidth=0.7, alpha=0.7, label='일별 값')
ax.plot(df['date'], df['jete_sma7'], color=BLUE, linewidth=2.3, label='7일 이동평균')
ax.axvspan(pd.Timestamp('2025-08-18'), pd.Timestamp('2025-08-25'), color='#F4B8C4', alpha=0.5, label='부상(급성)')
ax.axvspan(pd.Timestamp('2025-08-26'), pd.Timestamp('2025-09-26'), color='#F5D9A8', alpha=0.5, label='회복기')
style_axes(ax, '그랑 제떼 높이 트렌드 + 7일 이동평균 (2025)')
ax.set_ylabel('높이(cm)', fontproperties=kfont(11))
ax.legend(prop=kfont(10), frameon=False, loc='upper right')
plt.tight_layout()
plt.savefig('graphs/01_jete_trend_sma.png', dpi=140, facecolor=BG)
plt.close()

# ---------- 3) 시각화 ② 요일별 연습시간 vs 피로도 (필수) ----------
order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
klabel = {'Mon':'월','Tue':'화','Wed':'수','Thu':'목','Fri':'금','Sat':'토','Sun':'일'}
wk = df.groupby('weekday').agg(practice=('practice_time_min','mean'), fatigue=('fatigue_score_1_10','mean')).reindex(order)
fig, ax1 = plt.subplots(figsize=(9,4.8))
ax1.bar([klabel[w] for w in order], wk['practice'], width=0.55, color=ROSE, alpha=0.55, edgecolor=ROSE, linewidth=1.2, label='평균 연습시간(분) — 왼쪽축')
ax1.set_ylabel('연습시간(분) · 왼쪽축', fontproperties=kfont(10), color='#B5678A')
ax2 = ax1.twinx()
ax2.plot([klabel[w] for w in order], wk['fatigue'], color=LAVENDER, marker='o', markersize=5,
         linewidth=1.4, linestyle=(0,(4,2)), label='평균 피로도 — 오른쪽축')
ax2.set_ylabel('피로도(1-10) · 오른쪽축', fontproperties=kfont(10), color='#7A5F9B')
ax2.grid(False)
style_axes(ax1, '요일별 평균 연습시간 vs 피로도')
ax1.set_xlabel('요일', fontproperties=kfont(10))
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, prop=kfont(9), frameon=False, loc='upper left')
for lbl in ax2.get_yticklabels(): lbl.set_fontproperties(kfont(9))
plt.tight_layout()
plt.savefig('graphs/02_weekly_seasonality.png', dpi=140, facecolor=BG)
plt.close()

# ---------- 4) 시각화 ③ 상관관계 히트맵 ----------
cols = ['practice_time_min','fatigue_score_1_10','sleep_hours','releve_balance_sec','pirouette_turns','jete_height_cm','turnout_angle_deg']
short = {'practice_time_min':'연습시간','fatigue_score_1_10':'피로도','sleep_hours':'수면시간','releve_balance_sec':'를르베','pirouette_turns':'피루엣','jete_height_cm':'제떼','turnout_angle_deg':'턴아웃'}
corr = df[cols].corr()
corr.index = [short[c] for c in cols]; corr.columns = [short[c] for c in cols]
fig, ax = plt.subplots(figsize=(7.5,6.3))
cmap = sns.diverging_palette(260, 340, s=55, l=65, as_cmap=True)  # 라벤더(음)~로즈(양)
sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, center=0, vmin=-1, vmax=1,
            linewidths=1.5, linecolor=BG, square=True, cbar_kws={'shrink':0.8, 'label':'상관계수'},
            annot_kws={'fontproperties':kfont(9), 'color':'#222222'}, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=kfont(10), rotation=40, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), fontproperties=kfont(10), rotation=0)
cbar = ax.collections[0].colorbar
cbar.set_label('상관계수 (좌: 음의 상관 · 우: 양의 상관)', fontproperties=kfont(9))
for t in cbar.ax.get_yticklabels(): t.set_fontproperties(kfont(9))
ax.set_title('지표 간 상관관계 히트맵', fontproperties=kfont(15,'bold'), pad=14)
plt.tight_layout()
plt.savefig('graphs/03_correlation_heatmap.png', dpi=140, facecolor=BG)
plt.close()
print("=== 상관관계 (히트맵 수치) ===")
print(corr.round(3))
print()

# ---------- 5) 시각화 ④ 부상 전/중/회복 비교 (필수 - 구간별 통계) ----------
pre = df[df['date']<'2025-08-18']
during = df[df['event_type'].isin(['ankle_injury_acute','ankle_injury_modified'])]
recov = df[df['event_type']=='ankle_injury_recovery']
post = df[df['date']>='2025-09-27']
phase_metrics = ['releve_balance_sec','turnout_angle_deg','jete_height_cm']
phase_label = {'releve_balance_sec':'를르베(초)','turnout_angle_deg':'턴아웃(도)','jete_height_cm':'제떼(cm)'}
groups = {'부상 전': pre, '부상 중': during, '회복기': recov, '회복 완료 후': post}
# 단위가 서로 다른(초/도/cm) 지표를 그대로 비교하면 값의 크기가 큰 턴아웃(도)이
# 시각적으로 압도해 실제 변화 폭(%)이 왜곡되어 보인다. 따라서 "부상 전 평균=100"으로
# 지수화(정규화)해 단위 차이를 없애고 상대적 변화 폭을 공정하게 비교한다.
baseline = {m: pre[m].mean() for m in phase_metrics}
fig, ax = plt.subplots(figsize=(10,4.5))
x = np.arange(len(phase_metrics))
width = 0.19
phase_colors = ['#D9CFE0', ROSE, GOLD, SAGE]
for i, (gname, gdf) in enumerate(groups.items()):
    idx_vals = [(gdf[m].mean()/baseline[m])*100 for m in phase_metrics]
    bars = ax.bar(x + i*width, idx_vals, width, label=f"{gname}(n={len(gdf)})",
                   color=phase_colors[i], alpha=0.6, edgecolor=phase_colors[i], linewidth=1.3)
    for b, v in zip(bars, idx_vals):
        ax.text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.0f}", ha='center', fontproperties=kfont(8), color='#222222')
ax.axhline(100, color='#333333', linestyle=(0,(5,3)), linewidth=1, label='부상 전=100 기준선')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels([phase_label[m] for m in phase_metrics], fontproperties=kfont(11))
ax.set_ylabel('부상 전 대비 지수 (부상 전=100)', fontproperties=kfont(10))
style_axes(ax, '부상 단계별 성과 비교 — 단위 정규화(지수화)')
ax.legend(prop=kfont(9))
plt.tight_layout()
plt.savefig('graphs/04_injury_phase_comparison.png', dpi=140, facecolor=BG)
plt.close()
print("=== 부상 단계별 평균(원본값) ===")
for gname, gdf in groups.items():
    print(gname, {m: round(gdf[m].mean(),2) for m in phase_metrics})
print("=== 부상 단계별 지수(부상 전=100) ===")
for gname, gdf in groups.items():
    print(gname, {m: round((gdf[m].mean()/baseline[m])*100,1) for m in phase_metrics})
print()

# ---------- 6) [보너스] STL 시계열 분해 ----------
from statsmodels.tsa.seasonal import STL
ts = df.set_index('date')['jete_height_cm'].interpolate(limit_direction='both')
stl = STL(ts, period=7, robust=True).fit()

var_resid = stl.resid.var()
Fs = max(0, 1 - var_resid/(stl.seasonal+stl.resid).var())
Ft = max(0, 1 - var_resid/(stl.trend+stl.resid).var())
print(f"계절성 강도(Fs): {Fs:.3f} | 추세 강도(Ft): {Ft:.3f}")

fig, axes = plt.subplots(4,1, figsize=(11,9), sharex=True)
comps = [ts, stl.trend, stl.seasonal, stl.resid]
names = ['관측값(Observed)','추세(Trend)','계절성(Seasonal, 7일주기)','잔차(Residual)']
for a, comp, name in zip(axes, comps, names):
    a.plot(comp, color=BLUE, linewidth=1.1)
    a.set_ylabel(name, fontproperties=kfont(11))
    for lbl in a.get_yticklabels(): lbl.set_fontproperties(kfont(9))
    sns.despine(ax=a)

axes[0].axvspan(pd.Timestamp('2025-08-18'), pd.Timestamp('2025-09-26'), color='#F4B8C4', alpha=0.45)
axes[0].annotate('부상 구간\n(관측값 급락)', xy=(pd.Timestamp('2025-08-23'), 22),
                  xytext=(pd.Timestamp('2025-10-20'), 20), fontproperties=kfont(10,'bold'), color=RED,
                  arrowprops=dict(arrowstyle='->', color=RED))
axes[1].axvspan(pd.Timestamp('2025-08-18'), pd.Timestamp('2025-09-26'), color='#F4B8C4', alpha=0.35)
axes[1].annotate(f'추세 강도(Ft)={Ft:.3f}\n→ 관측값 변동의 상당 부분이\n장기 추세로 설명됨', xy=(pd.Timestamp('2025-08-23'), stl.trend.loc['2025-08-23']),
                  xytext=(pd.Timestamp('2025-10-10'), stl.trend.min()+1), fontproperties=kfont(10), color='#222222',
                  arrowprops=dict(arrowstyle='->', color='#222222'))
axes[2].annotate(f'계절성 강도(Fs)={Fs:.3f}\n7일 주기 반복 (목요일에 최고,\n월요일에 최저)', xy=(pd.Timestamp('2025-05-01'), stl.seasonal.loc['2025-05-01']),
                  xytext=(pd.Timestamp('2025-02-01'), stl.seasonal.max()+0.3), fontproperties=kfont(10), color='#222222',
                  arrowprops=dict(arrowstyle='->', color='#222222'))
top_resid_dates = stl.resid.abs().sort_values(ascending=False).head(3).index
for d in top_resid_dates:
    axes[3].annotate(f"{d.strftime('%m/%d')}", xy=(d, stl.resid.loc[d]),
                      xytext=(d, stl.resid.loc[d]+(1.5 if stl.resid.loc[d]>0 else -2)),
                      fontproperties=kfont(9,'bold'), ha='center', color='#E06C2C')

axes[0].set_title('STL 시계열 분해 — 그랑 제떼 높이 (7일 주기)', fontproperties=kfont(15,'bold'))
for lbl in axes[-1].get_xticklabels(): lbl.set_fontproperties(kfont(10))
plt.tight_layout()
plt.savefig('graphs/05_stl_decomposition.png', dpi=140, facecolor=BG)
plt.close()

resid_std = stl.resid.std()
top_resid = stl.resid.abs().sort_values(ascending=False).head(5)
print("=== STL 잔차 상위 5개(이상치 후보) ===")
print(top_resid.round(2))
print()

# ---------- 7) [보너스] ARIMA 14일 예측 ----------
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(ts, order=(1,1,1)).fit()
fc = model.get_forecast(steps=14)
fc_mean = fc.predicted_mean
fc_ci = fc.conf_int(alpha=0.05)

fig, ax = plt.subplots(figsize=(11,5))
ax.plot(ts.index[-60:], ts.values[-60:], color=ROSE, linewidth=1.3, alpha=0.85, label='최근 60일 실측(cm)')
ax.plot(fc_mean.index, fc_mean.values, color='#B5678A', linewidth=2.0, linestyle=(0,(5,2)), label='ARIMA(1,1,1) 예측(cm)')
ax.fill_between(fc_mean.index, fc_ci.iloc[:,0], fc_ci.iloc[:,1], color=ROSE, alpha=0.22, label='95% 신뢰구간')
ax.axvline(ts.index[-1], color=LAVENDER, linestyle=(0,(2,2)), linewidth=1.2)
ax.annotate('예측 시작점\n(마지막 실측일)', xy=(ts.index[-1], ts.values[-1]),
            xytext=(ts.index[-1]-pd.Timedelta(days=18), ts.values[-1]-4),
            fontproperties=kfont(10), color='#7A5F9B', arrowprops=dict(arrowstyle='->', color=LAVENDER))
ax.annotate(f'AR(1)={model.params["ar.L1"]:.2f}, MA(1)={model.params["ma.L1"]:.2f}\n신뢰구간이 갈수록 넓어짐\n(먼 미래일수록 불확실)',
            xy=(fc_mean.index[-1], fc_ci.iloc[-1,1]), xytext=(fc_mean.index[3], fc_ci.iloc[-1,1]+1),
            fontproperties=kfont(10), color='#222222')
style_axes(ax, 'ARIMA(1,1,1) 14일 예측 — 그랑 제떼 높이')
ax.set_ylabel('그랑 제떼 높이(cm) — 세로축', fontproperties=kfont(10))
ax.set_xlabel('날짜 — 가로축', fontproperties=kfont(10))
ax.legend(prop=kfont(9), frameon=False)
plt.tight_layout()
plt.savefig('graphs/06_arima_forecast.png', dpi=140, facecolor=BG)
plt.close()

print("=== ARIMA 예측 첫 5일 ===")
print(fc_mean.head())
print()
print("=== 핵심 통계 요약 ===")
print("전체 기간 jete 평균/표준편차:", round(ts.mean(),2), round(ts.std(),2))
print("부상 전 평균:", round(pre['jete_height_cm'].mean(),2))
print("연습시간-피로도 상관:", round(df['practice_time_min'].corr(df['fatigue_score_1_10']),3))
print("피로도-수면시간 상관:", round(df['fatigue_score_1_10'].corr(df['sleep_hours']),3))
print("Done.")

# ---------- 8) [추가] 훈련부하(AU) & 부상 전조 추적 ----------
d2 = df.copy()
d2['au_sma7'] = d2['session_load_au'].rolling(7, min_periods=1).mean()
FATIGUE_THRESHOLD = 6.5  # 스포츠과학에서 흔히 쓰이는 "중간-높은 피로" 경계값(10점 척도 기준)을 참고선으로 표시
fig, axes = plt.subplots(2,1, figsize=(11,7), sharex=True)
axes[0].bar(d2['date'], d2['session_load_au'], color=BLUE_LIGHT, alpha=0.9, width=0.9, label='일별 세션 부하(AU)')
axes[0].plot(d2['date'], d2['au_sma7'], color='#000000', linewidth=1.6, label='7일 이동평균')
axes[0].axvline(pd.Timestamp('2025-08-18'), color=RED, linestyle='--', linewidth=1.4, label='부상 발생(8/18)')
axes[0].set_ylabel('AU', fontproperties=kfont(12))
style_axes(axes[0], '훈련부하(AU) 추이와 부상 전조 신호')
axes[0].legend(prop=kfont(10))

axes[1].plot(d2['date'], d2['fatigue_score_1_10'], color=BLUE, marker='.', markersize=3, linewidth=1.0, label='피로도(1-10)')
axes[1].plot(d2['date'], d2['pain_score_0_10'], color=RED, marker='.', markersize=3, linewidth=1.0, label='통증 점수(0-10)')
axes[1].axhline(FATIGUE_THRESHOLD, color=RED, linestyle=':', linewidth=1.3, label=f'피로 임계선({FATIGUE_THRESHOLD})')
axes[1].axvspan(pd.Timestamp('2025-08-11'), pd.Timestamp('2025-08-17'), color=GOLD, alpha=0.3, label='전조 구간')
axes[1].axvspan(pd.Timestamp('2025-08-18'), pd.Timestamp('2025-08-25'), color=ROSE, alpha=0.3, label='급성 부상 구간')
axes[1].set_ylabel('점수', fontproperties=kfont(12))
sns.despine(ax=axes[1])
axes[1].legend(prop=kfont(9), ncol=2, loc='upper left')
for a in axes:
    for lbl in a.get_yticklabels(): lbl.set_fontproperties(kfont(10))
for lbl in axes[-1].get_xticklabels(): lbl.set_fontproperties(kfont(10))
plt.tight_layout()
plt.savefig('graphs/07_workload_injury_precursor.png', dpi=140, facecolor=BG)
plt.close()

# ACWR(급성:만성 부하비율) 검증
d2['acute_7d'] = d2['session_load_au'].rolling(7, min_periods=1).sum()
d2['chronic_28d_avg_weekly'] = d2['session_load_au'].rolling(28, min_periods=7).sum() / 4
d2['acwr'] = d2['acute_7d'] / d2['chronic_28d_avg_weekly']
print("=== ACWR 검증 (부상 전날 8/17) ===")
print(d2[d2['date']==pd.Timestamp('2025-08-17')][['date','acwr']])
print("전체 365일 중 ACWR>=1.5(과부하 기준) 일수:", (d2['acwr']>=1.5).sum())
print(f"피로 임계선({FATIGUE_THRESHOLD}) 초과 일수:", (d2['fatigue_score_1_10']>=FATIGUE_THRESHOLD).sum())

# ---------- 9) [추가] 예측모델 신뢰도 검증 (MAE, 베이스라인 비교, R^2) ----------
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

feat = ['practice_time_min','rpe_0_10','session_load_au','fatigue_score_1_10','sleep_hours','sleep_quality_1_5','pain_score_0_10']
dd = df.dropna(subset=['jete_height_cm']+feat).reset_index(drop=True)
X, y = dd[feat].values, dd['jete_height_cm'].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
maes, r2s, base_maes = [], [], []
for tr, te in kf.split(X):
    m = LinearRegression().fit(X[tr], y[tr])
    pred = m.predict(X[te])
    maes.append(mean_absolute_error(y[te], pred))
    r2s.append(r2_score(y[te], pred))
    base_maes.append(mean_absolute_error(y[te], [y[tr].mean()]*len(te)))

print("\n=== 다중회귀모델 신뢰도 (5-fold 교차검증) ===")
print("모델 MAE 평균:", round(np.mean(maes),3), "| 베이스라인(평균예측) MAE 평균:", round(np.mean(base_maes),3))
print("R^2 평균:", round(np.mean(r2s),3))

# ---------- 10) [추가] 지표별 예측 설명력(R^2) 비교 ----------
targets2 = ['jete_height_cm','releve_balance_sec','pirouette_turns','turnout_angle_deg']
tlabel2 = {'jete_height_cm':'그랑제떼','releve_balance_sec':'를르베','pirouette_turns':'피루엣','turnout_angle_deg':'턴아웃'}
res2 = {}
for t in targets2:
    dd2 = df.dropna(subset=[t]+feat).reset_index(drop=True)
    X2, y2 = dd2[feat].values, dd2[t].values
    m2, b2, r2_ = [], [], []
    for tr, te in kf.split(X2):
        mod = LinearRegression().fit(X2[tr], y2[tr])
        p = mod.predict(X2[te])
        m2.append(mean_absolute_error(y2[te], p))
        b2.append(mean_absolute_error(y2[te], [y2[tr].mean()]*len(te)))
        r2_.append(r2_score(y2[te], p))
    res2[t] = dict(mae=np.mean(m2), base=np.mean(b2), r2=np.mean(r2_))

fig, ax = plt.subplots(figsize=(9,4.5))
xs = np.arange(len(targets2))
bar_colors = [BLUE if res2[t]['r2']>=0.3 else NEUTRAL_GREY for t in targets2]
bars = ax.bar(xs, [res2[t]['r2'] for t in targets2], color=bar_colors, width=0.5)
ax.axhline(0.3, color='#333333', linestyle=(0,(1,1.5)), linewidth=1.1, label='참고선(R²=0.3)')
ax.set_xticks(xs); ax.set_xticklabels([tlabel2[t] for t in targets2], fontproperties=kfont(12))
ax.set_ylabel('R²', fontproperties=kfont(13))
style_axes(ax, '지표별 훈련부하 기반 예측 설명력(R²) 비교')
for i,t in enumerate(targets2):
    ax.text(i, res2[t]['r2']+0.012, f"{res2[t]['r2']:.3f}", ha='center', fontproperties=kfont(11), color='#222222')
ax.legend(prop=kfont(10), frameon=False, loc='upper right')
plt.tight_layout()
plt.savefig('graphs/08_prediction_r2_comparison.png', dpi=140, facecolor=BG)
plt.close()
print("\n=== 지표별 예측 설명력 ===")
for t in targets2: print(t, {k:round(v,3) for k,v in res2[t].items()})

# ---------- 11) [추가] 훈련부하의 이월효과 (Lag Effect) ----------
lags = [0,1,2,3]
lag_corrs = [df['practice_time_min'].shift(l).corr(df['fatigue_score_1_10']) for l in lags]
lag_colors = ['#2059A6','#3E82C8','#7CAEE0','#B9D3F0']  # 진한 파랑 → 옅은 파랑
fig, ax = plt.subplots(figsize=(7,4))
bars = ax.bar([f"{l}일 전" if l>0 else "당일" for l in lags], lag_corrs, color=lag_colors, width=0.55)
for i,v in enumerate(lag_corrs):
    ax.text(i, v+0.012, f"{v:.3f}", ha='center', fontproperties=kfont(11), color='#222222')
ax.set_ylabel('연습시간과 피로도의 상관계수', fontproperties=kfont(11))
style_axes(ax, '훈련부하의 이월효과 (Lag Effect)')
plt.tight_layout()
plt.savefig('graphs/09_lag_effect.png', dpi=140, facecolor=BG)
plt.close()
print("\n=== 이월효과(Lag Effect) ===", [round(c,3) for c in lag_corrs])

# ---------- 12) [추가] 시각화 ⑩ 산점도 — 훈련부하(AU) vs 피로도 ----------
# 8개 event_type을 4개 그룹으로 단순화해 산점도 범례가 한눈에 들어오도록 한다.
def simplify_group(et):
    if et == 'normal': return '평상시'
    if et in ('ankle_injury_acute','ankle_injury_modified','ankle_injury_recovery'): return '부상 관련'
    if et == 'performance_week': return '공연주간'
    return '기타'  # travel, mild_illness, intensive_workshop
df['scatter_group'] = df['event_type'].apply(simplify_group)
group_order = ['평상시','부상 관련','공연주간','기타']
group_colors = {'평상시':BLUE, '부상 관련':'#E45C5C', '공연주간':'#2CA05A', '기타':'#BBBBBB'}

fig, ax = plt.subplots(figsize=(9,6))
sns.scatterplot(data=df, x='session_load_au', y='fatigue_score_1_10', hue='scatter_group', hue_order=group_order,
                 palette=group_colors, alpha=0.85, s=42, edgecolor='white', linewidth=0.4, ax=ax)
sns.regplot(data=df, x='session_load_au', y='fatigue_score_1_10', scatter=False, ax=ax,
            color='#333333', line_kws={'linewidth':1.6, 'linestyle':(0,(5,2))})
r_au_fatigue = df['session_load_au'].corr(df['fatigue_score_1_10'])
ax.text(0.98, 0.04, f"r = {r_au_fatigue:.3f}", transform=ax.transAxes, ha='right',
        fontproperties=kfont(11), color='#444444')
ax.legend(loc='upper left', prop=kfont(10), frameon=False)
style_axes(ax, '산점도 — 훈련부하(AU) vs 피로도')
ax.set_xlabel('훈련부하 (AU)', fontproperties=kfont(11)); ax.set_ylabel('피로도 (1-10)', fontproperties=kfont(11))
plt.tight_layout()
plt.savefig('graphs/10_scatter_au_fatigue.png', dpi=140, facecolor=BG)
plt.close()
print("=== 산점도: AU vs 피로도 상관계수 ===", round(r_au_fatigue,3))
print("scatter_group 분포:")
print(df['scatter_group'].value_counts())

# ---------- 13) [추가] 시각화 ⑪ 박스플롯 — event_type별 피로도 분포 ----------
fig, ax = plt.subplots(figsize=(10,5.8))
fig.patch.set_facecolor(PANEL_BG)
ax.set_facecolor(PANEL_BG)
med_order = df.groupby('event_type')['fatigue_score_1_10'].median().sort_values(ascending=False).index.tolist()
sns.boxplot(data=df, x='event_type', y='fatigue_score_1_10', order=med_order, hue='event_type', hue_order=med_order,
            palette=[EVENT_COLORS[e] for e in med_order], legend=False, ax=ax, width=0.5,
            fliersize=3, linewidth=1.1, saturation=0.95)
sns.stripplot(data=df, x='event_type', y='fatigue_score_1_10', order=med_order,
              color='#333333', alpha=0.22, size=2.3, jitter=0.25, ax=ax)
ax.set_xticks(range(len(med_order)))
ax.set_xticklabels([EVENT_KLABEL[e]+f"\n(n={(df['event_type']==e).sum()})" for e in med_order], fontproperties=kfont(9))
ax.set_title('훈련 상황(event_type)별 피로도 분포 — 중앙값 높은 순', fontproperties=kfont(15,'bold'), pad=14, color='#111111')
for lbl in ax.get_xticklabels()+ax.get_yticklabels(): lbl.set_fontproperties(kfont(10))
sns.despine(ax=ax)
ax.set_xlabel('훈련 상황(event_type) — 가로축', fontproperties=kfont(10)); ax.set_ylabel('피로도(1-10) — 세로축', fontproperties=kfont(10))
ax.text(0.99, 0.03, '※ n이 작은 항목(1~6)은 박스가 찌그러지거나 선으로만 보임 — 표본 부족의 시각적 증거',
        transform=ax.transAxes, ha='right', fontproperties=kfont(8), color='#5C4E63')
plt.tight_layout()
plt.savefig('graphs/11_boxplot_fatigue_by_event.png', dpi=140, facecolor=PANEL_BG)
plt.close()
print("전체 스크립트 실행 완료 — graphs/ 폴더에 이미지 11개가 모두 생성되었는지 확인하세요.")
