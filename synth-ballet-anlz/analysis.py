# -*- coding: utf-8 -*-
"""
발레 훈련 365일 시계열 분석 파이프라인
- 데이터 정제 요약, 시각화 5종(필수2+보너스), STL/ARIMA(보너스), 인사이트용 통계 출력
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_prop = font_manager.FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

def kfont(size=11, weight='normal'):
    fp = font_manager.FontProperties(fname=FONT_PATH, size=size, weight=weight)
    return fp

df = pd.read_csv('data/ballet_training_cleaned_365.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# ---------- 1) 데이터 정제 요약 ----------
print("=== 데이터 정제 요약 ===")
print("총 일수:", len(df))
print("결측치(핵심 기술지표):")
for c in ['releve_balance_sec','pirouette_turns','jete_height_cm','turnout_angle_deg']:
    print(f"  {c}: {df[c].isna().sum()}일 결측 ({df[c].isna().mean()*100:.1f}%)")
print("measurement_status 분포:")
print(df['measurement_status'].value_counts())
print()

# ---------- 2) 시각화 1: 그랑 제떼 트렌드 + 7일 이동평균 (필수) ----------
df['jete_sma7'] = df['jete_height_cm'].rolling(7, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(11,4.5))
ax.plot(df['date'], df['jete_height_cm'], color='#C9BBA0', linewidth=0.8, label='일별 값')
ax.plot(df['date'], df['jete_sma7'], color='#8C3B47', linewidth=2, label='7일 이동평균')
ax.axvspan(pd.Timestamp('2025-08-18'), pd.Timestamp('2025-08-25'), color='#C46A5A', alpha=0.15, label='부상(급성)')
ax.axvspan(pd.Timestamp('2025-08-26'), pd.Timestamp('2025-09-26'), color='#4A6B5A', alpha=0.12, label='회복기')
ax.set_title('그랑 제떼 높이 트렌드 + 7일 이동평균 (2025)', fontproperties=kfont(14,'bold'))
ax.set_ylabel('높이(cm)', fontproperties=kfont(11))
ax.legend(prop=kfont(10))
for lbl in ax.get_xticklabels()+ax.get_yticklabels():
    lbl.set_fontproperties(kfont(9))
plt.tight_layout()
plt.savefig('graphs/01_jete_trend_sma.png', dpi=140)
plt.close()

# ---------- 3) 시각화 2: 요일별 연습시간 vs 피로도 (필수) ----------
order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
klabel = {'Mon':'월','Tue':'화','Wed':'수','Thu':'목','Fri':'금','Sat':'토','Sun':'일'}
wk = df.groupby('weekday').agg(practice=('practice_time_min','mean'), fatigue=('fatigue_score_1_10','mean')).reindex(order)
fig, ax1 = plt.subplots(figsize=(9,4.5))
ax1.bar([klabel[w] for w in order], wk['practice'], color='#C9BBA0', label='평균 연습시간(분)')
ax1.set_ylabel('연습시간(분)', fontproperties=kfont(11))
ax2 = ax1.twinx()
ax2.plot([klabel[w] for w in order], wk['fatigue'], color='#8C3B47', marker='o', linewidth=2, label='평균 피로도')
ax2.set_ylabel('피로도(1-10)', fontproperties=kfont(11))
ax1.set_title('요일별 평균 연습시간 vs 피로도', fontproperties=kfont(14,'bold'))
for lbl in ax1.get_xticklabels()+ax1.get_yticklabels()+ax2.get_yticklabels():
    lbl.set_fontproperties(kfont(10))
fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.88), prop=kfont(9))
plt.tight_layout()
plt.savefig('graphs/02_weekly_seasonality.png', dpi=140)
plt.close()

# ---------- 4) 시각화 3: 상관관계 히트맵 ----------
cols = ['practice_time_min','fatigue_score_1_10','sleep_hours','releve_balance_sec','pirouette_turns','jete_height_cm','turnout_angle_deg']
short = {'practice_time_min':'연습시간','fatigue_score_1_10':'피로도','sleep_hours':'수면시간','releve_balance_sec':'를르베','pirouette_turns':'피루엣','jete_height_cm':'제떼','turnout_angle_deg':'턴아웃'}
corr = df[cols].corr()
fig, ax = plt.subplots(figsize=(7,6))
im = ax.imshow(corr.values, cmap='RdYlGn_r', vmin=-1, vmax=1)
labels = [short[c] for c in cols]
ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels, fontproperties=kfont(10), rotation=45, ha='right')
ax.set_yticks(range(len(cols))); ax.set_yticklabels(labels, fontproperties=kfont(10))
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha='center', va='center', fontproperties=kfont(9),
                 color='white' if abs(corr.values[i,j])>0.5 else 'black')
ax.set_title('지표 간 상관관계 히트맵', fontproperties=kfont(14,'bold'))
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('상관계수', fontproperties=kfont(10))
plt.tight_layout()
plt.savefig('graphs/03_correlation_heatmap.png', dpi=140)
plt.close()
print("=== 상관관계 (히트맵 수치) ===")
print(corr.round(3))
print()

# ---------- 5) 시각화 4: 부상 전/중/회복 비교 (필수 - 구간별 통계) ----------
pre = df[df['date']<'2025-08-18']
during = df[df['event_type'].isin(['ankle_injury_acute','ankle_injury_modified'])]
recov = df[df['event_type']=='ankle_injury_recovery']
post = df[df['date']>='2025-09-27']
phase_metrics = ['releve_balance_sec','turnout_angle_deg','jete_height_cm']
phase_label = {'releve_balance_sec':'를르베(초)','turnout_angle_deg':'턴아웃(도)','jete_height_cm':'제떼(cm)'}
groups = {'부상 전': pre, '부상 중': during, '회복기': recov, '회복 완료 후': post}
fig, ax = plt.subplots(figsize=(10,4.5))
x = np.arange(len(phase_metrics))
width = 0.2
colors = ['#B9A88C','#C46A5A','#D9A87C','#4A6B5A']
for i, (gname, gdf) in enumerate(groups.items()):
    vals = [gdf[m].mean() for m in phase_metrics]
    ax.bar(x + i*width, vals, width, label=f"{gname}(n={len(gdf)})", color=colors[i])
ax.set_xticks(x + width*1.5)
ax.set_xticklabels([phase_label[m] for m in phase_metrics], fontproperties=kfont(11))
ax.set_title('부상 단계별 성과 비교 (전/중/회복기/회복완료후)', fontproperties=kfont(14,'bold'))
ax.legend(prop=kfont(9))
for lbl in ax.get_yticklabels():
    lbl.set_fontproperties(kfont(9))
plt.tight_layout()
plt.savefig('graphs/04_injury_phase_comparison.png', dpi=140)
plt.close()
print("=== 부상 단계별 평균 ===")
for gname, gdf in groups.items():
    print(gname, {m: round(gdf[m].mean(),2) for m in phase_metrics})
print()

# ---------- 6) [보너스] STL 시계열 분해 (주석 강화) ----------
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
    a.plot(comp, color='#8C3B47', linewidth=0.9)
    a.set_ylabel(name, fontproperties=kfont(10))
    for lbl in a.get_yticklabels(): lbl.set_fontproperties(kfont(8))

axes[0].axvspan(pd.Timestamp('2025-08-18'), pd.Timestamp('2025-09-26'), color='#8C3B47', alpha=0.12)
axes[0].annotate('부상 구간\n(관측값 급락)', xy=(pd.Timestamp('2025-08-23'), 22),
                  xytext=(pd.Timestamp('2025-10-20'), 20), fontproperties=kfont(9,'bold'), color='#8C3B47',
                  arrowprops=dict(arrowstyle='->', color='#8C3B47'))
axes[1].axvspan(pd.Timestamp('2025-08-18'), pd.Timestamp('2025-09-26'), color='#8C3B47', alpha=0.10)
axes[1].annotate(f'추세 강도(Ft)={Ft:.3f}\n→ 관측값 변동의 상당 부분이\n장기 추세로 설명됨', xy=(pd.Timestamp('2025-08-23'), stl.trend.loc['2025-08-23']),
                  xytext=(pd.Timestamp('2025-10-10'), stl.trend.min()+1), fontproperties=kfont(9), color='#2B2420',
                  arrowprops=dict(arrowstyle='->', color='#2B2420'))
axes[2].annotate(f'계절성 강도(Fs)={Fs:.3f}\n7일 주기 반복 (목요일에 최고,\n월요일에 최저)', xy=(pd.Timestamp('2025-05-01'), stl.seasonal.loc['2025-05-01']),
                  xytext=(pd.Timestamp('2025-02-01'), stl.seasonal.max()+0.3), fontproperties=kfont(9), color='#2B2420',
                  arrowprops=dict(arrowstyle='->', color='#2B2420'))
top_resid_dates = stl.resid.abs().sort_values(ascending=False).head(3).index
for d in top_resid_dates:
    axes[3].annotate(f"{d.strftime('%m/%d')}", xy=(d, stl.resid.loc[d]),
                      xytext=(d, stl.resid.loc[d]+(1.5 if stl.resid.loc[d]>0 else -2)),
                      fontproperties=kfont(8), ha='center', color='#4A6B5A')

axes[0].set_title('STL 시계열 분해 — 그랑 제떼 높이 (7일 주기)', fontproperties=kfont(14,'bold'))
for lbl in axes[-1].get_xticklabels(): lbl.set_fontproperties(kfont(9))
plt.tight_layout()
plt.savefig('graphs/05_stl_decomposition.png', dpi=140)
plt.close()

resid_std = stl.resid.std()
top_resid = stl.resid.abs().sort_values(ascending=False).head(5)
print("=== STL 잔차 상위 5개(이상치 후보) ===")
print(top_resid.round(2))
print()

# ---------- 7) [보너스] ARIMA 14일 예측 (주석 강화) ----------
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(ts, order=(1,1,1)).fit()
fc = model.get_forecast(steps=14)
fc_mean = fc.predicted_mean
fc_ci = fc.conf_int(alpha=0.05)

fig, ax = plt.subplots(figsize=(11,5))
ax.plot(ts.index[-60:], ts.values[-60:], color='#C9BBA0', linewidth=1, label='최근 60일 실측')
ax.plot(fc_mean.index, fc_mean.values, color='#8C3B47', linewidth=2, label='ARIMA(1,1,1) 예측')
ax.fill_between(fc_mean.index, fc_ci.iloc[:,0], fc_ci.iloc[:,1], color='#8C3B47', alpha=0.15, label='95% 신뢰구간')
ax.axvline(ts.index[-1], color='#4A6B5A', linestyle=':', linewidth=1)
ax.annotate('예측 시작점\n(마지막 실측일)', xy=(ts.index[-1], ts.values[-1]),
            xytext=(ts.index[-1]-pd.Timedelta(days=18), ts.values[-1]-4),
            fontproperties=kfont(9), color='#4A6B5A', arrowprops=dict(arrowstyle='->', color='#4A6B5A'))
ax.annotate(f'AR(1)={model.params["ar.L1"]:.2f}, MA(1)={model.params["ma.L1"]:.2f}\n신뢰구간이 갈수록 넓어짐\n(먼 미래일수록 불확실)',
            xy=(fc_mean.index[-1], fc_ci.iloc[-1,1]), xytext=(fc_mean.index[3], fc_ci.iloc[-1,1]+1),
            fontproperties=kfont(9), color='#2B2420')
ax.set_title('ARIMA(1,1,1) 14일 예측 — 그랑 제떼 높이', fontproperties=kfont(14,'bold'))
ax.legend(prop=kfont(9))
for lbl in ax.get_xticklabels()+ax.get_yticklabels(): lbl.set_fontproperties(kfont(9))
plt.tight_layout()
plt.savefig('graphs/06_arima_forecast.png', dpi=140)
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
fig, axes = plt.subplots(2,1, figsize=(11,7), sharex=True)
axes[0].bar(d2['date'], d2['session_load_au'], color='#C9BBA0', width=0.9, label='일별 세션 부하(AU)')
axes[0].plot(d2['date'], d2['au_sma7'], color='#2B2420', linewidth=2, label='7일 이동평균')
axes[0].axvline(pd.Timestamp('2025-08-18'), color='#8C3B47', linestyle='--', linewidth=1.5, label='부상 발생(8/18)')
axes[0].set_ylabel('AU', fontproperties=kfont(11))
axes[0].set_title('훈련부하(AU) 추이와 부상 전조 신호', fontproperties=kfont(14,'bold'))
axes[0].legend(prop=kfont(9))

axes[1].plot(d2['date'], d2['fatigue_score_1_10'], color='#D9A05B', marker='.', markersize=3, linewidth=1, label='피로도(1-10)')
axes[1].plot(d2['date'], d2['pain_score_0_10'], color='#8C3B47', marker='.', markersize=3, linewidth=1, label='통증 점수(0-10)')
axes[1].axhline(6.5, color='#8C3B47', linestyle=':', linewidth=1, label='피로 임계선(6.5)')
axes[1].axvspan(pd.Timestamp('2025-08-11'), pd.Timestamp('2025-08-17'), color='#D9A05B', alpha=0.12, label='전조 구간')
axes[1].axvspan(pd.Timestamp('2025-08-18'), pd.Timestamp('2025-08-25'), color='#8C3B47', alpha=0.10, label='급성 부상 구간')
axes[1].set_ylabel('점수', fontproperties=kfont(11))
axes[1].legend(prop=kfont(8), ncol=2)
for a in axes:
    for lbl in a.get_yticklabels(): lbl.set_fontproperties(kfont(9))
for lbl in axes[-1].get_xticklabels(): lbl.set_fontproperties(kfont(9))
plt.tight_layout()
plt.savefig('graphs/07_workload_injury_precursor.png', dpi=140)
plt.close()

# ACWR(급성:만성 부하비율) 검증
d2['acute_7d'] = d2['session_load_au'].rolling(7, min_periods=1).sum()
d2['chronic_28d_avg_weekly'] = d2['session_load_au'].rolling(28, min_periods=7).sum() / 4
d2['acwr'] = d2['acute_7d'] / d2['chronic_28d_avg_weekly']
print("=== ACWR 검증 (부상 전날 8/17) ===")
print(d2[d2['date']==pd.Timestamp('2025-08-17')][['date','acwr']])
print("전체 365일 중 ACWR>=1.5(과부하 기준) 일수:", (d2['acwr']>=1.5).sum())

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
print("Done (added sections).")

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
ax.bar(xs, [res2[t]['r2'] for t in targets2], color=['#8C3B47' if res2[t]['r2']>=0.3 else '#C9BBA0' for t in targets2])
ax.axhline(0.3, color='#4A6B5A', linestyle=':', linewidth=1, label='참고선(R²=0.3)')
ax.set_xticks(xs); ax.set_xticklabels([tlabel2[t] for t in targets2], fontproperties=kfont(11))
ax.set_ylabel('R²', fontproperties=kfont(11))
ax.set_title('지표별 훈련부하 기반 예측 설명력(R²) 비교', fontproperties=kfont(14,'bold'))
for i,t in enumerate(targets2):
    ax.text(i, res2[t]['r2']+0.01, f"{res2[t]['r2']:.3f}", ha='center', fontproperties=kfont(10))
ax.legend(prop=kfont(9))
plt.tight_layout()
plt.savefig('graphs/08_prediction_r2_comparison.png', dpi=140)
plt.close()
print("\n=== 지표별 예측 설명력 ===")
for t in targets2: print(t, {k:round(v,3) for k,v in res2[t].items()})

# ---------- 11) [추가] 훈련부하의 이월효과 (Lag Effect) ----------
lags = [0,1,2,3]
lag_corrs = [df['practice_time_min'].shift(l).corr(df['fatigue_score_1_10']) for l in lags]
fig, ax = plt.subplots(figsize=(7,4))
ax.bar([f"{l}일 전" if l>0 else "당일" for l in lags], lag_corrs, color='#8C3B47')
for i,v in enumerate(lag_corrs):
    ax.text(i, v+0.01, f"{v:.3f}", ha='center', fontproperties=kfont(10))
ax.set_ylabel('연습시간과 피로도의 상관계수', fontproperties=kfont(10))
ax.set_title('훈련부하의 이월효과 (Lag Effect)', fontproperties=kfont(14,'bold'))
for lbl in ax.get_xticklabels()+ax.get_yticklabels(): lbl.set_fontproperties(kfont(10))
plt.tight_layout()
plt.savefig('graphs/09_lag_effect.png', dpi=140)
plt.close()
print("\n=== 이월효과(지연 상관관계) ===", [round(c,3) for c in lag_corrs])
print("전체 스크립트 실행 완료 — graphs/ 폴더에 이미지 9개가 모두 생성되었는지 확인하세요.")
