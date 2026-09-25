"""
附件2 193A 数据分析：4种疗法比较 + 费用分析
"""
import numpy as np
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
import json

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ── 解析数据 ──
# 4种疗法:
# 1 = 600mg zidovudine + 400mg didanosine (交替使用)
# 2 = 600mg zidovudine + 2.25mg zalcitabine
# 3 = 600mg zidovudine + 400mg didanosine
# 4 = 600mg zidovudine + 400mg didanosine + 400mg nevirapine

data = defaultdict(lambda: {'therapy': None, 'age': None, 't': [], 'log_cd4': []})
with open(r'C:\Users\bshmz\Desktop\小学期数模\Bdata\附件2.txt', 'r', encoding='gbk', errors='ignore') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            therapy = int(parts[1])
            age = float(parts[2])
            t = float(parts[3])
            log_cd4 = float(parts[4])
        except (ValueError, IndexError):
            continue
        d = data[pid]
        d['therapy'] = therapy
        d['age'] = age
        d['t'].append(t)
        d['log_cd4'].append(log_cd4)

print(f"Total patients: {len(data)}")

# ── 按疗法分组 ──
groups = {1: [], 2: [], 3: [], 4: []}
for pid, d in data.items():
    if d['therapy'] in groups:
        groups[d['therapy']].append(pid)

for t in [1, 2, 3, 4]:
    print(f"Therapy {t}: {len(groups[t])} patients")

# ── 计算每个患者的关键指标 ──
def calc_metrics(pid):
    d = data[pid]
    t = np.array(d['t'])
    y = np.array(d['log_cd4'])
    if len(t) < 2:
        return None
    # Sort by time
    idx = np.argsort(t)
    t, y = t[idx], y[idx]

    initial = y[0]
    final = y[-1]
    change = final - initial

    # CD4 count = exp(log_cd4) - 1
    cd4_init = np.exp(initial) - 1
    cd4_final = np.exp(final) - 1
    cd4_change = cd4_final - cd4_init

    # 线性回归斜率
    slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
    # 计算 AUC (trapezoidal)
    auc = np.trapezoid(y, t) / (t[-1] - t[0]) if t[-1] > t[0] else y[0]

    return {
        'pid': pid,
        'therapy': d['therapy'],
        'age': d['age'],
        'n_visits': len(t),
        'duration': t[-1] - t[0],
        'log_cd4_init': initial,
        'log_cd4_final': final,
        'log_cd4_change': change,
        'cd4_init': cd4_init,
        'cd4_final': cd4_final,
        'cd4_change': cd4_change,
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'auc': auc,
        'responder': 1 if change > 0 else 0,
    }

all_metrics = []
for pid in data:
    m = calc_metrics(pid)
    if m:
        all_metrics.append(m)

print(f"\nValid patients: {len(all_metrics)}")

# ── 按疗法汇总 ──
therapy_names = {
    1: 'Therapy 1: ZDV+didanosine(alt)',
    2: 'Therapy 2: ZDV+zalcitabine',
    3: 'Therapy 3: ZDV+didanosine',
    4: 'Therapy 4: ZDV+didanosine+nevirapine'
}

print("\n=== 疗法比较 ===")
for t in [1, 2, 3, 4]:
    metrics_t = [m for m in all_metrics if m['therapy'] == t]
    if not metrics_t:
        continue
    changes = [m['log_cd4_change'] for m in metrics_t]
    slopes = [m['slope'] for m in metrics_t]
    cd4_changes = [m['cd4_change'] for m in metrics_t]
    responders = sum(m['responder'] for m in metrics_t)
    print(f"\n{therapy_names[t]} (n={len(metrics_t)})")
    print(f"  log(CD4+1) change: {np.mean(changes):.4f} ± {np.std(changes):.4f}")
    print(f"  CD4 count change: {np.mean(cd4_changes):.1f} ± {np.std(cd4_changes):.1f}")
    print(f"  Slope: {np.mean(slopes)*1000:.2f} x 1e-3/week")
    print(f"  Responders: {responders}/{len(metrics_t)} = {responders/len(metrics_t)*100:.1f}%")
    # 95% CI
    se = np.std(changes) / np.sqrt(len(changes))
    ci = 1.96 * se
    print(f"  95% CI for log change: [{np.mean(changes)-ci:.4f}, {np.mean(changes)+ci:.4f}]")

# ── ANOVA 检验 ──
print("\n=== 统计检验 ===")
therapy_groups = []
for t in [1, 2, 3, 4]:
    therapy_groups.append([m['log_cd4_change'] for m in all_metrics if m['therapy'] == t])

f_stat, p_anova = stats.f_oneway(*therapy_groups)
print(f"One-way ANOVA: F={f_stat:.2f}, p={p_anova:.4f}")

# Kruskal-Wallis (non-parametric)
h_stat, p_kw = stats.kruskal(*therapy_groups)
print(f"Kruskal-Wallis: H={h_stat:.2f}, p={p_kw:.4f}")

# Pairwise t-tests (with Bonferroni correction)
print("\nPairwise t-tests (Bonferroni corrected):")
best_therapy = 4  # Hypothesis
for t1, t2 in [(4, 1), (4, 2), (4, 3), (1, 2), (1, 3), (2, 3)]:
    g1 = therapy_groups[t1-1]
    g2 = therapy_groups[t2-1]
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
    p_corrected = min(p_val * 6, 1.0)  # Bonferroni for 6 comparisons
    sig = '***' if p_corrected < 0.001 else ('**' if p_corrected < 0.01 else ('*' if p_corrected < 0.05 else ''))
    print(f"  T{t1} vs T{t2}: t={t_stat:.2f}, p={p_val:.4f}, p_corr={p_corrected:.4f} {sig}")

# ── 子问题(2): 对最优疗法(T4)预测继续治疗效果 ──
t4_metrics = [m for m in all_metrics if m['therapy'] == 4]
t4_slopes = [m['slope'] for m in t4_metrics]
t4_log_init = [m['log_cd4_init'] for m in t4_metrics]
print(f"\n=== 最优疗法 (Therapy 4) 预测 ===")
print(f"T4 mean slope: {np.mean(t4_slopes)*1000:.3f} x 1e-3/week")
print(f"T4 median slope: {np.median(t4_slopes)*1000:.3f} x 1e-3/week")
# 预测继续治疗: 线性外推
mean_slope = np.mean(t4_slopes)
mean_init = np.mean(t4_log_init)
predict_weeks = [8, 16, 24, 32, 40, 48, 56, 64]
print(f"\nT4 继续治疗预测 (基于平均斜率):")
for w in predict_weeks:
    pred_log = mean_init + mean_slope * w
    pred_cd4 = np.exp(pred_log) - 1
    print(f"  Week {w}: log(CD4+1)={pred_log:.3f}, CD4={pred_cd4:.0f}")

# T4 最佳治疗时间 (基于斜率趋近于0)
# 线性递减模型: 假设斜率随时间递减
t4_durations = [m['duration'] for m in t4_metrics]
print(f"\nT4 平均观测时长: {np.mean(t4_durations):.1f} 周")

# T4 响应率最高时的特征
t4_responders = sum(1 for m in t4_metrics if m['responder'] == 1)
print(f"T4 响应率: {t4_responders}/{len(t4_metrics)} = {t4_responders/len(t4_metrics)*100:.1f}%")

# ════════════════════════════════════════
# 子问题(3): 费用分析
# ════════════════════════════════════════
print("\n=== 子问题(3): 费用-效果分析 ===")

# 药品价格:
# 600mg zidovudine: $1.60
# 400mg didanosine: $0.85
# 2.25mg zalcitabine: $1.85
# 400mg nevirapine: $1.20

# 计算每种疗法每周费用 (假设每天服药)
daily_cost = {
    1: 1.60 + 0.85,                          # ZDV + didanosine
    2: 1.60 + 1.85,                          # ZDV + zalcitabine
    3: 1.60 + 0.85,                          # ZDV + didanosine (same as T1)
    4: 1.60 + 0.85 + 1.20,                   # ZDV + didanosine + nevirapine
}
weekly_cost = {k: v * 7 for k, v in daily_cost.items()}
monthly_cost = {k: v * 30 for k, v in daily_cost.items()}

for t in [1, 2, 3, 4]:
    print(f"Therapy {t}: daily=${daily_cost[t]:.2f}, weekly=${weekly_cost[t]:.2f}, monthly=${monthly_cost[t]:.2f}")

# 费用效果比 (Cost-Effectiveness Ratio)
# CER = 每周费用 / log(CD4+1) 每周变化量
print("\n费用效果分析 (CER = $ per 0.01 log CD4 improvement per week):")
cer_results = []
for t in [1, 2, 3, 4]:
    metrics_t = [m for m in all_metrics if m['therapy'] == t]
    mean_slope = np.mean([m['slope'] for m in metrics_t])
    weekly = weekly_cost[t]
    if mean_slope > 0:
        cer = weekly / (mean_slope * 100)  # $ per 0.01 log improvement
    else:
        cer = float('inf')
    response_rate = sum(m['responder'] for m in metrics_t) / len(metrics_t) * 100
    cer_results.append({
        'therapy': t,
        'weekly_cost': weekly,
        'mean_slope': mean_slope,
        'response_rate': response_rate,
        'cer': cer if cer != float('inf') else 999,
    })
    print(f"  Therapy {t}: ${weekly:.2f}/week, slope={mean_slope*1000:.3f}x1e-3/wk, "
          f"CER=${cer:.2f}/0.01log, response={response_rate:.1f}%")

# 综合排名
print("\n=== 综合排名 (考虑效果+费用) ===")
# 使用 TOPSIS 或简单加权评分
# Score = w1 * (slope/max_slope) + w2 * (1 - cost/max_cost)
max_slope = max(r['mean_slope'] for r in cer_results)
max_cost = max(r['weekly_cost'] for r in cer_results)
scores = []
for r in cer_results:
    s_eff = r['mean_slope'] / max_slope if max_slope > 0 else 0
    s_cost = 1 - r['weekly_cost'] / max_cost
    s_resp = r['response_rate'] / 100
    # 综合得分 (等权重)
    total = (s_eff + s_cost + s_resp) / 3
    scores.append({
        'therapy': r['therapy'],
        'eff_score': round(s_eff, 3),
        'cost_score': round(s_cost, 3),
        'resp_score': round(s_resp, 3),
        'total': round(total, 3),
    })
    print(f"  Therapy {r['therapy']}: eff={s_eff:.3f}, cost={s_cost:.3f}, "
          f"resp={s_resp:.3f}, total={total:.3f}")

# 排序
scores.sort(key=lambda x: x['total'], reverse=True)
for i, s in enumerate(scores):
    print(f"  Rank {i+1}: Therapy {s['therapy']} (score={s['total']:.3f})")

# ════════════════════════════════════════
# 图4a-4d: 4种疗法对比 (拆分为4张独立图)
# ════════════════════════════════════════

colors_t = {1: '#2c7bb6', 2: '#d7191c', 3: '#fdae61', 4: '#5e3c99'}

# 图4a: log(CD4+1) 变化箱线图
fig, ax = plt.subplots(figsize=(10, 6))
bp_data = [therapy_groups[i] for i in range(4)]
bp = ax.boxplot(bp_data, patch_artist=True, widths=0.5)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(colors_t[i+1])
    patch.set_alpha(0.7)
ax.set_xticklabels(['T1\nZDV+didanosine\n(alt)', 'T2\nZDV+zalcitabine',
                     'T3\nZDV+didanosine', 'T4\nZDV+didanosine\n+nevirapine'], fontsize=10)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.set_ylabel('log(CD4+1) Change', fontsize=12)
ax.set_title('CD4 Change by Therapy\n不同疗法的 log(CD4+1) 变化量', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(r'E:\AI Switch\works\tb_therapy\fig4a_boxplot.png', dpi=200)
plt.close(fig)
print('Fig4a saved')

# 图4b: 响应率柱状图
fig, ax = plt.subplots(figsize=(8, 5))
rates = [sum(m['responder'] for m in all_metrics if m['therapy'] == t) / len([m for m in all_metrics if m['therapy'] == t]) * 100
         for t in [1, 2, 3, 4]]
counts = [len([m for m in all_metrics if m['therapy'] == t]) for t in [1, 2, 3, 4]]
bars = ax.bar([1, 2, 3, 4], rates, color=[colors_t[i] for i in [1,2,3,4]], edgecolor='white')
for bar, rate, count in zip(bars, rates, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{rate:.1f}%\n(n={count})',
            ha='center', fontsize=10)
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'], fontsize=11)
ax.set_ylabel('Response Rate (%)', fontsize=12)
ax.set_title('Response Rate by Therapy\n不同疗法的响应率', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(rates) * 1.25)
fig.tight_layout()
fig.savefig(r'E:\AI Switch\works\tb_therapy\fig4b_response_rate.png', dpi=200)
plt.close(fig)
print('Fig4b saved')

# 图4c: 费用效果平面图
fig, ax = plt.subplots(figsize=(9, 6))
costs = [monthly_cost[t] for t in [1, 2, 3, 4]]
effs = [np.mean([m['log_cd4_change'] for m in all_metrics if m['therapy'] == t]) for t in [1, 2, 3, 4]]
for t in [1, 2, 3, 4]:
    ax.scatter(costs[t-1], effs[t-1], s=counts[t-1]*2, c=colors_t[t], alpha=0.7, edgecolors='black', linewidth=1)
    ax.annotate(f'T{t}', (costs[t-1], effs[t-1]), textcoords='offset points', xytext=(8, 5), fontsize=12, fontweight='bold')
ax.set_xlabel('Monthly Cost ($)', fontsize=12)
ax.set_ylabel('Mean log(CD4+1) Change', fontsize=12)
ax.set_title('Cost-Effectiveness\n费用效果平面图 (气泡大小=样本量)', fontsize=13, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(r'E:\AI Switch\works\tb_therapy\fig4c_cost_effectiveness.png', dpi=200)
plt.close(fig)
print('Fig4c saved')

# 图4d: 综合评分柱状图
fig, ax = plt.subplots(figsize=(9, 6))
therapies = [s['therapy'] for s in scores]
eff_s = [s['eff_score'] for s in scores]
cost_s = [s['cost_score'] for s in scores]
resp_s = [s['resp_score'] for s in scores]
x = np.arange(len(therapies))
w = 0.25
ax.bar(x - w, eff_s, w, label='Efficacy', color='#2c7bb6', alpha=0.8)
ax.bar(x, cost_s, w, label='Cost', color='#d7191c', alpha=0.8)
ax.bar(x + w, resp_s, w, label='Response', color='#fdae61', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'T{t}' for t in therapies], fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Comprehensive Score by Therapy\n综合评分', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 1.1)
fig.tight_layout()
fig.savefig(r'E:\AI Switch\works\tb_therapy\fig4d_comprehensive_score.png', dpi=200)
plt.close(fig)
print('Fig4d saved')

# ════════════════════════════════════════
# 图5a-5b: T4 趋势预测 (拆分为2张独立图)
# ════════════════════════════════════════

# 图5a: T4 个体轨迹
fig, ax = plt.subplots(figsize=(11, 6))
t4_pids = groups[4]
for pid in t4_pids[:50]:
    d = data[pid]
    t = np.array(d['t'])
    y = np.array(d['log_cd4'])
    idx = np.argsort(t)
    ax.plot(t[idx], y[idx], '-', alpha=0.15, color='#5e3c99', linewidth=0.8)
ax.set_xlabel('Week', fontsize=12)
ax.set_ylabel('log(CD4+1)', fontsize=12)
ax.set_title('Therapy 4: Individual Trajectories\nT4 患者个体轨迹 (n=50)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(r'E:\AI Switch\works\tb_therapy\fig5a_individual.png', dpi=200)
plt.close(fig)
print('Fig5a saved')

# 图5b: T4 平均趋势 + 预测
fig, ax = plt.subplots(figsize=(11, 6))
all_t4_t, all_t4_y = [], []
for pid in t4_pids:
    d = data[pid]
    all_t4_t.extend(d['t'])
    all_t4_y.extend(d['log_cd4'])
all_t4_t = np.array(all_t4_t)
all_t4_y = np.array(all_t4_y)

bins = np.arange(0, 45, 4)
bmeans_t4, bstds_t4, bcents_t4 = [], [], []
for i in range(len(bins)-1):
    m = (all_t4_t >= bins[i]) & (all_t4_t < bins[i+1])
    if m.sum() > 5:
        bmeans_t4.append(all_t4_y[m].mean())
        bstds_t4.append(all_t4_y[m].std())
        bcents_t4.append((bins[i]+bins[i+1])/2)

bmeans_t4 = np.array(bmeans_t4); bstds_t4 = np.array(bstds_t4); bcents_t4 = np.array(bcents_t4)

ax.errorbar(bcents_t4, bmeans_t4, yerr=bstds_t4, fmt='o-', color='#5e3c99', lw=2, ms=8, capsize=3, label='Observed')

pred_w = np.arange(0, 65, 4)
pred_mean = mean_init + mean_slope * pred_w
ax.plot(pred_w, pred_mean, '--', color='#d7191c', lw=2, label=f'Predicted (slope={mean_slope*1000:.2f}x1e-3/wk)')

se_slope = np.std(t4_slopes) / np.sqrt(len(t4_slopes))
ci_upper = mean_init + (mean_slope + 1.96*se_slope) * pred_w
ci_lower = mean_init + (mean_slope - 1.96*se_slope) * pred_w
ax.fill_between(pred_w, ci_lower, ci_upper, alpha=0.15, color='#d7191c')

ax.set_xlabel('Week', fontsize=12)
ax.set_ylabel('log(CD4+1)', fontsize=12)
ax.set_title('Therapy 4: Trend & Prediction\nT4 平均趋势与预测', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(r'E:\AI Switch\works\tb_therapy\fig5b_prediction.png', dpi=200)
plt.close(fig)
print('Fig5b saved')

# ════════════════════════════════════════
# 图6: 费用 vs 效果散点图
# ════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
for r in cer_results:
    t = r['therapy']
    size = r['response_rate'] * 1.5 + 30  # 气泡大小反映响应率
    ax.scatter(r['mean_slope'], r['weekly_cost'], s=size**1.5,
               color=colors_t[t], edgecolors='white', linewidths=1.2,
               zorder=5, label=f'T{t}: {therapy_names[t].split(": ")[1]}')
    offset_y = 0.8 if t != 3 else -1.5  # 避免标签重叠
    ax.annotate(f'T{t}\n${r["weekly_cost"]:.2f}/wk\nslope={r["mean_slope"]*1000:.1f}e-3',
                (r['mean_slope'], r['weekly_cost']),
                xytext=(r['mean_slope'] + 0.0003, r['weekly_cost'] + offset_y),
                fontsize=10, fontweight='bold', color=colors_t[t],
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)

ax.set_xlim(-0.022, 0.007)
ax.set_ylim(15, 28)
# 象限标注 (在 set_xlim/set_ylim 之后放置)
ax.text(-0.020, 27.2, 'Effect ↓\nCost ↑\n(最差)',
        ha='left', va='top', fontsize=9, color='gray', alpha=0.7)
ax.text(0.005, 27.2, 'Effect ↑\nCost ↑',
        ha='right', va='top', fontsize=9, color='gray', alpha=0.7)
ax.text(-0.020, 16.0, 'Effect ↓\nCost ↓',
        ha='left', va='bottom', fontsize=9, color='gray', alpha=0.7)
ax.text(0.005, 16.0, 'Effect ↑\nCost ↓\n(最优)',
        ha='right', va='bottom', fontsize=9, color='gray', alpha=0.7)

ax.set_xlabel('Mean Slope (log(CD4+1) / week)', fontsize=12)
ax.set_ylabel('Weekly Cost ($)', fontsize=12)
ax.set_title('Cost vs. Effectiveness\n费用与效果散点图 (气泡大小=响应率)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.2)
fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.12)
fig.savefig(r'E:\AI Switch\works\tb_therapy\fig6_cost_effectiveness.png', dpi=200)
print('Fig6 saved')

# ── 保存结果 ──
summary_q2 = {
    'n_total': len(data),
    'therapy_counts': {str(t): len(groups[t]) for t in [1,2,3,4]},
    'anova_f': round(float(f_stat), 2),
    'anova_p': round(float(p_anova), 6),
    'kruskal_h': round(float(h_stat), 2),
    'kruskal_p': round(float(p_kw), 6),
    'best_therapy': 4,
    'best_therapy_responders_pct': round(float(t4_responders/len(t4_metrics)*100), 1),
    't4_mean_slope': round(float(mean_slope), 6),
    't4_pred_week48_cd4': round(float(np.exp(mean_init + mean_slope * 48) - 1), 1),
    'ranking': [{'rank': i+1, 'therapy': s['therapy'], 'score': s['total']} for i, s in enumerate(scores)],
}
with open(r'E:\AI Switch\works\tb_therapy\results_q2.json', 'w', encoding='utf-8') as f:
    json.dump(summary_q2, f, ensure_ascii=False, indent=2)

print("\n=== 结果已保存 ===")
for k, v in summary_q2.items():
    print(f"  {k}: {v}")
print("Done.")
