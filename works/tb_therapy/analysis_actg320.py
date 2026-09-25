"""
附件1 ACTG320 数据分析：CD4 + 病毒浓度轨迹拟合，最佳停药时间
"""
import numpy as np
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import defaultdict
import json

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ── 解析数据 (空格分隔) ──
data = defaultdict(lambda: {'cd4_t': [], 'cd4_v': [], 'vl_t': [], 'vl_v': []})
with open(r'C:\Users\bshmz\Desktop\小学期数模\Bdata\附件1.txt', 'r', encoding='gbk', errors='ignore') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if len(parts) >= 3:
            try:
                data[pid]['cd4_t'].append(float(parts[1]))
                data[pid]['cd4_v'].append(float(parts[2]))
            except ValueError:
                pass
        if len(parts) >= 5:
            try:
                data[pid]['vl_t'].append(float(parts[3]))
                data[pid]['vl_v'].append(float(parts[4]))
            except ValueError:
                pass

print(f"Total patients: {len(data)}")

# ── 模型函数 ──
def cd4_model(t, a, b, c):
    """CD4 = a * (1 - b * exp(-c * t))  渐进增长, a>0, 0<b<1, c>0"""
    return a * (1.0 - b * np.exp(-c * t))

def vl_model(t, a, b, c):
    """VLoad = a + b * exp(-c * t)  渐进衰减"""
    return a + b * np.exp(-c * t)

def linear_model(t, k, b):
    """简单线性"""
    return k * t + b

# ── 拟合 ──
results = []
for pid, d in data.items():
    cd4_t = np.array(d['cd4_t'])
    cd4_v = np.array(d['cd4_v'])
    vl_t = np.array(d['vl_t'])
    vl_v = np.array(d['vl_v'])

    if len(cd4_t) < 3:
        continue

    # CD4 拟合: 先用线性，再用指数
    cd4_fit_type = 'linear'
    cd4_params = None
    cd4_r2 = 0

    # 尝试指数拟合
    try:
        a0 = max(np.max(cd4_v) * 1.2, cd4_v[-1] * 1.3)
        b0 = max(0.3, min(0.95, 1.0 - cd4_v[0] / a0))
        c0 = 0.05
        popt, _ = curve_fit(cd4_model, cd4_t, cd4_v, p0=[a0, b0, c0],
                            bounds=([0, 0, 0.001], [np.inf, 1.0, 1.0]), maxfev=5000)
        pred = cd4_model(cd4_t, *popt)
        r2 = 1 - np.sum((cd4_v - pred)**2) / np.sum((cd4_v - np.mean(cd4_v))**2)
        if r2 > 0.5:
            cd4_fit_type = 'exponential'
            cd4_params = popt
            cd4_r2 = r2
    except:
        pass

    # 回退到线性
    if cd4_fit_type == 'linear':
        try:
            k = (cd4_v[-1] - cd4_v[0]) / (cd4_t[-1] - cd4_t[0] + 1e-6)
            b = cd4_v[0]
            popt_lin = [k, b]
            pred = linear_model(cd4_t, *popt_lin)
            r2 = 1 - np.sum((cd4_v - pred)**2) / np.sum((cd4_v - np.mean(cd4_v))**2)
            cd4_params = popt_lin
            cd4_r2 = max(0, r2)
        except:
            cd4_params = [0, cd4_v[0]]
            cd4_r2 = 0

    # VLoad 拟合
    vl_params = None
    vl_r2 = 0
    try:
        if len(vl_t) >= 3:
            a0 = max(0.5, np.min(vl_v) * 0.9)
            b0 = vl_v[0] - a0
            c0 = 0.1
            popt_vl, _ = curve_fit(vl_model, vl_t, vl_v, p0=[a0, b0, c0],
                                   bounds=([0, -10, 0.001], [10, 10, 1.0]), maxfev=5000)
            pred_vl = vl_model(vl_t, *popt_vl)
            vl_r2 = 1 - np.sum((vl_v - pred_vl)**2) / np.sum((vl_v - np.mean(vl_v))**2)
            vl_params = popt_vl
    except:
        pass

    # 最佳停药时间: CD4 达到渐近值 95% 的时刻
    if cd4_fit_type == 'exponential' and cd4_params is not None:
        a, b, c = cd4_params
        if c > 0.001:
            t_stop = -np.log(0.05 / max(b, 0.01)) / c
        else:
            t_stop = cd4_t[-1]
        t_stop = np.clip(t_stop, cd4_t[0] + 5, min(cd4_t[-1] * 2, 80))
    else:
        # 线性情况: 取观测终点
        t_stop = cd4_t[-1]

    results.append({
        'pid': pid,
        'n_cd4': len(cd4_t),
        'n_vl': len(vl_t),
        'cd4_init': cd4_v[0],
        'cd4_final': cd4_v[-1],
        'cd4_change': cd4_v[-1] - cd4_v[0],
        'cd4_change_pct': (cd4_v[-1] - cd4_v[0]) / max(cd4_v[0], 1) * 100,
        'vl_init': vl_v[0] if len(vl_v) > 0 else np.nan,
        'vl_final': vl_v[-1] if len(vl_v) > 0 else np.nan,
        'vl_change': (vl_v[0] - vl_v[-1]) if len(vl_v) > 1 else np.nan,
        'cd4_r2': cd4_r2,
        'vl_r2': vl_r2,
        't_stop': t_stop,
        'cd4_fit_type': cd4_fit_type,
    })

print(f"Valid fits: {len(results)}")
n_exp = sum(1 for r in results if r['cd4_fit_type'] == 'exponential')
print(f"Exponential fits: {n_exp}, Linear fits: {len(results) - n_exp}")

# ── 汇总 ──
cd4_changes = [r['cd4_change'] for r in results]
cd4_pct_changes = [r['cd4_change_pct'] for r in results]
vl_changes = [r['vl_change'] for r in results if not np.isnan(r['vl_change'])]
t_stops = [r['t_stop'] for r in results]

print(f"\n=== 子问题(1) 汇总统计 ===")
print(f"有效患者数: {len(results)}")
print(f"CD4 平均变化: {np.mean(cd4_changes):.1f} ± {np.std(cd4_changes):.1f}")
print(f"CD4 平均变化率: {np.mean(cd4_pct_changes):.1f}% ± {np.std(cd4_pct_changes):.1f}%")
print(f"CD4 改善比例: {sum(1 for c in cd4_changes if c > 0)}/{len(cd4_changes)} = {sum(1 for c in cd4_changes if c>0)/len(cd4_changes)*100:.1f}%")
print(f"VLoad 平均降低: {np.mean(vl_changes):.2f}")
print(f"VLoad 改善比例: {sum(1 for v in vl_changes if v > 0)}/{len(vl_changes)}")
print(f"最佳停药时间 (周): mean={np.mean(t_stops):.1f}, median={np.median(t_stops):.1f}, std={np.std(t_stops):.1f}")
print(f"停药时间范围: [{np.min(t_stops):.0f}, {np.max(t_stops):.0f}]")
print(f"CD4 拟合 R² 均值: {np.mean([r['cd4_r2'] for r in results]):.3f}")

# ════════════════════════════════════════
# 图1a-1f: 典型患者轨迹 (6个独立图)
# ════════════════════════════════════════
sorted_results = sorted(results, key=lambda r: r['cd4_change'], reverse=True)
sample_indices = [0, 1, len(results)//4, len(results)//4+1, -2, -1]
sample_pids = [sorted_results[i]['pid'] for i in sample_indices]
sample_labels = ['(a) 最佳改善', '(b) 次佳改善', '(c) 中等改善', '(d) 中等改善', '(e) 改善较差', '(f) 改善最差']

for i, pid in enumerate(sample_pids):
    fig, ax = plt.subplots(figsize=(10, 6))
    d = data[pid]
    r = next(r for r in results if r['pid'] == pid)

    t_max = max(max(d['cd4_t']), max(d['vl_t']) if len(d['vl_t']) > 0 else 0) * 1.3
    t_smooth = np.linspace(0, t_max, 100)

    ax.scatter(d['cd4_t'], d['cd4_v'], c='#2c7bb6', s=80, zorder=5, marker='o', label='CD4 (observed)')
    k = (d['cd4_v'][-1] - d['cd4_v'][0]) / max(d['cd4_t'][-1] - d['cd4_t'][0], 1)
    trend = linear_model(t_smooth, k, d['cd4_v'][0])
    ax.plot(t_smooth, trend, '--', color='#2c7bb6', alpha=0.5, linewidth=2, label='Trend')

    if len(d['vl_t']) > 0:
        ax2 = ax.twinx()
        ax2.scatter(d['vl_t'], d['vl_v'], c='#d7191c', s=60, zorder=5, marker='s', label='VLoad')
        ax2.set_ylabel('VLoad (log)', color='#d7191c', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='#d7191c')

    ax.axvline(x=r['t_stop'], color='green', linestyle=':', linewidth=2, alpha=0.8)
    ax.text(r['t_stop'] + 0.5, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.1,
            f'T={r["t_stop"]:.0f}w', fontsize=11, color='green')

    ax.set_title(f'Patient {pid} {sample_labels[i]}\nCD4: {d["cd4_v"][0]:.0f} → {d["cd4_v"][-1]:.0f} (Δ={r["cd4_change"]:.0f})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('CD4 Count', color='#2c7bb6', fontsize=12)
    ax.tick_params(axis='y', labelcolor='#2c7bb6')
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(rf'E:\AI Switch\works\tb_therapy\fig1{chr(97+i)}_trajectory.png', dpi=200)
    plt.close(fig)

print('Fig1a-1f saved')

# ════════════════════════════════════════
# 图2: CD4变化分布 + 停药时间分布
# ════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# ════════════════════════════════════════
# 图2a: CD4变化分布
# ════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
colors_bar = ['#d7191c' if c < 0 else '#2c7bb6' for c in cd4_changes]
ax.bar(range(len(cd4_changes)), sorted(cd4_changes), color=colors_bar, alpha=0.8, width=1)
ax.axhline(y=0, color='black', linewidth=1)
ax.axhline(y=np.mean(cd4_changes), color='orange', linestyle='--', linewidth=2, label=f'Mean={np.mean(cd4_changes):.0f}')
ax.set_xlabel('Patient (sorted)', fontsize=12)
ax.set_ylabel('CD4 Change', fontsize=12)
ax.set_title(f'CD4 Change Distribution\nCD4变化分布 (n={len(cd4_changes)}, {sum(1 for c in cd4_changes if c>0)} improved)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(r'E:\AI Switch\works\tb_therapy\fig2a_cd4_change.png', dpi=200)
plt.close(fig)
print('Fig2a saved')

# ════════════════════════════════════════
# 图2b: 最佳停药时间分布
# ════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(t_stops, bins=25, color='#5e3c99', edgecolor='white', alpha=0.8)
ax.axvline(x=np.median(t_stops), color='black', linestyle='--', linewidth=2, label=f'Median={np.median(t_stops):.0f}w')
ax.axvline(x=np.mean(t_stops), color='red', linestyle='-', linewidth=2, label=f'Mean={np.mean(t_stops):.0f}w')
ax.set_xlabel('Optimal Stop Time (weeks)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Optimal Treatment Duration\n最佳停药时间分布 (mean={np.mean(t_stops):.0f}±{np.std(t_stops):.0f}w)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(r'E:\AI Switch\works\tb_therapy\fig2b_stop_time.png', dpi=200)
plt.close(fig)
print('Fig2b saved')

# ════════════════════════════════════════
# 图2c: 治疗响应率饼图
# ════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
improved = sum(1 for c in cd4_changes if c > 0)
declined = sum(1 for c in cd4_changes if c <= 0)
colors_pie = ['#2c7bb6', '#d7191c']
ax.pie([improved, declined], labels=[f'Improved\n{improved}', f'Declined\n{declined}'],
       colors=colors_pie, autopct='%1.1f%%', startangle=90, explode=(0.02, 0))
ax.set_title('Treatment Response Rate\n治疗响应率', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(r'E:\AI Switch\works\tb_therapy\fig2c_response_pie.png', dpi=200)
plt.close(fig)
print('Fig2c saved')

# ════════════════════════════════════════
# 图3a: 全体CD4平均趋势
# ════════════════════════════════════════
all_cd4_t, all_cd4_v = [], []
for d in data.values():
    all_cd4_t.extend(d['cd4_t'])
    all_cd4_v.extend(d['cd4_v'])
all_cd4_t = np.array(all_cd4_t)
all_cd4_v = np.array(all_cd4_v)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(all_cd4_t, all_cd4_v, alpha=0.1, s=15, c='#2c7bb6')
bins = np.arange(0, 60, 5)
bmeans, bstds, bcents = [], [], []
for i in range(len(bins)-1):
    m = (all_cd4_t >= bins[i]) & (all_cd4_t < bins[i+1])
    if m.sum() > 10:
        bmeans.append(all_cd4_v[m].mean())
        bstds.append(all_cd4_v[m].std())
        bcents.append((bins[i]+bins[i+1])/2)
bmeans, bstds, bcents = map(np.array, [bmeans, bstds, bcents])
ax.errorbar(bcents, bmeans, yerr=bstds, fmt='o-', color='#d7191c', lw=2, ms=8, capsize=3, label='Mean ± SD')
ax.set_xlabel('Week', fontsize=12)
ax.set_ylabel('CD4 Count', fontsize=12)
ax.set_title('Population CD4 Trend\n全体患者 CD4 平均趋势', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(r'E:\AI Switch\works\tb_therapy\fig3a_cd4_trend.png', dpi=200)
plt.close(fig)
print('Fig3a saved')

# ════════════════════════════════════════
# 图3b: 全体VLoad平均趋势
# ════════════════════════════════════════
all_vl_t, all_vl_v = [], []
for d in data.values():
    all_vl_t.extend(d['vl_t']); all_vl_v.extend(d['vl_v'])
all_vl_t = np.array(all_vl_t); all_vl_v = np.array(all_vl_v)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(all_vl_t, all_vl_v, alpha=0.1, s=15, c='#d7191c')
bvmeans, bvcents = [], []
for i in range(len(bins)-1):
    m = (all_vl_t >= bins[i]) & (all_vl_t < bins[i+1])
    if m.sum() > 10:
        bvmeans.append(all_vl_v[m].mean())
        bvcents.append((bins[i]+bins[i+1])/2)
bvmeans, bvcents = map(np.array, [bvmeans, bvcents])
ax.plot(bvcents, bvmeans, 'o-', color='#2c7bb6', lw=2, ms=8, label='Mean')
ax.set_xlabel('Week', fontsize=12)
ax.set_ylabel('VLoad (log)', fontsize=12)
ax.set_title('Population VLoad Trend\n全体患者病毒浓度平均趋势', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(r'E:\AI Switch\works\tb_therapy\fig3b_vload_trend.png', dpi=200)
plt.close(fig)
print('Fig3b saved')

# ── 保存结果 ──
summary = {
    'n_total': len(data),
    'n_valid': len(results),
    'cd4_mean_change': round(float(np.mean(cd4_changes)), 1),
    'cd4_std_change': round(float(np.std(cd4_changes)), 1),
    'cd4_mean_pct': round(float(np.mean(cd4_pct_changes)), 1),
    'cd4_improve_rate': round(float(sum(1 for c in cd4_changes if c > 0) / len(cd4_changes) * 100), 1),
    'vl_mean_reduction': round(float(np.mean(vl_changes)), 2),
    'vl_improve_rate': round(float(sum(1 for v in vl_changes if v > 0) / len(vl_changes) * 100), 1),
    't_stop_mean': round(float(np.mean(t_stops)), 1),
    't_stop_median': round(float(np.median(t_stops)), 1),
    't_stop_std': round(float(np.std(t_stops)), 1),
    't_stop_min': round(float(np.min(t_stops)), 0),
    't_stop_max': round(float(np.max(t_stops)), 0),
    'cd4_r2_mean': round(float(np.mean([r['cd4_r2'] for r in results])), 3),
    'recommendation': f"建议治疗 {np.median(t_stops):.0f} 周（约 {np.median(t_stops)/4:.1f} 个月），范围 {np.min(t_stops):.0f}-{np.max(t_stops):.0f} 周",
    'improved_pct': round(float(sum(1 for c in cd4_changes if c > 0) / len(cd4_changes) * 100), 1),
    'declined_pct': round(float(sum(1 for c in cd4_changes if c <= 0) / len(cd4_changes) * 100), 1),
}
with open(r'E:\AI Switch\works\tb_therapy\results_q1.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\n=== 结果已保存 ===")
for k, v in summary.items():
    print(f"  {k}: {v}")
print("Done.")
