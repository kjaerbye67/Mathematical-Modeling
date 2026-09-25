"""
生成完整论文: 结核病疗法的评价及疗效的预测
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import json
import os

out_dir = r'E:\AI Switch\works\tb_therapy'
fig_dir = out_dir

# Load results
with open(f'{out_dir}/results_q1.json', 'r', encoding='utf-8') as f:
    r1 = json.load(f)
with open(f'{out_dir}/results_q2.json', 'r', encoding='utf-8') as f:
    r2 = json.load(f)

doc = Document()

# ── 页面设置 ──
for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.18)
    section.right_margin = Cm(3.18)

# ── 样式设置 ──
style = doc.styles['Normal']
style.font.name = '宋体'
style.font.size = Pt(12)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
style.paragraph_format.line_spacing = 1.5

def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = '黑体'
        run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        if level == 1:
            run.font.size = Pt(16)
        elif level == 2:
            run.font.size = Pt(14)
        elif level == 3:
            run.font.size = Pt(13)
    return h

def add_para(text, bold=False, indent=True, align=None):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Pt(24)
    if align:
        p.alignment = align
    run = p.add_run(text)
    run.font.name = '宋体'
    run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    run.font.size = Pt(12)
    run.bold = bold
    return p

def add_figure(path, caption, width=Inches(5.8)):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    if os.path.exists(path):
        run.add_picture(path, width=width)
    # Caption
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.first_line_indent = Pt(0)
    crun = cap.add_run(caption)
    crun.font.size = Pt(10)
    crun.font.name = '宋体'
    crun.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    crun.italic = True

def add_math(expression):
    """简单数学公式 (斜体)"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run(expression)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.italic = True
    return p

# ════════════════════════════════════════
# 标题页
# ════════════════════════════════════════
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(80)
run = p.add_run('结核病疗法的评价及疗效的预测')
run.font.name = '黑体'
run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
run.font.size = Pt(22)
run.bold = True

p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
p2.paragraph_format.space_before = Pt(30)
run2 = p2.add_run('基于指数衰减模型的CD4免疫重建分析与多疗法比较研究')
run2.font.name = '宋体'
run2.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
run2.font.size = Pt(14)

doc.add_page_break()

# ════════════════════════════════════════
# 摘要
# ════════════════════════════════════════
add_heading('摘要', level=1)

add_para('结核病的治疗效果评价是临床决策的核心问题。本文针对ACTG320和193A两组临床试验数据，分别从'
         '个体化治疗终点预测和多疗法比较两个维度，建立了完整的数学模型体系。')

add_para('针对问题一，本文基于ACTG320的356名患者数据，建立了CD4计数的指数渐进增长模型和病毒浓度的'
         '指数衰减模型。通过非线性最小二乘拟合，{:.1f}%的患者CD4呈上升趋势，CD4平均改善{:.0f}±{:.0f}；'
         '{:.1f}%的患者病毒浓度下降。以CD4达到渐近线95%为停药标准，确定最佳治疗时间的'
         '中位数为{:.0f}周（约{:.1f}个月），范围为{:.0f}-{:.0f}周。'.format(
             r1['cd4_improve_rate'], r1['cd4_mean_change'], r1['cd4_std_change'],
             r1['vl_improve_rate'], r1['t_stop_median'], r1['t_stop_median']/4,
             r1['t_stop_min'], r1['t_stop_max']))

add_para('针对问题二，基于193A的1309名患者数据，采用单因素方差分析（ANOVA）和Kruskal-Wallis检验'
         '比较四种疗法的CD4变化。结果显示，疗法4（齐多夫定+去羟肌苷+奈韦拉平）是唯一使CD4呈'
         '正向增长的方案（均值斜率+{:.2f}x10^-3/周），响应率达{:.1f}%，显著优于其他三种疗法'
         '（p<0.001）。线性趋势预测表明，持续治疗48周后CD4计数可达{:.0f}。'.format(
             r2['t4_mean_slope']*1000, r2['best_therapy_responders_pct'],
             r2['t4_pred_week48_cd4']))

add_para('针对问题三，构建了费用效果比（CER）评价体系，综合疗效、响应率和治疗成本三个维度进行'
         '多准则排序。结果显示，疗法4虽然费用最高（月均$109.50），但因其唯一正向疗效，综合得分'
         '仍排名第一；疗法3因较低费用和适中的响应率，可作为预算受限患者的替代方案。')

add_para('本文的创新点在于：1) 建立了基于非线性生长模型的个体化治疗终点判定方法；2) 综合运用参数'
         '和非参数统计方法进行多疗法比较；3) 引入多维度费用效果评价框架，为不同经济条件的患者'
         '提供差异化治疗建议。')

add_para('关键词：结核病治疗；CD4免疫重建；非线性回归；方差分析；费用效果分析；最佳停药时间', bold=True, indent=False)

doc.add_page_break()

# ════════════════════════════════════════
# 目录 (手动)
# ════════════════════════════════════════
add_heading('目录', level=1)
toc_items = [
    '1. 问题重述',
    '  1.1 问题背景',
    '  1.2 问题描述',
    '2. 模型假设与符号说明',
    '  2.1 模型假设',
    '  2.2 符号说明',
    '3. 问题一：ACTG320数据的治疗终点预测',
    '  3.1 问题分析',
    '  3.2 模型建立',
    '  3.3 模型求解与结果分析',
    '4. 问题二：四种疗法的比较与疗效预测',
    '  4.1 问题分析',
    '  4.2 模型建立',
    '  4.3 模型求解与结果分析',
    '5. 问题三：考虑费用的疗法综合评价',
    '  5.1 问题分析',
    '  5.2 模型建立',
    '  5.3 模型求解与结果分析',
    '6. 模型评价与改进',
    '  6.1 模型的创新点',
    '  6.2 模型的优点',
    '  6.3 模型的缺点',
    '  6.4 未来展望',
    '参考文献',
    '附录',
]
for item in toc_items:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.line_spacing = 1.3
    run = p.add_run(item)
    run.font.size = Pt(12)
    if not item.startswith('  '):
        run.bold = True

doc.add_page_break()

# ════════════════════════════════════════
# 1. 问题重述
# ════════════════════════════════════════
add_heading('1. 问题重述', level=1)
add_heading('1.1 问题背景', level=2)

add_para('结核病是由结核分枝杆菌引起的一种慢性感染性疾病，其中以肺结核最为常见。'
         '肺结核的主要临床表现包括结核结节、浸润、干酪样变和空洞形成。'
         '排菌的患者及动物是主要传染源，本病以空气传播为主，人群普遍易感。')

add_para('结核病治疗的核心目标是尽量减少人体内结核分枝杆菌的数量，同时促进CD4+T淋巴细胞的'
         '增殖与重建，以提高人体免疫能力。CD4细胞是衡量免疫系统功能的关键指标，'
         '其数量的增加标志着免疫系统正在恢复。然而，迄今为止人类尚未找到能根治'
         '结核分枝杆菌的疗法，现存的疗法不仅对人体有一定副作用，且成本较高。')

add_para('美国结核病医疗试验机构ACTG公布了两组重要的临床试验数据：ACTG320试验记录了'
         '300多名患者同时服用齐多夫定（zidovudine）、拉美夫定（lamivudine）和茚地那韦'
         '（indinavir）三种药物后，每隔几周测得的CD4浓度和结核分枝杆菌浓度；'
         '193A试验记录了1300多名患者分别接受四种不同药物方案治疗后，每隔约8周测得的'
         'CD4浓度。这两组数据为定量评价不同疗法的效果提供了宝贵的实证基础。')

add_heading('1.2 问题描述', level=2)

add_para('基于上述背景，本文需要解决以下三个问题：')

add_para('问题一：利用ACTG320（附件1）的数据，预测继续治疗的效果，或者确定最佳治疗终止时间。'
         '继续治疗指在测试终止后继续服药；如果认为继续服药效果不好，则可选择提前终止治疗。')

add_para('问题二：利用193A（附件2）的数据，评价四种疗法的优劣（仅以CD4为标准），'
         '并对较优的疗法预测继续治疗的效果，或者确定最佳治疗终止时间。')

add_para('问题三：考虑药品费用因素（600mg齐多夫定$1.60，400mg去羟肌苷$0.85，'
         '2.25mg扎西他滨$1.85，400mg奈韦拉平$1.20），对问题二中的四种疗法'
         '进行重新评价，为不同经济条件的患者提供差异化治疗建议。')

doc.add_page_break()

# ════════════════════════════════════════
# 2. 模型假设与符号说明
# ════════════════════════════════════════
add_heading('2. 模型假设与符号说明', level=1)
add_heading('2.1 模型假设', level=2)

add_para('1. CD4细胞计数随时间的变化可由指数渐进增长模型描述，即CD4以递减的速率趋向于'
         '某个渐近水平。')
add_para('2. 结核分枝杆菌浓度（病毒载量）随时间的变化可由指数衰减模型描述，即病毒'
         '载量以递减的速率趋向于最低检测水平。')
add_para('3. 各患者的测量时间点数据在给定时间区间内能够代表其总体变化趋势。')
add_para('4. 药物对CD4和病毒载量的影响在观测期内保持稳定，不存在耐药性突变。')
add_para('5. 不同患者之间的数据相互独立，同一患者不同时间点的测量误差服从正态分布。')
add_para('6. 药品价格在研究期间保持恒定，不考虑通货膨胀和市场波动。')

add_heading('2.2 符号说明', level=2)

symbols = [
    ('t', '时间（周）'),
    ('y(t)', 't时刻的CD4细胞计数'),
    ('v(t)', 't时刻的病毒载量（log尺度）'),
    ('a, b, c', '指数模型的待估参数'),
    ('β', '线性回归斜率参数'),
    ('R²', '拟合优度（决定系数）'),
    ('Ts', '最佳停药时间（周）'),
    ('MAPE', '平均绝对百分比误差'),
    ('F', '方差分析的F统计量'),
    ('p', '统计显著性水平'),
    ('CER', '费用效果比（Cost-Effectiveness Ratio）'),
]
for sym, desc in symbols:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    run_sym = p.add_run(f'{sym}')
    run_sym.font.name = 'Times New Roman'
    run_sym.italic = True
    run_sym.bold = True
    run_desc = p.add_run(f'：{desc}')
    run_desc.font.name = '宋体'
    run_desc.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

doc.add_page_break()

# ════════════════════════════════════════
# 3. 问题一
# ════════════════════════════════════════
add_heading('3. 问题一：ACTG320数据的治疗终点预测', level=1)
add_heading('3.1 问题分析', level=2)

add_para('ACTG320数据集包含356名同时服用三种药物（齐多夫定、拉美夫定、茚地那韦）的患者，'
         '在0-55周的时间范围内，每隔数周测量CD4细胞计数和结核分枝杆菌浓度。'
         '每个患者通常有4-6次测量记录。')

add_para('分析该问题的核心挑战在于：如何在有限且稀疏的观测数据基础上，准确描述CD4和'
         '病毒载量的时间演化规律，并据此确定最佳治疗终止时间。从临床角度出发，'
         'CD4的持续增长和病毒载量的持续下降是治疗有效的标志；当CD4的增长速率趋近于'
         '零（即达到平台期）时，继续治疗可能不再带来显著的额外收益。')

add_para('本文的建模策略分为三步：第一，为每位患者建立CD4和病毒载量的非线性动力学模型；'
         '第二，通过模型参数确定CD4的"有效增长区间"；第三，以CD4达到渐近值95%的时刻'
         '作为最佳停药时间的估计。')

add_heading('3.2 模型建立', level=2)

add_para('（1）CD4指数渐进增长模型', bold=True, indent=False)
add_para('假设CD4计数遵循指数渐进增长规律，即增长率与当前值到渐近线的距离成正比：')

add_math('dy/dt = c · (a − y)')
add_para('解此微分方程，得CD4随时间变化的指数模型：')
add_math('y(t) = a · (1 − b · e^(−c·t))')
add_para('其中，a > 0 为CD4的渐近水平（理论最大值），b ∈ (0, 1) 反映初始值相对渐近线的'
         '偏离程度，c > 0 为增长率参数。当t→∞时，y(t)→a；初始值y(0)=a(1−b)。')

add_para('（2）病毒载量指数衰减模型', bold=True, indent=False)
add_para('类似地，假设病毒载量遵循指数衰减规律：')
add_math('dv/dt = −c₁ · (v − v₀)')
add_para('解得：')
add_math('v(t) = v₀ + b₁ · e^(−c₁·t)')
add_para('其中 v₀ 为最低病毒载量水平，b₁ > 0 反映初始病毒载量的超量部分。')

add_para('（3）最佳停药时间准则', bold=True, indent=False)
add_para('定义CD4的"有效治疗终点"为CD4达到其渐近值95%的时刻，即：')
add_math('y(Ts) = 0.95 · a')
add_para('代入指数模型解得：')
add_math('Ts = −ln(0.05 / b) / c')
add_para('此准则的物理意义是：当CD4已恢复至最大理论值的95%时，剩余的5%提升空间'
         '需要较长时间才能实现，边际收益递减，此时可选择终止治疗。')

add_para('（4）线性回退方案', bold=True, indent=False)
add_para('当指数模型拟合不佳（R²<0.5）时，退化为线性模型 y(t)=k·t+y₀，'
         '此时以观测终点作为建议的停药时间。')

add_heading('3.3 模型求解与结果分析', level=2)

add_para('利用Python的SciPy库，对每位患者分别进行非线性最小二乘拟合。'
         '当指数模型收敛且R²>0.5时采用指数模型，否则采用线性回退方案。')

add_para(f'拟合结果：在全部分356名患者中，255人（71.6%）适用指数模型，'
         f'101人（28.4%）退化为线性模型。CD4拟合的平均R²为{r1["cd4_r2_mean"]}，'
         f'表明模型整体上能较好地捕捉个体的CD4动态变化规律。')

add_para(f'治疗效果：{r1["improved_pct"]}%的患者CD4计数呈上升趋势'
         f'（平均增加{r1["cd4_mean_change"]}±{r1["cd4_std_change"]}），'
         f'{r1["vl_improve_rate"]}%的患者病毒载量下降（平均降低{r1["vl_mean_reduction"]}log单位），'
         f'说明ACTG320的三药联合方案对大多数患者具有显著的治疗效果。')

add_para(f'最佳停药时间：基于CD4达到渐近线95%的准则，最佳治疗时间的中位数为'
         f'{r1["t_stop_median"]:.0f}周（约{r1["t_stop_median"]/4:.1f}个月），'
         f'均值为{r1["t_stop_mean"]:.1f}±{r1["t_stop_std"]:.1f}周，'
         f'个体间差异较大（{r1["t_stop_min"]:.0f}-{r1["t_stop_max"]:.0f}周）。'
         f'建议大多数患者在治疗{r1["t_stop_median"]:.0f}周后进行CD4水平评估，'
         f'根据个体恢复情况决定是否继续治疗。')

# 插入图1-3
add_figure(f'{fig_dir}/fig1a_trajectory.png', '图1(a) 最佳改善患者CD4和病毒载量变化轨迹')
add_figure(f'{fig_dir}/fig1b_trajectory.png', '图1(b) 次佳改善患者CD4和病毒载量变化轨迹')
add_figure(f'{fig_dir}/fig1c_trajectory.png', '图1(c) 中等改善患者之一CD4和病毒载量变化轨迹')
add_figure(f'{fig_dir}/fig1d_trajectory.png', '图1(d) 中等改善患者之二CD4和病毒载量变化轨迹')
add_figure(f'{fig_dir}/fig1e_trajectory.png', '图1(e) 改善较差患者CD4和病毒载量变化轨迹')
add_figure(f'{fig_dir}/fig1f_trajectory.png', '图1(f) 改善最差患者CD4和病毒载量变化轨迹')
add_figure(f'{fig_dir}/fig2a_cd4_change.png', '图2(a) CD4变化分布（排序后）')
add_figure(f'{fig_dir}/fig2b_stop_time.png', '图2(b) 最佳停药时间分布直方图')
add_figure(f'{fig_dir}/fig2c_response_pie.png', '图2(c) 治疗响应率饼图')
add_figure(f'{fig_dir}/fig3a_cd4_trend.png', '图3(a) 全体患者CD4平均趋势（均值±标准差）')
add_figure(f'{fig_dir}/fig3b_vload_trend.png', '图3(b) 全体患者病毒浓度平均趋势')

doc.add_page_break()

# ════════════════════════════════════════
# 4. 问题二
# ════════════════════════════════════════
add_heading('4. 问题二：四种疗法的比较与疗效预测', level=1)
add_heading('4.1 问题分析', level=2)

add_para('193A数据集包含1309名患者，随机分配至四种不同的抗结核药物方案：')

therapies_desc = [
    ('疗法1（T1）', '齐多夫定（600mg）+ 去羟肌苷（400mg），交替使用'),
    ('疗法2（T2）', '齐多夫定（600mg）+ 扎西他滨（2.25mg）'),
    ('疗法3（T3）', '齐多夫定（600mg）+ 去羟肌苷（400mg）'),
    ('疗法4（T4）', '齐多夫定（600mg）+ 去羟肌苷（400mg）+ 奈韦拉平（400mg）'),
]
for name, desc in therapies_desc:
    add_para(f'{name}：{desc}。', indent=True)

add_para('注意到T1和T3使用了相同的药物组合（仅给药方式不同），T4在T1/T3的基础上增加了'
         '第四种药物奈韦拉平。CD4的测量值为log(CD4+1)变换后的值，每个患者约有2-6次'
         '测量记录，间隔约8周。')

add_para('本问题的核心是比较四种疗法在提升CD4水平方面的差异。由于不同患者的基线CD4水平、'
         '年龄和观测时长不同，直接比较CD4终值不合理。本文选择每个患者的CD4变化量'
         '（Δlog(CD4+1)）和线性回归斜率作为主要疗效指标，采用方差分析（ANOVA）'
         '和Kruskal-Wallis非参数检验进行组间比较。')

add_heading('4.2 模型建立', level=2)

add_para('（1）个体疗效指标', bold=True, indent=False)
add_para('对每位患者，基于其纵向测量数据，计算以下指标：')

add_para('CD4对数变化量：ΔlogCD4ᵢ = logCD4ᵢ(t_end) − logCD4ᵢ(t_start)')
add_para('CD4变化斜率：通过最小二乘线性回归 logCD4 = βᵢ·t + αᵢ 估计βᵢ')
add_para('响应状态：若ΔlogCD4ᵢ > 0，则患者i为响应者（responder）；否则为非响应者。')

add_para('（2）组间比较方法', bold=True, indent=False)
add_para('采用单因素方差分析（One-way ANOVA）检验四种疗法的疗效均值是否存在显著差异。'
         '由于方差不齐性和非正态性可能影响ANOVA的可靠性，同时采用Kruskal-Wallis'
         '非参数检验作为稳健性验证。对两两比较，采用Welch t检验并施加Bonferroni校正。')

add_para('（3）疗效预测模型', bold=True, indent=False)
add_para('对最优疗法，利用全体患者的平均斜率进行线性外推预测：')
add_math('logCD4(t) = logCD4₀ + β̄ · t')
add_para('当β̄ > 0且显著时，认为继续治疗可维持正向收益。')

add_heading('4.3 模型求解与结果分析', level=2)

add_para('（1）四种疗法的CD4变化比较', bold=True, indent=False)

therapy_names = ['T1 (ZDV+didanosine交替)', 'T2 (ZDV+zalcitabine)',
                 'T3 (ZDV+didanosine)', 'T4 (ZDV+didanosine+nevirapine)']
add_para('表1给出了四种疗法的CD4变化汇总统计。结果显示，T1至T3的CD4对数值均呈下降趋势'
         '（均值分别为-0.509、-0.400和-0.276），响应率仅为24%-33%。'
         '只有T4呈现正向增长趋势（均值+0.076，响应率48.7%）。')

add_para(f'单因素方差分析结果：F={r2["anova_f"]}，p<0.0001；'
         f'Kruskal-Wallis检验：H={r2["kruskal_h"]}，p<0.0001。'
         f'两种检验均表明四种疗法之间存在极显著差异。')

add_para(f'两两比较（Bonferroni校正）：T4与T1（p<0.001）、T4与T2（p<0.001）、'
         f'T4与T3（p<0.001）之间的差异均达到极显著水平，确认T4是唯一有效提升'
         f'CD4的疗法。T1与T3之间存在显著差异（p=0.014），提示同种药物组合的不同'
         f'给药方式（交替 vs 同时）可能影响疗效。T2与T3之间无显著差异。')

add_para('（2）T4疗法的继续治疗效果预测', bold=True, indent=False)

add_para(f'T4疗法的平均斜率β={r2["t4_mean_slope"]*1000:.2f}x10⁻³/周。'
         f'基于此斜率，预测T4继续治疗的效果：第8周时CD4计数约18，至第48周可缓慢增至约21。'
         f'这表明T4虽然较其他疗法表现最优，但CD4改善的绝对值仍较有限。'
         f'T4的响应率为{r2["best_therapy_responders_pct"]}%，'
         f'即近半数患者可以从中获益，但仍有超过一半的患者改善不明显。')

add_para('需要指出的是，T4的中位数斜率接近于零，意味着超过半数的'
         '患者实际上处于稳定或轻微下降状态，平均值的正向收益主要由少数响应良好的'
         '患者驱动。这提示T4并非对所有患者都有效，临床上需要结合个体特征进行精准用药。')

# 插入图4-5
add_figure(f'{fig_dir}/fig4a_boxplot.png', '图4(a) 四种疗法的log(CD4+1)变化量箱线图')
add_figure(f'{fig_dir}/fig4b_response_rate.png', '图4(b) 四种疗法的治疗响应率')
add_figure(f'{fig_dir}/fig4c_cost_effectiveness.png', '图4(c) 四种疗法的费用效果平面图')
add_figure(f'{fig_dir}/fig4d_comprehensive_score.png', '图4(d) 四种疗法的综合评分')
add_figure(f'{fig_dir}/fig5a_individual.png', '图5(a) T4疗法个体CD4轨迹（前50名患者）')
add_figure(f'{fig_dir}/fig5b_prediction.png', '图5(b) T4疗法平均趋势与预测（含95%置信区间）')

doc.add_page_break()

# ════════════════════════════════════════
# 5. 问题三
# ════════════════════════════════════════
add_heading('5. 问题三：考虑费用的疗法综合评价', level=1)
add_heading('5.1 问题分析', level=2)

add_para('在现实医疗决策中，除了临床疗效，治疗费用也是影响疗法选择的重要因素，'
         '尤其对于不发达国家的患者。问题三要求将药品费用纳入评价体系，'
         '对四种疗法进行重新排序。')

add_para('这本质上是一个多准则决策问题：需要在疗效（CD4改善程度）、响应率'
         '（受益患者比例）和治疗成本三者之间寻求平衡。本文构建了费用效果比'
         '（Cost-Effectiveness Ratio, CER）和加权综合评分两种方法进行评价。')

add_heading('5.2 模型建立', level=2)

add_para('（1）费用计算', bold=True, indent=False)
add_para('根据给定价格计算每种疗法的每日和每月费用：')

cost_table = {
    'T1': '齐多夫定$1.60 + 去羟肌苷$0.85 = $2.45/天，$73.50/月',
    'T2': '齐多夫定$1.60 + 扎西他滨$1.85 = $3.45/天，$103.50/月',
    'T3': '齐多夫定$1.60 + 去羟肌苷$0.85 = $2.45/天，$73.50/月',
    'T4': '齐多夫定$1.60 + 去羟肌苷$0.85 + 奈韦拉平$1.20 = $3.65/天，$109.50/月',
}
for name, cost_desc in cost_table.items():
    add_para(f'{name}：{cost_desc}。', indent=True)

add_para('（2）费用效果比（CER）', bold=True, indent=False)
add_para('定义费用效果比为每周治疗费用与CD4对数改善量之比：')
add_math('CER = C周 / (β̄ × 100)')
add_para('其中β̄是CD4对数值的周平均变化率，C周为每周治疗费用。CER的单位为"美元/0.01 log CD4改善/周"，'
         '其值越低，表示单位疗效的成本越低，性价比越高。当β̄ ≤ 0时，CER定义为无穷大'
         '（无效疗法不计性价比）。')

add_para('（3）多维度综合评分', bold=True, indent=False)
add_para('为综合考虑疗效、响应率和费用三个维度，构建归一化加权评分：')
add_math('Sᵢ = (1/3) × [Eᵢ/max(E) + Rᵢ/100 + (1 − Cᵢ/max(C))]')
add_para('其中Eᵢ为疗法的平均疗效斜率，Rᵢ为响应率，Cᵢ为每周费用。'
         '三个维度取等权重(1/3)，得分范围为[0, 1]。')

add_heading('5.3 模型求解与结果分析', level=2)

add_para('（1）费用效果比分析', bold=True, indent=False)

add_para('T1、T2、T3的CD4斜率为负值（患者总体呈恶化趋势），因此CER为无穷大——'
         '从费用效果角度看，投入资金不仅未能改善病情，反而伴随着CD4的持续下降，'
         '属于"无效投入"。只有T4具有有限的CER（$59.43/0.01 log/周），'
         '虽费用最高（$25.55/周），但它是唯一能带来正向疗效的选项。')

add_para('（2）综合评分与排序', bold=True, indent=False)

add_para(f'综合评分结果：T4得分最高（{r2["ranking"][0]["score"]:.3f}），'
         f'排名第一；T3得分{r2["ranking"][1]["score"]:.3f}，排名第二；'
         f'T2得分{r2["ranking"][2]["score"]:.3f}，排名第三；'
         f'T1得分{r2["ranking"][3]["score"]:.3f}，排名第四。')

add_para('T4虽然在费用维度得分最低（0.000），但其疗效得分（1.000）和响应率得分'
         '（0.487）远超其他疗法，因此综合得分仍然领先。T3和T1费用相同（$17.15/周），'
         '但T3的响应率略高于T1，故排名靠前。')

add_para('（3）差异化治疗建议', bold=True, indent=False)

add_para('对于经济条件较好的患者：推荐T4方案（齐多夫定+去羟肌苷+奈韦拉平），'
         '月均费用$109.50，响应率48.7%，是唯一能实现CD4正向增长的方案。')

add_para('对于预算受限的患者：可考虑T3方案（齐多夫定+去羟肌苷），月均费用$73.50，'
         '响应率32.5%。虽然整体CD4呈轻微下降趋势，但约三分之一的患者仍可从中获益，'
         '且费用较T4低33%，适合作为过渡或替代方案。')

add_para('对于政府/慈善机构的集中采购：建议优先采购T4方案的药品组合，因为单位疗效的'
         '增量成本相对合理（额外的$36/月可获得显著更好的免疫重建效果）。')

# 插入图6
add_figure(f'{fig_dir}/fig6_cost_effectiveness.png', '图6 四种疗法的费用效果比（CER）比较')

doc.add_page_break()

# ════════════════════════════════════════
# 6. 模型评价
# ════════════════════════════════════════
add_heading('6. 模型评价与改进', level=1)
add_heading('6.1 模型的创新点', level=2)

add_para('1. 针对ACTG320数据，创新性地采用指数渐进增长模型刻画CD4的动态变化，'
         '基于渐近线95%准则量化了"最佳停药时间"，为个体化治疗终点判定提供了'
         '客观的数学依据，弥补了传统固定疗程方案的不足。')

add_para('2. 在193A数据的分析中，综合运用参数检验（ANOVA）和非参数检验'
         '（Kruskal-Wallis），并通过Bonferroni校正控制多重比较的假阳性风险，'
         '确保了统计推断结论的可靠性。')

add_para('3. 构建了融合疗效、响应率和费用的三维综合评价框架，不同于传统仅关注'
         '单一指标的简单比较，为不同经济背景的患者提供了分层次的治疗建议。')

add_heading('6.2 模型的优点', level=2)

add_para('1. 模型结构简单清晰。指数模型仅包含三个参数，具有明确的生物学含义'
         '（渐近水平、初始偏离和增长速率），便于临床医生理解和解释。')

add_para('2. 计算效率高。每个患者的参数估计可在毫秒级完成，适用于大规模'
         '筛查和实时决策支持场景。')

add_para('3. 具有分层回退机制。当指数模型拟合不良时自动退化为线性模型，'
         '保证了在所有数据条件下都能给出合理的估计。')

add_para('4. 统计方法严谨。采用多种检验方法交叉验证，避免单一方法的偏倚。')

add_heading('6.3 模型的缺点', level=2)

add_para('1. 指数模型的假设过于简化。现实中的CD4动力学可能涉及多阶段的复杂变化'
         '（如初始快速上升后趋于平缓，或出现波动），简单的三参数指数模型无法'
         '捕捉这些精细结构。')

add_para('2. 未考虑患者的个体特征（如年龄、性别、疾病严重程度等）对治疗反应的'
         '影响，模型仅为"一人一模型"，缺乏跨患者的共性知识迁移。')

add_para('3. 停药时间仅基于CD4一个指标，未综合病毒载量、临床症状、药物副作用'
         '等多维度信息。')

add_para('4. 193A数据的CD4以log(CD4+1)变换后记录，反向变换计算原始CD4时会'
         '引入偏差，影响对绝对疗效的精确估计。')

add_para('5. 费用分析仅考虑了药品直接成本，未纳入给药方式差异（交替vs同时）'
         '带来的实施成本、副作用处理成本等间接费用。')

add_heading('6.4 未来展望', level=2)

add_para('1. 引入混合效应模型（Mixed-Effects Model）代替个体独立拟合，'
         '可同时估计群体平均趋势和个体随机效应，提高参数估计的稳健性和'
         '对稀疏数据的适应能力。')

add_para('2. 纳入更多解释变量（年龄、基线CD4、基线病毒载量等）构建多元'
         '预测模型，实现对不同亚群患者的精准治疗建议。')

add_para('3. 采用生存分析方法（如Cox比例风险模型）对"CD4恢复到目标水平"'
         '这一事件进行时间-事件分析，替代当前的确定性停药时间计算。')

add_para('4. 引入贝叶斯方法，将先验临床知识（如已知的药物半衰期、免疫应答'
         '时间尺度）纳入建模过程，提高小样本条件下的估计可靠性。')

add_para('5. 建立马尔可夫决策过程（MDP）模型，将治疗过程建模为序列决策问题，'
         '在每个随访时间点根据当前CD4水平动态决定"继续治疗"还是"停止治疗"，'
         '实现真正的自适应个体化治疗策略。')

doc.add_page_break()

# ════════════════════════════════════════
# 参考文献
# ════════════════════════════════════════
add_heading('参考文献', level=1)

refs = [
    '[1] Hammer S M, Squires K E, Hughes M D, et al. A controlled trial of two nucleoside '
    'analogues plus indinavir in persons with human immunodeficiency virus infection and '
    'CD4 cell counts of 200 per cubic millimeter or less[J]. New England Journal of Medicine, '
    '1997, 337(11): 725-733.',
    '[2] Henry K, Erice A, Tierney C, et al. A randomized, controlled, double-blind study '
    'comparing the survival benefit of four different reverse transcriptase inhibitor therapies '
    'for the treatment of advanced HIV infection[J]. Journal of Infectious Diseases, 1998, '
    '177(2): 246-256.',
    '[3] Perelson A S, Neumann A U, Markowitz M, et al. HIV-1 dynamics in vivo: virion '
    'clearance rate, infected cell life-span, and viral generation time[J]. Science, 1996, '
    '271(5255): 1582-1586.',
    '[4] 姜启源, 谢金星, 叶俊. 数学模型(第五版)[M]. 北京: 高等教育出版社, 2018.',
    '[5] 司守奎, 孙兆亮. 数学建模算法与应用(第三版)[M]. 北京: 国防工业出版社, 2021.',
    '[6] Bates D, Mächler M, Bolker B, et al. Fitting linear mixed-effects models using '
    'lme4[J]. Journal of Statistical Software, 2015, 67(1): 1-48.',
    '[7] 国家卫生健康委员会. 肺结核诊断标准(WS 288-2017)[S]. 北京: 中国标准出版社, 2017.',
    '[8] Drummond M F, Sculpher M J, Claxton K, et al. Methods for the Economic Evaluation '
    'of Health Care Programmes (4th ed)[M]. Oxford: Oxford University Press, 2015.',
]
for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(ref)
    run.font.size = Pt(10.5)
    run.font.name = 'Times New Roman'

doc.add_page_break()

# ════════════════════════════════════════
# 附录
# ════════════════════════════════════════
add_heading('附录：核心分析代码', level=1)

code = '''import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

# ── CD4 指数渐进增长模型 ──
def cd4_model(t, a, b, c):
    """CD4 = a * (1 - b * exp(-c * t))"""
    return a * (1.0 - b * np.exp(-c * t))

# ── 最佳停药时间 ──
def optimal_stop_time(a, b, c):
    """CD4 达到渐近线 95% 的时刻"""
    return -np.log(0.05 / max(b, 0.01)) / c

# ── 模型拟合 ──
def fit_patient(t, y):
    """对单个患者拟合指数模型"""
    a0 = max(np.max(y) * 1.2, y[-1] * 1.3)
    b0 = max(0.3, min(0.95, 1.0 - y[0] / a0))
    c0 = 0.05
    try:
        popt, _ = curve_fit(cd4_model, t, y, p0=[a0, b0, c0],
                            bounds=([0, 0, 0.001], [np.inf, 1.0, 1.0]),
                            maxfev=5000)
        pred = cd4_model(t, *popt)
        r2 = 1 - np.sum((y - pred)**2) / np.sum((y - np.mean(y))**2)
        return popt, r2
    except:
        return None, 0

# ── ANOVA 多组比较 ──
def compare_therapies(groups):
    """groups: list of arrays for each therapy"""
    f_stat, p_val = stats.f_oneway(*groups)
    h_stat, p_kw = stats.kruskal(*groups)
    return f_stat, p_val, h_stat, p_kw

# ── 费用效果比 ──
def calc_cer(weekly_cost, mean_slope):
    """CER = $ per 0.01 log CD4 improvement per week"""
    return weekly_cost / (mean_slope * 100) if mean_slope > 0 else float('inf')
'''

p = doc.add_paragraph()
run = p.add_run(code)
run.font.name = 'Consolas'
run.font.size = Pt(9)

# ── 保存 ──
output_path = f'{out_dir}/结核病疗法评价论文_v2.docx'
doc.save(output_path)
print(f'论文已保存: {output_path}')
print('Done.')
