# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(15, 11))
ax.set_xlim(0, 15)
ax.set_ylim(0, 11)
ax.axis('off')

# 颜色
color_input = "#E3F2FD"
color_module = "#FFF9C4"
color_det = "#BBDEFB"
color_loc = "#C8E6C9"
color_output = "#FFCCBC"
color_arrow = "#333333"

def draw_box(ax, x, y, w, h, text, facecolor, edgecolor='black', fontsize=11, weight='bold', text_color='black'):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.15",
                          facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, weight=weight, color=text_color)

def draw_arrow(ax, start_x, start_y, end_x, end_y, color='black', linewidth=2, style='->'):
    arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                            arrowstyle=style, mutation_scale=25,
                            color=color, linewidth=linewidth)
    ax.add_patch(arrow)

# 加图例说明（放在左上角）
ax.text(0.2, 10.6, "📌 数据流方向：从上到下，主线居中，左右分支汇入", fontsize=12, weight='bold', color='darkblue')

# ========== 1. 输入层 ==========
draw_box(ax, 7.5, 10.0, 5.5, 0.9, "输入图片 + 用户指令\n(Input Image + Instruction)", color_input, fontsize=12)

# ========== 2. 并行分支：DTG + 视觉编码器 ==========
draw_box(ax, 3.0, 8.5, 3.0, 0.8, "域标签生成器\n(DTG)", color_module)
draw_box(ax, 12.0, 8.5, 3.0, 0.8, "视觉编码器\n(CLIP ViT + Projector)", color_module)

draw_box(ax, 3.0, 7.2, 3.0, 0.7, "域标签 T_tag\n(PS/DeepFake/AIGC)", color_module, fontsize=10)
draw_box(ax, 12.0, 7.2, 3.0, 0.7, "图像令牌 T_img\n([IMG1], [IMG2], ...)", color_module, fontsize=10)

# ========== 3. DTE-FDM 区域（蓝色） ==========
det_box = FancyBboxPatch((1.0, 4.2), 13.0, 2.6,
                         boxstyle="round,pad=0.2",
                         facecolor=color_det, edgecolor='blue',
                         linewidth=2, linestyle='--', alpha=0.4)
ax.add_patch(det_box)
ax.text(7.5, 6.6, "DTE-FDM：检测与解释模块", ha='center', va='center', fontsize=14, weight='bold', color='blue')

draw_box(ax, 7.5, 5.6, 9.0, 1.3, "大语言模型 (LLM)\n冻结参数 + LoRA 微调", color_module, fontsize=12)

draw_box(ax, 7.5, 4.4, 6.5, 0.8, "文本描述 O_det\n(检测结果 + 位置 + 判断依据)", color_output, fontsize=11)

# ========== 4. MFLM 区域（绿色）—— 重点修复连续连接 ==========
loc_box = FancyBboxPatch((1.0, 0.8), 13.0, 3.1,
                         boxstyle="round,pad=0.2",
                         facecolor=color_loc, edgecolor='green',
                         linewidth=2, linestyle='--', alpha=0.4)
ax.add_patch(loc_box)
ax.text(7.5, 3.8, "MFLM：多模态伪造定位模块", ha='center', va='center', fontsize=14, weight='bold', color='green')

# 子模块定位（统一 y 轴布局）
y_tcm = 2.9
y_sam_enc = 1.9
y_hseg = 1.9
y_sam_dec = 1.9
y_output = 0.3

# ① TCM（文本→语义信号）
draw_box(ax, 7.5, y_tcm, 4.5, 0.8, "篡改压缩模块 TCM\n(提取 <SEG> 特征)", color_module, fontsize=11)

# ② 全连接层 → h_seg 口令（放在 TCM 右侧）
draw_box(ax, 11.8, y_tcm, 2.5, 0.7, "全连接层\n(h_seg 口令)", color_module, fontsize=10)

# ③ SAM 编码器（左下，处理原图）
draw_box(ax, 3.2, y_sam_enc, 3.2, 0.8, "SAM 编码器\n(图像特征提取)", color_module, fontsize=11)

# ④ SAM 解码器（中心，融合两种信号）
draw_box(ax, 7.5, y_sam_dec, 4.2, 0.8, "SAM 解码器\n(生成像素掩码)", color_module, fontsize=11)

# ========== 5. 最终输出 ==========
draw_box(ax, 7.5, y_output, 6.0, 0.9, "最终输出：\n真假判定 + 文字解释 + 篡改掩码", color_output, fontsize=12, weight='bold')

# =================================================================
# 连接线（按数据流顺序，确保连续不断）
# =================================================================

# ① 输入 → 两条分支
draw_arrow(ax, 7.5, 9.55, 3.0, 8.9)
draw_arrow(ax, 7.5, 9.55, 12.0, 8.9)

# ② DTG → T_tag , 编码器 → T_img
draw_arrow(ax, 3.0, 8.1, 3.0, 7.55)
draw_arrow(ax, 12.0, 8.1, 12.0, 7.55)

# ③ T_tag 和 T_img 汇入 LLM（两条线汇合到中间）
draw_arrow(ax, 3.0, 6.85, 3.0, 6.2)
draw_arrow(ax, 3.0, 6.2, 7.5, 6.2)   # 横线到中心
draw_arrow(ax, 12.0, 6.85, 12.0, 6.2)
draw_arrow(ax, 12.0, 6.2, 7.5, 6.2)

# ④ LLM → O_det
draw_arrow(ax, 7.5, 4.95, 7.5, 4.8)

# ------------------- MFLM 内部连续连接（核心修复） -------------------
# ⑤ O_det → TCM（文本描述流入 TCM）
draw_arrow(ax, 7.5, 4.0, 7.5, 3.3)

# ⑥ TCM → 全连接层 → h_seg（向右流动）
draw_arrow(ax, 9.75, 2.9, 11.8, 2.9)

# ⑦ h_seg 折返向下流入 SAM 解码器（作为条件提示）
draw_arrow(ax, 11.8, 2.55, 11.8, 2.3)
draw_arrow(ax, 11.8, 2.3, 9.6, 2.3)   # 横线向左到解码器上方

# ⑧ 输入图片 → SAM 编码器（左上分支独立下去）
draw_arrow(ax, 7.5, 9.55, 5.0, 9.55)
draw_arrow(ax, 5.0, 9.55, 3.2, 2.7)   # 斜线连到 SAM 编码器

# ⑨ SAM 编码器 → SAM 解码器（图像特征向右汇入）
draw_arrow(ax, 4.8, 1.9, 5.4, 1.9)    # 短箭头连接

# ⑩ 来自 h_seg 的线 与 来自编码器的线 在解码器上方汇合（标注汇聚点）
# 在解码器正上方加一个小圈表示“融合”
circle = plt.Circle((7.5, 2.3), 0.15, color='red', fill=True)
ax.add_patch(circle)
ax.text(7.5, 2.1, "特征融合", ha='center', va='center', fontsize=9, weight='bold', color='red')

# 11 SAM 解码器 → 最终输出
draw_arrow(ax, 7.5, 1.5, 7.5, 0.75)

# ========== 额外标注：MFLM 内部三条数据流说明 ==========
ax.text(0.8, 2.8, "文本语义流", fontsize=10, weight='bold', color='darkgreen')
ax.text(0.8, 1.8, "图像视觉流", fontsize=10, weight='bold', color='darkgreen')
ax.text(0.8, 1.0, "融合解码流", fontsize=10, weight='bold', color='darkred')

# 保存
plt.tight_layout()
plt.savefig("FakeShield_Architecture_Fixed.png", dpi=300, bbox_inches='tight')
print("✅ 修复版架构图已生成：FakeShield_Architecture_Fixed.png")