# -*- coding: utf-8 -*-
"""
FakeShield 论文解读 PPT 自动生成脚本
运行前请先安装：pip install python-pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# 创建演示文稿（默认16:9宽屏）
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# 辅助函数：在任意位置添加文本框
def add_text_box(slide, left, top, width, height, text, font_size=18, bold=False, 
                 color=RGBColor(0, 0, 0), align=PP_ALIGN.LEFT, is_title=False):
    shape = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = align
    return shape

# 辅助函数：添加带项目符号的列表
def add_bullet_list(slide, left, top, width, height, items, font_size=18):
    shape = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, item in enumerate(items):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = item
        p.level = 0
        p.font.size = Pt(font_size)
        p.bullet = True

# 辅助函数：添加表格（修正列宽为EMU整数）
def add_table(slide, left, top, width, height, data, headers=None, font_size=14):
    rows = len(data)
    cols = len(data[0])
    table = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(height)).table
    # 计算每列宽度（英寸），再转为EMU整数
    col_width_inches = width / cols
    for i in range(cols):
        table.columns[i].width = Inches(col_width_inches)   # 自动转为整数EMU

    for r, row in enumerate(data):
        for c, cell_text in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(cell_text)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(font_size)
                if r == 0 and headers:  # 表头加粗
                    paragraph.font.bold = True
    return table

# -------- 1. 封面 --------
slide = prs.slides.add_slide(prs.slide_layouts[6])  # 空白布局
add_text_box(slide, 1.5, 1.5, 10, 1.2, "FakeShield: Explainable Image Forgery Detection", 
             font_size=36, bold=True, align=PP_ALIGN.CENTER)
add_text_box(slide, 1.5, 2.7, 10, 0.8, "and Localization Via Multi-modal Large Language Models", 
             font_size=28, bold=True, align=PP_ALIGN.CENTER)
add_text_box(slide, 1.5, 4.2, 10, 0.6, "（基于多模态大语言模型的可解释图像伪造检测与定位）", 
             font_size=24, align=PP_ALIGN.CENTER)
add_text_box(slide, 1.5, 5.5, 10, 0.6, "作者：Zhipei Xu, Xuanyu Zhang, Runyi Li, Zecheng Tang, Qing Huang, Jian Zhang", 
             font_size=16, align=PP_ALIGN.CENTER)
add_text_box(slide, 1.5, 6.2, 10, 0.5, "来源：ICLR 2025", 
             font_size=18, bold=True, align=PP_ALIGN.CENTER)

# -------- 2. 目录 --------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text_box(slide, 0.5, 0.5, 12, 1, "目 录", font_size=44, bold=True, align=PP_ALIGN.CENTER)
items = [
    "1. 研究背景与动机",
    "2. FakeShield 框架",
    "3. 实验结果与分析",
    "4. 结论与未来工作"
]
add_bullet_list(slide, 3, 2.5, 8, 3, items, font_size=32)

# -------- 3. 研究背景与动机 --------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text_box(slide, 0.5, 0.3, 12, 0.8, "研究背景与动机", font_size=36, bold=True)
items = [
    "🔴 虚假图像泛滥，危害严重",
    "   • AIGC和DeepFake技术让图像篡改毫无痕迹",
    "   • 社交媒体谣言、法律取证、虚假新闻急需可靠的鉴伪手段",
    "",
    "🔴 现有图像鉴伪（IFDL）方法存在致命缺陷",
    "   • 黑盒问题：只输出'真/假'概率和热力图，无法解释判断依据",
    "   • 泛化能力差：针对PS修图、DeepFake换脸、AIGC编辑难以通吃",
    "",
    "🔴 引入多模态大模型（M-LLM）的挑战",
    "   • 通用M-LLM缺乏细粒度鉴伪能力（看不出边缘伪影、光照不一致）",
    "   • 现有M-LLM在鉴伪任务上无法精准定位篡改区域"
]
add_bullet_list(slide, 0.5, 1.2, 12, 6, items, font_size=18)

# -------- 4. 核心贡献 --------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text_box(slide, 0.5, 0.3, 12, 0.8, "FakeShield 核心贡献", font_size=36, bold=True)
items = [
    "✅ 首次提出'可解释图像鉴伪'（Explainable IFDL）新任务",
    "   • 不仅判断真假，还能用自然语言写出'鉴定报告'",
    "",
    "✅ 构建MMTD-Set多模态篡改描述数据集",
    "   • 利用GPT-4o对三类篡改（PS/DeepFake/AIGC）自动生成'图像-掩码-描述'三元组",
    "",
    "✅ 设计解耦双模块框架",
    "   • DTE-FDM（检测+解释模块）：判断真假并生成文本理由",
    "   • MFLM（定位分割模块）：根据文本描述精准生成篡改掩码",
    "",
    "✅ 性能全面领先：在PS、DeepFake、AIGC三个领域检测准确率均达SOTA"
]
add_bullet_list(slide, 0.5, 1.2, 12, 6, items, font_size=18)

# -------- 5. 整体框架 --------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text_box(slide, 0.5, 0.3, 12, 0.8, "FakeShield 整体框架", font_size=36, bold=True)
items = [
    "输入：一张可疑图像 + 文本指令（如'请找出篡改区域'）",
    "",
    "三步走流程：",
    "   1. 域标签生成（DTG）：分类器判断图像属于'PS / DeepFake / AIGC'",
    "   2. DTE-FDM（检测与解释）：LLaVA-13B（LoRA）输出检测结论+判断依据",
    "   3. MFLM（定位分割）：提取<SEG>特征，引导SAM生成像素级篡改掩码",
    "",
    "输出：真假判定 + 文字解释 + 篡改区域掩码（红框/白块）"
]
add_bullet_list(slide, 0.5, 1.2, 12, 6, items, font_size=18)

# -------- 6. MMTD-Set数据集 --------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text_box(slide, 0.5, 0.3, 12, 0.8, "MMTD-Set：多模态篡改描述数据集", font_size=32, bold=True)
add_text_box(slide, 0.5, 1.1, 12, 0.5, "数据来源（三类篡改）", font_size=20, bold=True)
data = [
    ["篡改类型", "来源数据集", "典型特征"],
    ["PhotoShop", "CASIAv2, Fantastic Reality", "边缘伪影、光照不一致"],
    ["DeepFake", "FFHQ, FaceApp", "局部模糊、对称性异常"],
    ["AIGC-Editing", "COCO + SD-Inpainting", "纹理模糊、文字混乱"]
]
add_table(slide, 0.5, 1.7, 12, 2.5, data, headers=True, font_size=14)
items = [
    "GPT-4o辅助标注（核心创新）：",
    "   • 输入：篡改图 + 篡改掩码 + 分类型提示词",
    "   • 输出三段式描述：位置（绝对+相对）+ 内容 + 判断依据"
]
add_bullet_list(slide, 0.5, 4.8, 12, 2, items, font_size=18)

# -------- 7. DTE-FDM 模块 --------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text_box(slide, 0.5, 0.3, 12, 0.8, "DTE-FDM：域标签引导的可解释伪造检测", font_size=32, bold=True)
items = [
    "为什么要加'域标签'？",
    "   • PS修图看边缘伪影，DeepFake看面部对称性，AIGC看纹理模糊——特征完全不同",
    "   • 域标签告诉LLM该重点检查什么，缓解数据域冲突",
    "",
    "工作流程：",
    "   1. 原始图像 → DTG分类器 → 输出模板：'This is a suspected {PS/DF/AIGC}-tampered picture.'",
    "   2. CLIP ViT编码 + 域标签 + 用户指令 → LLaVA-13B（LoRA微调）",
    "   3. 自回归生成 O_det（检测结果 + 位置描述 + 判断依据）",
    "",
    "训练方式：冻结LLM主干，仅用LoRA（rank=128）微调"
]
add_bullet_list(slide, 0.5, 1.2, 12, 6, items, font_size=17)

# -------- 8. MFLM 模块 --------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text_box(slide, 0.5, 0.3, 12, 0.8, "MFLM：多模态伪造精准定位模块", font_size=32, bold=True)
items = [
    "为什么单独做定位模块？",
    "   • 检测/解释靠语言理解，定位靠视觉先验——两者联合训练会互相干扰",
    "   • 解耦设计：先让LLM写'小作文'，再让SAM看'小作文'画图",
    "",
    "工作流程：",
    "   1. 文本描述 O_det + 图像令牌 → Tamper Comprehension Module (TCM)",
    "   2. TCM提取特殊令牌 <SEG> 的最后一层嵌入向量 h_<SEG>",
    "   3. 原始图像 → SAM编码器 → 图像中间特征 E_mid",
    "   4. h_<SEG> 作为条件提示 → SAM解码器 → 输出篡改掩码 M_loc",
    "",
    "优势：继承SAM强大的边界分割能力，掩码干净、完整、边界清晰"
]
add_bullet_list(slide, 0.5, 1.2, 12, 6, items, font_size=17)

# -------- 9. 实验一：检测性能 --------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text_box(slide, 0.5, 0.3, 12, 0.8, "检测性能对比（ACC / F1）", font_size=32, bold=True)
data1 = [
    ["方法", "CASIA1+", "IMD2020", "DeepFake", "AIGC-Editing"],
    ["SPAN", "0.60/0.44", "0.70/0.81", "0.24/0.39", "0.35/0.52"],
    ["ManTraNet", "0.52/0.68", "0.75/0.85", "0.95/0.97", "0.50/0.67"],
    ["MVSS-Net", "0.62/0.76", "0.75/0.85", "0.65/0.79", "0.44/0.24"],
    ["FakeShield", "0.95/0.95", "0.83/0.90", "0.98/0.99", "0.93/0.93"]
]
add_table(slide, 0.5, 1.2, 12, 2.5, data1, headers=True, font_size=14)
add_text_box(slide, 0.5, 4.0, 12, 0.5, "DeepFake专项对比", font_size=20, bold=True)
data2 = [
    ["方法", "ACC", "F1"],
    ["CADDM", "0.5227", "0.5982"],
    ["HiFi-DeepFake", "0.5177", "0.6403"],
    ["FakeShield", "0.9835", "0.9915"]
]
add_table(slide, 0.5, 4.7, 8, 1.8, data2, headers=True, font_size=14)

# -------- 10. 实验二：定位性能 --------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text_box(slide, 0.5, 0.3, 12, 0.8, "定位性能对比（IoU / F1）", font_size=32, bold=True)
data = [
    ["方法", "CASIA1+", "IMD2020", "Columbia", "AIGC-Editing"],
    ["OSN", "0.47/0.51", "0.38/0.47", "0.58/0.69", "0.07/0.09"],
    ["PSCC-Net", "0.36/0.46", "0.22/0.32", "0.64/0.74", "0.10/0.15"],
    ["MVSS-Net", "0.40/0.48", "0.23/0.31", "0.48/0.61", "0.18/0.24"],
    ["FakeShield", "0.54/0.60", "0.50/0.57", "0.67/0.75", "0.18/0.24"]
]
add_table(slide, 0.5, 1.2, 12, 2.5, data, headers=True, font_size=14)
items = [
    "定性可视化结论：",
    "   • FakeShield生成的掩码边界清晰、区域完整",
    "   • 对比方法（如PSCC-Net）掩码分散、模糊、过度预测"
]
add_bullet_list(slide, 0.5, 4.2, 12, 2.5, items, font_size=20)

# -------- 11. 实验三：可解释性 --------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text_box(slide, 0.5, 0.3, 12, 0.8, "可解释性对比（CSS余弦语义相似度）", font_size=28, bold=True)
data = [
    ["方法", "CASIA1+", "IMD2020", "DSO", "DeepFake", "AIGC"],
    ["GPT-4o", "0.5183", "0.5326", "0.5804", "0.5643", "0.6289"],
    ["LLaVA-v1.6", "0.6457", "0.5193", "0.5034", "0.6273", "0.6352"],
    ["InternVL2", "0.6760", "0.5750", "0.6484", "0.6570", "0.6751"],
    ["FakeShield", "0.8758", "0.7537", "0.8873", "0.8446", "0.8860"]
]
add_table(slide, 0.5, 1.2, 12, 2.5, data, headers=True, font_size=13)
items = [
    "关键发现：",
    "   • 通用M-LLM能发现明显物理违背，但看不出光照不一致、透视错误、边缘伪影",
    "   • FakeShield通过领域定制微调 + 域标签引导，细粒度分析能力大幅提升"
]
add_bullet_list(slide, 0.5, 4.2, 12, 2.5, items, font_size=18)

# -------- 12. 消融实验 --------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text_box(slide, 0.5, 0.3, 12, 0.8, "消融实验", font_size=36, bold=True)
add_text_box(slide, 0.5, 1.2, 12, 0.5, "① 域标签（DTG）有多重要？", font_size=20, bold=True)
data = [
    ["配置", "CASIA1+ (ACC/F1)", "DeepFake (ACC/F1)", "AIGC (ACC/F1)"],
    ["w/o DTG", "0.92 / 0.92", "0.89 / 0.90", "0.72 / 0.78"],
    ["w/ DTG", "0.95 / 0.95", "0.98 / 0.99", "0.93 / 0.93"]
]
add_table(slide, 0.5, 1.8, 12, 1.8, data, headers=True, font_size=14)
items = [
    "→ 去掉域标签，DeepFake检测F1直接掉0.09，证明域标签对缓解数据冲突至关重要",
    "",
    "② LLM是否必要？（去掉LLM，将检测+定位塞给一个模块）",
    "   • 定位性能（CASIA1+）全程低于原框架，且提前收敛",
    "   • 证明解耦设计（先语言解释，后视觉分割）优于端到端联合训练"
]
add_bullet_list(slide, 0.5, 4.0, 12, 3, items, font_size=18)

# -------- 13. 鲁棒性分析 --------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text_box(slide, 0.5, 0.3, 12, 0.8, "鲁棒性分析（退化图像下的表现）", font_size=32, bold=True)
data = [
    ["退化类型", "CSS（解释质量）", "IoU", "F1"],
    ["JPEG 70", "0.8355", "0.5022", "0.5645"],
    ["JPEG 80", "0.8511", "0.5026", "0.5647"],
    ["Gaussian 5", "0.8283", "0.4861", "0.5494"],
    ["Gaussian 10", "0.8293", "0.4693", "0.5297"],
    ["原始（无退化）", "0.8758", "0.5432", "0.6032"]
]
add_table(slide, 0.5, 1.2, 12, 2.8, data, headers=True, font_size=14)
items = [
    "结论：",
    "   • 对JPEG压缩和Gaussian噪声具有较强鲁棒性（性能下降很小）",
    "   • M-LLM依赖高层语义，低级视觉退化影响有限",
    "   • 非常适合社交媒体传播场景（图像经过多次压缩）"
]
add_bullet_list(slide, 0.5, 4.5, 12, 2.5, items, font_size=18)

# -------- 14. 结论与未来工作 --------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text_box(slide, 0.5, 0.3, 12, 0.8, "结论与未来工作", font_size=36, bold=True)
items = [
    "✅ 结论",
    "   • 首次将M-LLM引入可解释图像鉴伪（Explainable IFDL）任务",
    "   • FakeShield兼具检测、解释、定位三重能力，打破传统黑盒困境",
    "   • 在Photoshop、DeepFake、AIGC三类篡改上全面达到SOTA",
    "   • 为数字取证、新闻核查、法庭证据提供可信、可交互的AI辅助工具",
    "",
    "🔭 未来工作",
    "   • 引入Chain-of-Thought（CoT）机制，增强复杂DeepFake的推理能力",
    "   • 扩充训练数据集，涵盖更多篡改类型（身份交换、全身生成）",
    "   • 优化模块，提升对细粒度面部篡改的检测精度",
    "   • 探索主动水印方案与FakeShield的协同应用"
]
add_bullet_list(slide, 0.5, 1.2, 12, 6, items, font_size=18)

# -------- 15. 结尾/感谢 --------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text_box(slide, 3, 2.5, 8, 1, "感谢聆听！", font_size=48, bold=True, align=PP_ALIGN.CENTER)
add_text_box(slide, 2, 4.0, 9, 0.8, "项目主页 & 开源代码", font_size=24, bold=True, align=PP_ALIGN.CENTER)
add_text_box(slide, 2, 4.8, 9, 0.8, "https://github.com/zhipeixu/FakeShield", font_size=20, align=PP_ALIGN.CENTER, color=RGBColor(0, 0, 255))
add_text_box(slide, 2, 5.8, 9, 0.6, "ICLR 2025  |  Zhipei Xu, Xuanyu Zhang, Runyi Li, et al.", font_size=16, align=PP_ALIGN.CENTER)

# 保存文件
output_path = "FakeShield_Presentation.pptx"
prs.save(output_path)
print(f"✅ PPT 已成功生成：{output_path}")