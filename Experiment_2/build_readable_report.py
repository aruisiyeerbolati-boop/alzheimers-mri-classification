from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "第三次实验记录.docx"
SAMPLE_IMAGE = BASE / "diagnostics" / "final_check_image_1.png"
LOSS_IMAGE = BASE / "results" / "figures" / "report_final" / "epoch_losses.png"


def set_font(run, size=11, bold=False):
    run.font.name = "Microsoft YaHei"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold


def add_body(document, text):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.first_line_indent = Cm(0.74)
    paragraph.paragraph_format.line_spacing = 1.5
    paragraph.paragraph_format.space_after = Pt(5)
    set_font(paragraph.add_run(text))


def add_heading(document, text):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(8)
    paragraph.paragraph_format.space_after = Pt(4)
    set_font(paragraph.add_run(text), size=13, bold=True)


def shade(cell, fill="D9EAF7"):
    element = OxmlElement("w:shd")
    element.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(element)


def add_table(document, headers, rows):
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = ""
        set_font(cell.paragraphs[0].add_run(str(value)), size=9, bold=True)
        shade(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for values in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, values):
            cell.text = ""
            set_font(cell.paragraphs[0].add_run(str(value)), size=9)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_picture(document, path, caption, width):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(1)
    paragraph.add_run().add_picture(str(path), width=Cm(width))
    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.paragraph_format.space_after = Pt(5)
    set_font(caption_paragraph.add_run(caption), size=9)


def build():
    document = Document()
    section = document.sections[0]
    section.page_width = Cm(21.59)
    section.page_height = Cm(27.94)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)

    normal = document.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(11)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(14)
    set_font(title.add_run("第三次实验记录——5000张新数据"), size=18, bold=True)

    add_body(document, "本次实验更换了一套公开脑部磁共振数据。原数据集共有86437张图片，本次从中选取5000张，按照原实验的二分类任务重新训练和测试。实验目的有两个：一是观察模型换用新数据后的表现；二是比较图片不增加额外处理和进行对比度增强后的差别。原有code文件夹没有修改，所有新数据、程序和结果都放在new_experiment文件夹中。")

    add_heading(document, "一、数据集说明")
    add_body(document, "新数据来自Kaggle平台公开的OASIS阿尔茨海默病检测数据集。图片是二维脑部磁共振图像，画面中包含海马结构所在区域，但不是已经单独裁剪好的海马区域。原数据按照无痴呆、极轻度痴呆、轻度痴呆和中度痴呆分为四类。")
    add_body(document, "抽取时尽量让四类数量接近：无痴呆、极轻度痴呆和轻度痴呆各1504张，中度痴呆共488张，合计5000张。中度痴呆原始数据只有488张，因此四类无法做到完全相等，也没有通过复制图片凑数。进行二分类时，无痴呆记为0，其余三类合并为有痴呆并记为1，最终无痴呆1504张、有痴呆3496张。")
    add_body(document, "为减少同一个人的相似图片同时出现在训练和测试中的情况，本实验按照受试者编号划分数据。同一受试者的图片只会出现在一个集合中。训练集3476张、验证集637张、测试集887张，对应受试者数量分别为239人、51人和52人。")

    add_table(document, ["用途", "图片数", "受试者数", "无痴呆", "有痴呆"], [
        ["训练", 3476, 239, 1087, 2389],
        ["验证", 637, 51, 212, 425],
        ["测试", 887, 52, 205, 682],
    ])
    add_picture(document, SAMPLE_IMAGE, "图1  新数据中的四类脑部磁共振图片样例", 16.0)

    add_heading(document, "二、实验方法")
    add_body(document, "实验设置两种输入方式。第一组只进行模型运行所必需的操作，包括转为灰度图、统一缩放到128×128像素以及数值归一化，可理解为不增加额外处理。第二组在相同步骤上增加CLAHE对比度增强，用来突出亮暗差别和局部细节。两组使用完全相同的数据划分和训练参数，保证比较条件一致。")
    add_body(document, "模型沿用原实验的三种方案。第一种是CNN卷积神经网络，它可以直接从图片中学习形状、纹理等特征；第二种是支持向量机，先把图片信息整理后再进行分类；第三种是CNN-SVM组合模型，先由CNN提取特征，再交给支持向量机完成判断。CNN共训练12轮，每批输入32张图片，学习率为0.001。验证集用于选择训练过程中表现较好的模型，测试集只用于最后评价。")
    add_body(document, "评价指标包括准确率、F1和AUC。准确率表示全部图片中判断正确的比例；F1综合考虑漏判和误判；AUC用于观察模型区分两类图片的整体能力。三个指标越接近1通常越好，但必须结合数据分布和混淆矩阵一起判断，不能只看一个数字。")
    add_picture(document, LOSS_IMAGE, "图2  两种输入方式的CNN训练过程", 16.2)

    add_heading(document, "三、实验结果")
    add_table(document, ["输入方式", "模型", "准确率", "F1", "AUC"], [
        ["无额外处理", "CNN", "0.768", "0.834", "0.856"],
        ["无额外处理", "支持向量机", "0.450", "0.483", "0.729"],
        ["无额外处理", "CNN-SVM", "0.743", "0.812", "0.806"],
        ["对比度增强", "CNN", "0.556", "0.619", "0.809"],
        ["对比度增强", "支持向量机", "0.436", "0.459", "0.695"],
        ["对比度增强", "CNN-SVM", "0.619", "0.693", "0.695"],
    ])
    add_body(document, "从结果看，无额外处理组的CNN表现最好，准确率为76.8%，F1为0.834，AUC为0.856。CNN-SVM组合模型的准确率为74.3%，低于CNN，但明显高于单独使用支持向量机。增加对比度增强后，三种模型的准确率都没有提高，其中CNN下降到55.6%。这说明对比度增强并不一定适合当前数据，过强地改变亮暗关系可能影响模型学习原始图像特征。")
    add_body(document, "测试集中有痴呆图片682张，无痴呆图片205张，两类数量不平衡。如果把所有图片都判断为有痴呆，准确率也能达到76.89%。本实验CNN准确率为76.8%，与这个简单基线接近，因此不能只根据准确率认为模型已经取得理想效果。CNN的F1和AUC较高，说明模型具有一定区分能力，但对少数类无痴呆图片的识别仍需改进。")

    add_heading(document, "四、结论与不足")
    add_body(document, "本次实验完成了5000张新数据上的二分类训练，并比较了两种图片输入方式。综合结果表明，直接使用基础输入的CNN效果相对最好；本次对比度增强没有带来提升，因此后续不能默认图片处理越多效果越好。支持向量机单独使用时表现较弱，CNN提取的图像特征对分类更有帮助。")
    add_body(document, "本实验仍有明显限制。5000张图片并不等于5000位患者，同一受试者可能有多张切片；中度痴呆图片只来自两位受试者，代表性有限；图片是完整脑部切片，并非单独的海马区域；本次只进行了一次固定划分，也没有证明模型能够用于临床诊断。后续可在保持受试者隔离的前提下调整类别比例，增加不同随机划分的重复实验，并尝试更准确地定位海马区域。")

    add_heading(document, "五、文件说明")
    add_body(document, "数据清单、5000张图片、训练程序、模型文件和实验结果均保存在new_experiment文件夹中。原有code文件夹未做修改。报告中的结果来自已完成的正式训练记录，本次只重新整理了报告表达和排版，没有重新训练模型，也没有更改任何实验数值。")

    document.core_properties.title = "第三次实验记录——5000张新数据"
    document.core_properties.subject = "脑部磁共振二分类实验"
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
