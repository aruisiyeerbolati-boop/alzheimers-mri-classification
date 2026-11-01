"""Build traceable Chinese reports and figures from the third experiment's real CSVs."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from sklearn.metrics import roc_curve

from prepare_dataset import sha256, write_json

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
FIGURES = RESULTS / 'figures'
BRANCHES = ('raw','processed')
MODELS = ('cnn','pixel_svm','embedding_svm')
NAMES = {'cnn':'CNN','pixel_svm':'PCA-SVM','embedding_svm':'CNN-SVM'}
BRANCH_NAMES = {'raw':'基础输入','processed':'CLAHE'}


def json_file(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_file(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def alignment_rows():
    return [
        ['数据','6400张旧数据','5000张OASIS新图','同第二次5000张','更换数据'],
        ['划分','图片级分层，患者ID未知','受试者隔离3476/637/887','保持第二次同一划分','用户明确保留的差异'],
        ['输入','灰度128×128','基础输入＋CLAHE','基础输入主对照；CLAHE附加对照','CLAHE不属于第一次原始方案'],
        ['PCA维数','128','64','128','保存模型已核实'],
        ['PCA whiten','True','False','True','保存模型已核实'],
        ['PCA求解器','auto','randomized','auto','保存模型已核实'],
        ['像素SVM核 / C / gamma','RBF / 10 / 0.01','RBF / 1 / scale','RBF / 10 / 0.01','保存模型已核实'],
        ['Hybrid核 / C / gamma','linear / 0.1 / scale','RBF / 1 / scale','linear / 0.1 / scale','保存模型已核实；linear的gamma不参与计算'],
        ['SVM类别权重','balanced','balanced','balanced','保存模型已核实'],
        ['SVM参数选择','历史选择后保存最佳参数','固定第二次参数','固定第一次选定参数','不重新搜索新参数'],
        ['CNN结构 / 特征','SmallCNN / 128维','同一SmallCNN / 128维','同一SmallCNN / 128维','三个文件字节一致'],
        ['CNN epochs / lr','PPT：50 / 0.0001','12 / 0.001','50 / 0.0001','PPT已核实'],
        ['CNN阈值','保存值约0.865','0.5','保存值约0.865','固定旧数值，不在测试集调阈值'],
        ['batch','历史未记录；当前旧代码64','32','64','按可见旧代码补齐'],
        ['weight_decay','历史未记录；当前旧代码0.0001','0.01','0.0001','按可见旧代码补齐'],
        ['CNN loss','历史未记录；当前旧代码类别加权CE','无类别权重CE','训练类权重CE','按可见旧代码补齐'],
        ['CNN选模','历史准则未记录；当前旧代码验证accuracy','验证loss最低','验证argmax accuracy最高','按可见旧代码补齐，同分保留最早轮'],
        ['学习率调度 / 梯度裁剪','历史未记录；可见旧代码有','主循环无','ReduceLROnPlateau / max_norm=1','按可见旧代码补齐'],
        ['随机种子','42','42','42','保持一致'],
    ]


def figures(metrics, second, epochs, predictions):
    FIGURES.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'figure.dpi':120})
    fig,axes = plt.subplots(2,2,figsize=(13,8))
    for branch,color in [('raw','#2563eb'),('processed','#dc2626')]:
        rows = epochs[branch]
        x = [int(r['epoch']) for r in rows]
        for axis,key in zip(axes.ravel(),('train_loss','val_loss','train_accuracy','val_accuracy')):
            axis.plot(x,[float(r[key]) for r in rows],color=color,label=BRANCH_NAMES[branch])
            axis.set_title(key); axis.set_xlabel('epoch'); axis.grid(alpha=.2)
        selected = max(rows,key=lambda r:float(r['val_accuracy']))
        axes[1,1].scatter(int(selected['epoch']),float(selected['val_accuracy']),color=color,s=45)
    for axis in axes.ravel(): axis.legend()
    fig.suptitle('第三次：真实50轮训练日志（无平滑）')
    fig.tight_layout(); fig.savefig(FIGURES/'training_curves.png',dpi=180); plt.close(fig)

    fig,axes = plt.subplots(2,2,figsize=(13,8))
    for row,branch in enumerate(BRANCHES):
        for col,key in enumerate(('accuracy','auc')):
            axis = axes[row,col]; x=np.arange(3)
            axis.bar(x-.18,[float(second[branch,m][key]) for m in MODELS],.36,label='第二次')
            axis.bar(x+.18,[float(metrics[branch,m][key]) for m in MODELS],.36,label='第三次：统一参数')
            baseline=682/887 if key=='accuracy' else .5
            axis.axhline(baseline,color='black',linestyle='--',label='全阳性／常数分数基线')
            axis.set_xticks(x,[NAMES[m] for m in MODELS]); axis.set_ylim(0,1)
            axis.set_title(f'{BRANCH_NAMES[branch]}：{key}'); axis.legend(fontsize=8); axis.grid(axis='y',alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES/'second_third_comparison.png',dpi=180); plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for i,branch in enumerate(BRANCHES):
        for j,name in enumerate(MODELS):
            matrix=np.array(json.loads(metrics[branch,name]['confusion']))
            axis=axes[i,j]; axis.imshow(matrix,cmap='Blues',vmin=0,vmax=682)
            for r in range(2):
                for c in range(2): axis.text(c,r,str(matrix[r,c]),ha='center',va='center',fontsize=14,
                                            color='white' if matrix[r,c]>400 else 'black')
            axis.set_xticks([0,1],['无痴呆','有痴呆']);axis.set_yticks([0,1],['无痴呆','有痴呆'])
            axis.set_xlabel('预测');axis.set_ylabel('真实');axis.set_title(f'{BRANCH_NAMES[branch]} {NAMES[name]}')
    fig.tight_layout();fig.savefig(FIGURES/'confusion_matrices.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    roc_rows=[]
    for axis,branch in zip(axes,BRANCHES):
        for name in MODELS:
            rows=predictions[branch,name]
            labels=[int(r['true_label']) for r in rows];scores=[float(r['score']) for r in rows]
            fpr,tpr,thresholds=roc_curve(labels,scores)
            axis.plot(fpr,tpr,label=f'{NAMES[name]} AUC={float(metrics[branch,name]["auc"]):.4f}')
            roc_rows.extend({'branch':branch,'model':name,'fpr':float(a),'tpr':float(b),
                             'threshold':float(c) if np.isfinite(c) else 'inf'} for a,b,c in zip(fpr,tpr,thresholds))
        axis.plot([0,1],[0,1],'k--');axis.set_xlabel('假阳性率');axis.set_ylabel('真阳性率')
        axis.set_title(BRANCH_NAMES[branch]);axis.legend();axis.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'roc_curves.png',dpi=180);plt.close(fig)
    with (RESULTS/'roc_points.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(roc_rows[0]));writer.writeheader();writer.writerows(roc_rows)

    manifest=csv_file(ROOT/'dataset/manifest.csv')
    fig,axes=plt.subplots(4,2,figsize=(6,11))
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    for i,source_class in enumerate(('Non_Demented','Very_Mild_Demented','Mild_Demented','Moderate_Demented')):
        row=next(r for r in manifest if r['source_class']==source_class)
        with Image.open(ROOT/'dataset/selected'/row['relative_path']) as image:
            gray=np.array(image.convert('RGB').convert('L').resize((128,128),Image.Resampling.BILINEAR))
        for j,arr in enumerate((gray,clahe.apply(gray))):
            axes[i,j].imshow(arr,cmap='gray',vmin=0,vmax=255);axes[i,j].axis('off')
            axes[i,j].set_title(f'{source_class}\n'+('raw' if j==0 else 'CLAHE'),fontsize=9)
    fig.tight_layout();fig.savefig(FIGURES/'mri_pairs.png',dpi=180);plt.close(fig)


def main():
    verification=json_file(RESULTS/'verification.json')
    if verification['status']!='passed': raise ValueError('Third verification must pass before report generation')
    config=json_file(RESULTS/'source_config.json');run=json_file(RESULTS/'run_config.json')
    inventory=json_file(RESULTS/'models/inventory.json');summary=json_file(ROOT/'dataset/summary.json')
    metrics={};second={};epochs={};predictions={}
    for branch in BRANCHES:
        epochs[branch]=csv_file(RESULTS/branch/'epoch_metrics.csv')
        for row in csv_file(RESULTS/branch/'metrics.csv'):metrics[branch,row['model']]=row
        for name in MODELS:predictions[branch,name]=csv_file(RESULTS/branch/f'{name}_predictions.csv')
    for row in csv_file(RESULTS/'second_reference/metrics.csv'):second[row['branch'],row['model']]=row
    figures(metrics,second,epochs,predictions)
    beijing=timezone(timedelta(hours=8))
    start=datetime.fromisoformat(run['started_utc']).astimezone(beijing)
    end=datetime.fromisoformat(run['finished_utc']).astimezone(beijing)
    result_rows=[];comparison_rows=[];error_rows=[]
    for branch in BRANCHES:
        for name in MODELS:
            r=metrics[branch,name];old=second[branch,name]
            result_rows.append([BRANCH_NAMES[branch],NAMES[name],f'{float(r["accuracy"]):.2%}',
                                f'{float(r["balanced_accuracy"]):.2%}',f'{float(r["f1"]):.4f}',f'{float(r["auc"]):.4f}'])
            comparison_rows.append([BRANCH_NAMES[branch],NAMES[name],f'{float(old["accuracy"]):.2%}',
                                    f'{float(r["accuracy"]):.2%}',f'{100*(float(r["accuracy"])-float(old["accuracy"])):+.2f}',
                                    f'{float(old["auc"]):.4f}',f'{float(r["auc"]):.4f}'])
            tn,fp,fn,tp=[int(r[k]) for k in ('tn','fp','fn','tp')]
            error_rows.append([BRANCH_NAMES[branch],NAMES[name],tn,fp,fn,tp,f'{tp/682:.2%}',f'{tn/205:.2%}'])
    alignment=alignment_rows()
    duration=(end-start).total_seconds()
    sections=[]
    def add(title,text=None,headers=None,rows=None,picture=None):
        sections.append({'title':title,'text':text,'headers':headers,'rows':rows,'picture':picture})
    add('实验目的与对齐范围',
        '本次在第二次5000张新数据基础上，恢复第一次PPT和保存模型中可核实的参数。'
        '用户明确选择保留第二次的受试者隔离划分。因此本实验不能宣称与第一次唯一差别只有数据集：'
        '划分方式仍不同；历史CNN未记录的细节按可见旧代码补齐；CLAHE作为沿用第二次的附加对照，'
        '基础输入raw是与第一次原始方案对照的主要分支。')
    add('真实运行记录',f'北京时间 {start:%Y-%m-%d %H:%M:%S} 至 {end:%Y-%m-%d %H:%M:%S}，'
        f'完整训练与测试用时约{duration:.1f}秒。设备为{config["environment"]["gpu"]}。'
        '两分支均从随机初始化训练50轮，未加载第一次或第二次的训练权重；SVM固定第一次选定参数。')
    add('第一次、第二次与第三次设置',headers=['项目','第一次证据','第二次','第三次','对齐状态'],rows=alignment)
    add('数据和划分',f'源数据为Kaggle OASIS衍生二维JPG。5000张图来自342人，四类分别1504、1504、1504、488张。'
        '第三次图片字节与第二次完全一致，抽样及split清单SHA-256与原参考值一致。'
        '训练3476张/239人，验证637张/51人，测试887张/52人；同一受试者不跨集合。'
        '测试205张阴性、682张阳性。中度仅两人，验证集无中度，测试中度244张全来自一人。'
        '结果为同数据集患者隔离的切片级留出评价，不是外部队列或独立海马识别。',picture='mri_pairs.png')
    training_rows=[]
    for branch in BRANCHES:
        m=inventory['branches'][branch]
        training_rows.append([BRANCH_NAMES[branch],50,m['best_epoch'],f'{m["best_val_accuracy"]:.2%}',f'{m["selected_val_loss"]:.6f}'])
    add('训练与选择',
        'AdamW初始学习率0.0001、weight_decay=0.0001、batch64、训练类权重交叉熵、梯度范数裁剪1。'
        'ReduceLROnPlateau依据验证argmax准确率降低学习率；最佳模型也依据该准确率选择，同分保留最早轮。'
        '上述PPT未记录的细节按当前旧代码补齐，未将它们冒充为已证实的历史设置。'
        '最终CNN分类固定采用第一次保存的阈值约0.865，未在测试集调阈值。',
        headers=['输入','训练轮数','最佳轮','选中验证accuracy','该轮验证loss'],rows=training_rows,picture='training_curves.png')
    add('第三次测试结果',headers=['输入','模型','Accuracy','Balanced accuracy','阳性F1','AUC'],rows=result_rows,picture='roc_curves.png')
    add('与第二次的相同测试集比较',
        '第二次数字由其原保存模型在同一887张测试图上重新计算，原文件未修改。'
        '第三次同时改变多项算法与决策设置，表中差值不能解释为PCA、C、gamma或阈值某单项的独立效果。',
        headers=['输入','模型','第二次accuracy','第三次accuracy','差值/百分点','第二次AUC','第三次AUC'],
        rows=comparison_rows,picture='second_third_comparison.png')
    add('混淆矩阵与基线',
        f'全部预测阳性的accuracy为{682/887:.2%}，阳性F1为{1364/1569:.4f}，balanced accuracy为50%，常数分数AUC为0.5。'
        '本次六组accuracy均未超过该多数类基线，但balanced accuracy和AUC均高于0.5，需分别看两类召回和分数排序。'
        'CNN阈值从第二次0.5固定恢复为旧值约0.865，直接影响误判/漏判权衡；不能只用accuracy下降判断训练完全失败。',
        headers=['输入','模型','TN','FP','FN','TP','阳性召回','阴性特异度'],rows=error_rows,picture='confusion_matrices.png')
    clahe=[]
    for name in MODELS:
        raw=metrics['raw',name];processed=metrics['processed',name]
        clahe.append(f'{NAMES[name]} accuracy由{float(raw["accuracy"]):.2%}变为{float(processed["accuracy"]):.2%}'
                     f'（{100*(float(processed["accuracy"])-float(raw["accuracy"])):+.2f}个百分点），'
                     f'AUC由{float(raw["auc"]):.4f}变为{float(processed["auc"]):.4f}')
    add('本次结果的解释与限制','；'.join(clahe)+'。这是当前两套独立训练流程的实际结果，不能预设CLAHE提高所有指标。'
        '统一已知参数没有保证分数统一提高。数据类别和切片贡献不平衡，原数据患者身份未知，'
        '保留的受试者划分差异、未记录的历史CNN配置、单次种子和运行环境差异限制了“只更换数据”的严格解释。'
        '本次不证明临床泛化，也没有多种子显著性检验、患者级聚合、海马分割或三维分析。')
    add('文件核验与运行',
        f'核心核验状态：{verification["status"]}；六组预测指标、两套50轮日志、PCA/SVM全部可见参数、'
        f'5000张图片以及前两次{verification.get("protected_files_verified",0)}个原文件的内容完整性已核对。'
        'reference_profile.json记录第一次的参数与来源；dataset/manifest.csv记录原样沿用的第二次划分；'
        'results/raw与processed保留训练CSV、逐图预测和指标；results/second_reference保留第二次复算证据。'
        '已有结果可运行python -B third_experiment/run_experiment.py --evaluate-only；'
        '重新训练必须指定新的--output-dir，原结果不会覆盖。')

    markdown=['# 第三次实验记录：统一第一次已知参数，保留受试者划分','']
    document=Document()
    style=document.styles['Normal'];style.font.name='Microsoft YaHei';style.font.size=Pt(10)
    style.element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
    document.add_heading('第三次实验记录：统一参数',0)
    for section in sections:
        markdown.extend(['## '+section['title'],''])
        document.add_heading(section['title'],1)
        if section['text']:
            markdown.extend([section['text'],'']);document.add_paragraph(section['text'])
        if section['headers']:
            markdown.extend([table(section['headers'],section['rows']),''])
            doc_table=document.add_table(rows=1,cols=len(section['headers']));doc_table.style='Table Grid'
            for cell,value in zip(doc_table.rows[0].cells,section['headers']):cell.text=str(value)
            for row in section['rows']:
                for cell,value in zip(doc_table.add_row().cells,row):cell.text=str(value)
            for row in doc_table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        for run_part in paragraph.runs:run_part.font.size=Pt(8)
        if section['picture']:
            image=FIGURES/section['picture']
            markdown.extend([f'![{section["title"]}](results/figures/{section["picture"]})',''])
            document.add_picture(str(image),width=Cm(16))
    report_md=ROOT/'第三次实验记录_统一参数.md'
    report_docx=ROOT/'第三次实验记录_统一参数.docx'
    report_md.write_text('\n'.join(markdown),encoding='utf-8')
    document.save(report_docx)
    (ROOT/'参数统一说明.md').write_text('# 参数统一与保留差异\n\n'
        '以第一次PPT和保存模型为基准；受试者划分按用户选择保留。未记录历史细节按可见旧代码补齐。'
        '基础输入为主对照，CLAHE为附加分析；不能宣称所有条件完全一致。\n\n'+
        table(['项目','第一次证据','第二次','第三次','对齐状态'],alignment)+'\n',encoding='utf-8')
    artifacts=[report_md,report_docx,ROOT/'参数统一说明.md',*FIGURES.glob('*.png'),RESULTS/'roc_points.csv']
    write_json(RESULTS/'report_artifacts.json',{'generated_utc':datetime.now(timezone.utc).isoformat(),
        'files':{p.relative_to(ROOT).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size} for p in artifacts}})
    print(f'Reports generated: {report_docx.name}; {report_md.name}',flush=True)


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
