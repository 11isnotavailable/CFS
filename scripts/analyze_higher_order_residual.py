"""Input-level uncertainty, full controls, and a reproducible negative-result report."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from src.higher_order_residual import SetResidual


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
    out=root/'analysis';out.mkdir(exist_ok=False);lock=json.loads((root/'selection_lock.json').read_text())
    frame=pd.concat([pd.read_csv(root/f'eval_{s}/per_input.csv') for s in ('test','confirmation')],ignore_index=True)
    frame.to_csv(root/'routing_per_input.csv',index=False)
    draw_cache={}
    def stat(values):
        x=np.asarray(values,float);assert np.isfinite(x).all();n=len(x)
        if n not in draw_cache:draw_cache[n]=np.random.default_rng(42).integers(0,n,(2000,n),dtype=np.int32)
        ci=np.quantile(x[draw_cache[n]].mean(1),[.025,.975])
        return dict(n=n,mean=float(x.mean()),sd_inputs=float(x.std(ddof=1)),ci_low=float(ci[0]),ci_high=float(ci[1]))
    summaries=[];paired=[];seeds=[]
    averaged=frame.groupby(['cohort','sample','method','keep'])[['kl','correct']].mean().reset_index()
    for keys,g in frame.groupby(['cohort','method','keep','seed']):
        for metric in ('kl','correct'):
            seeds.append(dict(zip(('cohort','method','keep','seed'),keys),metric=metric,**stat(g.sort_values('sample')[metric])))
    per_seed=pd.DataFrame(seeds)
    for (cohort,method,k),g in averaged.groupby(['cohort','method','keep']):
        for metric in ('kl','correct'):
            selected=per_seed.query('cohort==@cohort and method==@method and keep==@k and metric==@metric')['mean']
            summaries.append(dict(cohort=cohort,method=method,keep=k,metric=metric,seed_mean_sd=float(selected.std(ddof=1)),**stat(g.sort_values('sample')[metric])))
    for (cohort,k),g in averaged.groupby(['cohort','keep']):
        for metric in ('kl','correct'):
            pivot=g.pivot(index='sample',columns='method',values=metric)
            for method in ('plain','cfs','shuffled','plain_raw','cfs_raw','shuffled_raw'):
                refs=('A3','R2','Gating')+ (('plain_raw','shuffled_raw') if method=='cfs_raw' else ())
                for reference in refs:paired.append(dict(cohort=cohort,keep=k,metric=metric,method=method,reference=reference,**stat(pivot[method]-pivot[reference])))
    summary=pd.DataFrame(summaries);paired=pd.DataFrame(paired)
    diag=pd.read_csv(root/'prediction_diagnostics/per_input.csv')
    diag=diag.groupby(['split','sample','kind','mode','keep'])[['mse','mae','base_mse','regret','base_regret']].mean().reset_index()
    ds=[];dp=[]
    for (split,kind,mode,k),g in diag.groupby(['split','kind','mode','keep']):
        for metric in ('mse','mae','base_mse','regret','base_regret'):
            ds.append(dict(split=split,kind=kind,mode=mode,keep=k,metric=metric,**stat(g.sort_values('sample')[metric])))
    for (split,mode,k),g in diag.groupby(['split','mode','keep']):
        for metric in ('mse','regret'):
            pivot=g.pivot(index='sample',columns='kind',values=metric)
            for reference in ('plain','shuffled'):
                dp.append(dict(split=split,mode=mode,keep=k,metric=metric,contrast=f'cfs_minus_{reference}',**stat(pivot.cfs-pivot[reference])))
            for kind in ('plain','cfs','shuffled'):
                h=g[g.kind==kind].sort_values('sample')
                dp.append(dict(split=split,mode=mode,keep=k,metric=metric,contrast=f'{kind}_minus_base',**stat(h[metric]-h['base_'+metric])))
    tables={'summary':summary,'paired':paired,'per_seed':per_seed,'prediction_summary':pd.DataFrame(ds),'prediction_paired':pd.DataFrame(dp)}
    for name,table in tables.items():
        table.to_csv(out/f'{name}.csv',index=False);table.to_json(out/f'{name}.json',orient='records',indent=2)
        table.to_latex(out/f'{name}.tex',index=False,float_format='%.7f')
    np.savez_compressed(out/'bootstrap_draws.npz',**{f'n{n}':v for n,v in draw_cache.items()})
    params=sum(p.numel() for p in SetResidual('cfs').parameters())
    assert all(all(g==0 for g in v['gamma']) for v in lock['variants'].values()), 'Report narrative requires re-audit if selection changes'
    complexity=pd.concat([pd.read_csv(root/f'eval_{s}/timings.csv') for s in ('test','confirmation')])
    complexity.to_csv(out/'timings.csv',index=False)
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors={'plain_raw':'#0072B2','cfs_raw':'#D55E00','shuffled_raw':'#009E73'}
    fig,axes=plt.subplots(1,2,figsize=(7,2.7))
    for ax,cohort in zip(axes,('test','confirmation')):
        for offset,(method,color) in enumerate(colors.items()):
            r=paired.query('cohort==@cohort and metric=="kl" and method==@method and reference=="A3"').sort_values('keep')
            ax.errorbar(r.keep+(offset-1)*.12,r['mean'],yerr=np.stack([r['mean']-r.ci_low,r.ci_high-r['mean']]),fmt='o',ms=3,capsize=2,label=method.replace('_raw',''),color=color)
        ax.axhline(0,color='black',lw=.7);ax.set(title=f'{cohort}: unit-strength diagnostic',xlabel='Retained heads K',ylabel='KL(residual) - KL(A3)',xticks=[3,6,9])
    axes[0].legend(frameon=False,fontsize=6);fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(out/f'figure1_hard_routing.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7,2.7))
    for ax,metric in zip(axes,('mse','regret')):
        for offset,(kind,color) in enumerate((('plain','#0072B2'),('cfs','#D55E00'),('shuffled','#009E73'))):
            contrast=kind+'_minus_base';r=tables['prediction_paired'].query('split=="train" and mode=="uncalibrated" and metric==@metric and contrast==@contrast').sort_values('keep')
            ax.errorbar(r.keep+(offset-1)*.12,r['mean'],yerr=np.stack([r['mean']-r.ci_low,r.ci_high-r['mean']]),fmt='o',ms=3,capsize=2,label=kind,color=color)
        ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Retained heads K',ylabel=f'{metric} difference vs calibrated A3',xticks=[3,6,9],title='Residual-held-out training masks')
    axes[0].legend(frameon=False,fontsize=6);fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(out/f'figure2_prediction_vs_ranking.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    def md(table):return table.to_markdown(index=False,floatfmt='.7f')
    hard=summary.query('metric=="kl"')
    tables_md=[]
    for cohort in ('test','confirmation'):
        sub=hard[hard.cohort==cohort];rows=[]
        for method in ('A3','R2','Gating','plain','cfs','shuffled','plain_raw','cfs_raw','shuffled_raw'):
            row={'method':method}
            for k in (3,6,9):
                v=sub[(sub.method==method)&(sub.keep==k)].iloc[0];row[f'K={k}']=f'{v["mean"]:.7f} [{v.ci_low:.7f}, {v.ci_high:.7f}]'
            rows.append(row)
        tables_md.append(f'### {cohort}\n\n'+md(pd.DataFrame(rows)))
    pair_table=paired.query('method=="cfs_raw" and metric=="kl" and reference in ["A3","plain_raw","shuffled_raw"]')
    pred_table=tables['prediction_paired'].query('split=="train" and mode=="uncalibrated" and metric=="mse"')
    histories=pd.read_csv(root/'training_history.csv');times=histories.groupby(['kind','seed']).seconds.max()
    text=f'''# CFS-conditioned latent interaction residual feasibility

日期：2026-11-15。冻结来源 a7c87f5。**验证选择：{lock['primary']}；是否通过预设提升条件：{lock['eligible']}。**

这轮实现没有取得预期Routing增益。独立确认集上单位强度CFS残差的K3/K6/K9均值为0.116239/0.031450/0.009817，原A3为0.105810/0.027364/0.009669。K3/K6相对A3的配对差区间为正；K9差异未分辨。原R2在该确认集K9为0.006436，仍较强。A3在旧测试三个预算均优于Gating的点估计现象，没有在新确认集K9上保持；不能把旧测试均值结论扩大为普遍优势。

## 1. 实际执行范围与科学边界

仅做方向 A 的离线可行性实验；A3、原 R2、ViT-B Backbone 全部冻结。三种轻量残差、三个固定种子17/23/41，各20epochs，lr3e-4，batch64；不搜索网络、Loss权重、学习率或时长。每种新增 {params:,} 个参数，完整A3+残差为 {825091+params:,} 个，比825,091增加 {params/825091*100:.3f}%。新增模型不是原825K参数匹配主实验。

已有38,016个真实四元组来自旧tune/test，**没有拿来训练**。本轮只复用4,096个训练输入的Joint-KL标签。每输入/预算48个唯一Mask固定拆为36拟合+12诊断。冻结A3此前见过这些标签，因此这里只能称为“对残差预测器留出的Mask”，不是完整管线完全未见的数据。旧测试448张已被多轮查看，仅作探索。新增256张确认图在拟合前锁定，排除旧4,608个输入的路径和SHA256；来自同一Imagenette验证总体，不代表跨域外推。

该实现是**保留集合条件化的潜在成对残差**：训练目标是联合KL预测误差，成对输出未被真实二阶差分唯一识别，不能称作已学得每一对Head的物理/因果交互。没有把任意背景下的四元组相加。结果不改变原始CFS定义，也不证明独立补偿近似是误差的唯一原因。

## 2. 数学形式与公平对照

原结构效用U仍调用项目的裁剪和聚合实现。每预算仅在36个拟合Mask上拟合D0=a-bU，b>=1e-8，保持原排序。预测D_new=D0+gamma_K*sd_K*r(x,R)。CFS版对删除对使用共享小MLP，输入包含同一个保留集合R、冻结输入表示、重要性、Head身份以及八个有向CFS关系特征，取成对输出平均。它等价于带gamma_K/P系数的成对和，不假定可物理相加。分数可为负，记录原值，未为了好看而裁剪KL预测。

Plain是相同参数量的普通集合MLP；Shuffled保留网络和每个输入图的数值分布，但按固定随机置换破坏Head对齐。所有方案使用相同冻结编码、原效用、数据、初始化种子、优化预算、校准规则。Plain仍通过冻结A3编码间接包含CFS训练信息；对照检验显式图结构的增量，而不是全部CFS信息的存在与否。成对网络计算量高于Plain，参数相同不代表延迟相同。

## 3. 验证锁定与补充诊断

各方法按三个seed平均tune标准化MSE选一个共同epoch，再对每预算从0/.25/.5/1中按tune选择真实候选KL最小的gamma，平局取较小gamma。验证仅有48个候选/预算，实际排序枚举220/924/220个；保留这个代理偏差限制。锁定结果如下，测试/确认不参与：

```json
{json.dumps(lock,indent=2)}
```

全部方案选择了gamma=0，均回退原A3。看到此验证结果后、**读取任何本轮测试/确认输出之前**，新增固定gamma=1的统一诊断（`*_raw`），见独立补充协议。它不参与选择，三个方法/预算全部报告，不能根据其测试结果升级为主模型。没有追加训练。

## 4. 真实硬门控KL：Mean [95% CI]

所有方法使用同一ViT-B Checkpoint、预处理、最后一层Head门控和KL(p_dense||p_routed)。原三个基线在旧测试上与历史结果核对一致，dense logits逐元素一致。下表的plain/cfs/shuffled是锁定策略，*_raw是单位强度诊断。

{chr(10).join(tables_md)}

### 关键配对差（CFS单位强度减参考；负数有利于CFS）

{md(pair_table[['cohort','keep','reference','mean','ci_low','ci_high']])}

## 5. 留出Mask的预测与排序

以下为残差留出Mask的MSE差。完整MAE、选择regret、基线误差和所有配对区间保存在prediction_summary/paired。降低绝对KL预测误差可能主要改善输入难度/整体尺度估计，**不能直接推出同一输入内子集排序改善**。关键增量应比较CFS与Plain/Shuffled，而非只比较校准后的基础模型。

{md(pred_table[['keep','contrast','mean','ci_low','ci_high']])}

K9的关键反例：CFS残差相对校准A3的留出Mask MSE约下降0.001451，但选中Mask的regret约增加0.000612。同时CFS的MSE比普通MLP高约0.000258、比Shuffled低约0.000059。因而虽然真实图对齐相对打乱图对预测有一定增量，却没有形成超越普通残差、再改善原Router的完整证据链。确认集K9上CFS与Plain/Shuffled的配对KL差区间均跨零，不据此宣布二者等效。

## 6. 不确定性、成本与结论限制

独立输入为重采样单位，2,000次Bootstrap，seed42；先在每输入内平均三个训练seed，再重采样。旧测试和确认队列分开分析。per_seed表给每个种子均值/区间，summary给输入SD与seed均值SD。区间条件于这三个种子，未作多重比较校正。没有把Mask、Head Pair或种子当作额外独立输入。

九次训练记录总耗时 {times.sum():.2f} 秒（不含特征缓存、数据加载和模型评估）；推理各阶段实测耗时在timings.csv。离线候选打分循环额外计算了冻结encoder，并且*_raw与锁定策略共享计算；这些计时不是优化后的端到端部署延迟。本实验没有推理加速声明。

在预设验证选择下没有得到可以替换A3的残差方法。若单位强度测试/确认的某个点估计更好，也不能反过来选择系数。是否存在CFS结构的增量应读上面的配对对照和独立确认，而不能把非零残差、较低MSE或个别预算均值作为充分证据。当前范围不支持继续叠加方向B，更不能据此把潜在分解称为已验证的高阶因果交互算法。

## 7. Rebuttal wording

We evaluated a bounded, frozen-router residual extension using only existing training joint-deletion labels. Parameter-matched plain, CFS-conditioned pair-decomposed, and shuffled-CFS residuals were trained with identical budgets. The pair terms are latent joint-loss corrections, not identified causal interaction effects. Validation selected zero residual strength for all three methods and budgets, so the primary router remains A3. We report fixed unit-strength diagnostics on both the previously inspected test set and 256 additional confirmation images, without reselecting models. This study distinguishes improvement in absolute joint-KL prediction from improvement in subset selection; it does not establish a deployable higher-order routing advantage or inference acceleration.

## 8. 文件与复现

新增src/higher_order_residual.py、prepare_residual_confirmation.py、train_higher_order_residual.py、evaluate_higher_order_residual.py、analyze_higher_order_residual.py、verify_higher_order_residual.py和对应测试。旧Attention/MLP/Scaling实现不变。主协议与pre-test amendment同时保留。完整权重、逐Mask预测、逐输入KL、确认图片哈希、验证锁和Bootstrap抽样均归档。完整本地测试套件59项通过。

在新实验目录放置scripts/src，source链接到上轮cfs_interaction，baseline链接到learned_gating；confirmation复制本轮锁定的图片与confirmation.npz，勿带features.npz。使用原实验Python环境和base.pth：

```bash
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4
python scripts/train_higher_order_residual.py --root . --stage all
python scripts/evaluate_higher_order_residual.py --root . --stage all
python scripts/analyze_higher_order_residual.py --root .
python scripts/verify_higher_order_residual.py --root .
```

输出目录必须是新的；原始数据和结果不覆盖。分模型拟合可用--stage train --kind cfs --seed 17；所有九个拟合完成后运行--stage lock和--stage diagnostic。图1为实际重排序的配对差，图2分开呈现预测MSE与子集选择regret。
'''
    (root/'REPORT.md').write_text(text,encoding='utf-8')
    print(hard.pivot(index=['cohort','method'],columns='keep',values='mean').to_string())


if __name__=='__main__':main()
