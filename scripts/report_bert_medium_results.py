"""Reviewable presentation of the frozen seven-model analysis; no metric changes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.bert_medium import AXES,SPECS

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;source=root/'analysis';out=root/'interpretation';out.mkdir(exist_ok=True)
    assert (source/'COMPLETE.json').exists()
    table=pd.read_csv(source/'main_table.csv').set_index('model')
    summary=pd.read_csv(source/'summary.csv');paired=pd.read_csv(source/'paired.csv')
    stability=pd.read_csv(source/'stability_1B_2B.csv');trends=pd.read_csv(source/'trends.csv')
    matrices=pd.read_csv(source/'matrix_stability_per_input.csv')
    assert set(table.index)==set(SPECS) and (table.training_tokens==2000000000).all()
    identities=[]
    for model in SPECS:
        meta=json.loads((root/'cfs'/model/'2B/metadata.json').read_text())
        identities.append(meta['input_document_hashes']);assert meta['samples']==256
    assert all(x==identities[0] for x in identities) and len(set(identities[0]))==256
    cost=sum(json.loads(p.read_text())['gpu_hours'] for p in (root/'cfs').glob('*/*/WALL_COST.json'))
    hours=table.reserved_training_attempt_gpu_hours.sum()
    actual_costs=table[['training_tokens','reserved_training_attempt_gpu_hours']].copy()
    actual_costs['end_to_end_training_tokens_per_second']=actual_costs.training_tokens/(actual_costs.reserved_training_attempt_gpu_hours*3600)
    actual_costs['mean_seconds_per_update_including_validation_io']=actual_costs.reserved_training_attempt_gpu_hours*3600/61036
    actual_costs['successful_cfs_allocation_gpu_hours']=[sum(json.loads(p.read_text())['gpu_hours'] for p in (root/'cfs'/model).glob('*/WALL_COST.json')) for model in actual_costs.index]
    actual_costs.to_csv(out/'actual_costs.csv');actual_costs.to_latex(out/'actual_costs.tex',float_format='%.5f')
    mc=[]
    for model in SPECS:
        frame=pd.read_csv(source/model/'subgraph_sampling.csv')
        for metric,part in frame.groupby('metric'):
            mc.append(dict(model=model,metric=metric,aggregate_subgraph_mc_se=np.sqrt(np.square(part.subgraph_mc_se).sum())/len(part),
                mean100_minus500=(part.mean100-part.mean500).mean(),mean250_minus500=(part.mean250-part.mean500).mean()))
    pd.DataFrame(mc).to_csv(out/'sampling_precision.csv',index=False)
    metrics=['raw','matched_2','matched_4','matched_5','cover_full','rank_full','cover_4','rank_4','valid_source_ratio','mean_deletion_kl']
    compact=table[['parameters','validation_ce','validation_accuracy']+metrics].copy()
    compact['validation_accuracy']*=100
    compact.to_csv(out/'main_table.csv');compact.to_latex(out/'main_table.tex',float_format='%.5f')
    formatted=summary.copy()
    formatted['mean_and_95ci']=formatted.apply(lambda r:f"{r['mean']:.5f} [{r.ci_low:.5f}, {r.ci_high:.5f}]",axis=1)
    ci=formatted.pivot(index='model',columns='metric',values='mean_and_95ci')
    ci.to_csv(out/'confidence_intervals.csv');ci.to_latex(out/'confidence_intervals.tex')
    clarified=[]
    for axis,names in AXES.items():
        for metric in metrics+['validation_ce']:
            if table.loc[names,metric].isna().any():continue
            diff=np.diff(table.loc[names,metric])
            label='constant' if np.all(diff==0) else 'increasing' if np.all(diff>=0) else 'decreasing' if np.all(diff<=0) else 'non-monotonic'
            clarified.append(dict(axis=axis,metric=metric,direction=label))
    pd.DataFrame(clarified).to_csv(out/'clarified_trends.csv',index=False)
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    labels={'raw':'Raw oracle','matched_1':'Matched-1','matched_2':'Matched-2','matched_3':'Matched-3','matched_4':'Matched-4','matched_5':'Matched-5',
            'cover_full':'Cover/H','rank_full':'Rank/H','cover_4':'Cover/4','rank_4':'Rank/4','validation_ce':'Validation MLM CE'}
    def panel(ax,names,metric_list,axis):
        x=[SPECS[n][{'width':0,'depth':1,'heads':2}[axis]] for n in names]
        for metric in metric_list:
            if metric=='validation_ce':
                mid=table.loc[names,metric];lo=table.loc[names,metric+'_ci_low'];hi=table.loc[names,metric+'_ci_high']
            else:
                rows=summary[summary.metric==metric].set_index('model').loc[names]
                mid=rows['mean'];lo=rows.ci_low;hi=rows.ci_high
            ax.plot(x,mid,'o-',ms=3,label=labels[metric]);ax.fill_between(x,lo,hi,alpha=.15)
        ax.set_xticks(x);ax.set_xlabel({'width':'Hidden width','depth':'Encoder layers','heads':'Attention heads'}[axis])
        ax.grid(alpha=.15);ax.legend(fontsize=6,frameon=False)
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/f'{name}.pdf',bbox_inches='tight');fig.savefig(out/f'{name}.png',dpi=250,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(7.1,4.6))
    for col,(axis,names) in enumerate(AXES.items()):
        panel(axs[0,col],names,['validation_ce'],axis)
        panel(axs[1,col],names,['raw','matched_2'],axis)
        axs[0,col].set_title(axis.capitalize())
    axs[0,0].set_ylabel('Cross-entropy');axs[1,0].set_ylabel('Oracle recovery');save(fig,'performance_and_oracle')
    fig,axs=plt.subplots(2,3,figsize=(7.1,4.6))
    for col,(axis,names) in enumerate(AXES.items()):
        panel(axs[0,col],names,[f'matched_{n}' for n in ([1,2,3] if axis=='heads' else [2,4,5])],axis)
        panel(axs[1,col],names,['cover_4','rank_4'],axis);axs[0,col].set_title(axis.capitalize())
    axs[0,0].set_ylabel('Matched oracle');axs[1,0].set_ylabel('Normalized graph statistic');save(fig,'candidate_and_subgraph_controls')
    chosen=stability[stability.metric.isin(['raw','matched_2','cover_full','rank_full'])]
    def series(names,metric):return ' → '.join(f'{table.loc[n,metric]:.5f}' for n in names)
    def direction(axis,metric):return trends[(trends.axis==axis)&(trends.metric==metric)].iloc[0].direction
    text='# BERT-Medium 七模型受控 Scaling：完整 2B 结果\n\n'
    text+='七个模型均从随机初始化训练到精确 2,000,000,000 非 padding 输入位置；B2 是三轴共享的唯一基准。'
    text+='主结果固定为 2B，未按 CFS 选择 checkpoint，也未启动 3B。全部 28 个里程碑 checkpoint 保留远端，其路径与 SHA256 见 CHECKPOINT_MANIFEST.json。\n\n'
    display=compact[['parameters','validation_ce','validation_accuracy','raw','matched_2','cover_full','rank_full']]
    text+='## 主结果\n\n'+display.to_markdown(floatfmt='.5f')+'\n\n'
    text+='Accuracy 列为百分数；Mean deletion KL 单位为自然对数。256 个相同独立验证文档用于 CFS，2048 个用于 MLM 验证。'
    text+='全部均值、输入标准差、95% CI、配对差异和子图采样误差见 analysis/ 与 interpretation/；表中 Matched-4/5 对 H=4 不适用，保留缺失值。\n\n'
    text+='## Q1：现有资源上的成本\n\n'
    text+=f'实际训练进程累计占用 {hours:.3f} GPU-hours（含启动、验证和存盘）；成功 CFS 采集累计占用 {cost:.3f} GPU-hours。'
    text+='CFS 的占用时长包含 CPU 计算/传输等待，不等于 GPU 忙碌时间；准备、benchmark 和失败诊断未计入这两项。'
    text+='原 GPT-2 七模型训练记录约 38.00 GPU-hours，但架构、tokenizer、序列长度、目标和硬件型号均不同，不能当作受控加速倍数。\n\n'
    text+='正式训练吞吐量以完整训练进程的墙钟时间计算，包含验证和存盘，详见 actual_costs.csv：\n\n'+actual_costs.to_markdown(floatfmt='.4f')+'\n\n'
    text+='## Q2：训练充分性、非退化性与稳定性\n\n'
    text+=f"2B 有效 source 比例范围为 {table.valid_source_ratio.min():.4%}–{table.valid_source_ratio.max():.4%}，mean deletion KL 为 {table.mean_deletion_kl.min():.6f}–{table.mean_deletion_kl.max():.6f}。"
    text+='这些结果支持删除干预在数值上非退化，不等于已实现推理压缩。\n\n'
    text+=chosen.to_markdown(index=False,floatfmt='.5f')+'\n\n'
    text+='上表只比较共同 64 输入，因此其 2B 均值不应与全 256 输入主表混用。1B→2B 部分 Oracle 变化区间不跨零；'
    text+='不能宣称所有模型的功能组织已完全稳定。CI 跨零也不是等价性证明。完整学习曲线及逐输入矩阵变化已保存。'
    text+='2B 仍是有效的预先指定等预算比较，但“充分稳定/收敛”的措辞应弱化；如需延长，须另行决定七模型共同 3B 协议，本次未启动。\n\n'
    matrix_summary=matrices.groupby('model')[['matrix_mae','matrix_spearman']].mean()
    text+='逐输入矩阵稳定性的均值（共同有效源，仅描述性，不把 pair 当独立样本）：\n\n'+matrix_summary.to_markdown(floatfmt='.5f')+'\n\n'
    text+='## Q3：Width 与 Depth\n\n'
    text+=f"Width B1→B2→B3：Raw {series(AXES['width'],'raw')}，Matched-2 {series(AXES['width'],'matched_2')}。"
    text+='固定候选数后仍应依据配对区间判断，不以 Raw alone 代替候选控制。\n\n'
    text+='本次 Width 的 Matched-2/4/5 均上升；Matched-2 两段配对区间均高于零。Fixed-K=4 的 Cover 不再单调，Rank/4 的 B2→B3 差异区间跨零，因此图指标的连续趋势证据比 Oracle 弱。\n\n'
    text+=f"Depth B4→B2→B5：Raw {series(AXES['depth'],'raw')}，Matched-2 {series(AXES['depth'],'matched_2')}。"
    text+='本次深度结果为非单调，不能给出简单的随深度增减规律。最终层的绝对深度随模型改变，这一限制保留。\n\n'
    text+='## Q4：固定参数容量的 Head 分解\n\n'
    text+=f"B6→B2→B7 均为 41,404,730 参数。Raw {series(AXES['heads'],'raw')}；Matched-2 {series(AXES['heads'],'matched_2')}；MLM CE {series(AXES['heads'],'validation_ce')}。"
    text+=f"Raw 方向为 {direction('heads','raw')}，Matched-2 为 {direction('heads','matched_2')}。"
    text+='固定 K=4 的结果和配对区间同时展示；不能把 head 数带来的候选优势直接解释为同样强的固有替代性提升，也不能把关联解释成 CFS 对性能的因果收益。\n\n'
    text+='本次 Matched-1/2/3 随 head 数增加而下降，Matched-2 两段差异的输入级区间均低于零。'
    text+='完整图 Rank/H 的下降在 Fixed-K=4 后变为 Rank/4 上升；这两个不同问题的统计量不能混为同一趋势。'
    text+='固定 K 的 Rank 两段配对区间均高于零，Cover 的第一段区间仍跨零。8-head 的 MLM CE 优于4/16-head，且对本次固定训练种子的两个配对 CE 区间均不跨零；没有独立训练种子重复，不能外推训练稳健性。\n\n'
    selected=[]
    for axis,names in AXES.items():
        for left,right in zip(names[:-1],names[1:]):
            selected.append(paired[(paired.axis==axis)&(paired.left==left)&(paired.right==right)&paired.metric.isin(['matched_2','cover_4','rank_4','validation_ce'])])
    important=pd.concat(selected)[['axis','left','right','metric','change','ci_low','ci_high']]
    important.to_csv(out/'key_paired_differences.csv',index=False)
    text+='相邻模型的变化（right − left）及 95% 输入配对区间：\n\n'+important.to_markdown(index=False,floatfmt='.5f')+'\n\n'
    text+='## Q5：跨架构结论与限制\n\n'
    text+='原 CFS 干预、输出 KL、alpha grid 和最终论文 Cover/Rank 实现能在这套从头训练的双向 MLM 家族上产生可比较的非退化矩阵。'
    text+='这支持受控分析从原 decoder 家族拓展到 bidirectional encoder；它不保证跨架构方向一致或普适单调规律。'
    text+='本报告不将仓库旧 section5 中的受控 ViT 表误作 GPT-2 对照，也不跨不同输出协议直接比较绝对 CFS。\n\n'
    text+='区间均为 2000 次文档级 bootstrap，只覆盖输入不确定性，不包含独立训练种子变异；多指标区间未做多重比较校正。'
    text+='Fixed-K 的500次子图抽样在每个输入内进行，未作为独立样本；H=K 时只计算一次。sampling_precision.csv 单独给出 Monte Carlo SE 与100/250/500次的敏感性。\n\n'
    text+='clarified_trends.csv 将全相等的指标标为 constant；原自动分析的非严格单调标签不能把恒为100%的有效源比例解读成增长。\n\n'
    text+='## 数值修复与可复现性\n\n'
    text+='首次训练完成后 CFS 曾因 FP16 batch 形状的前缀缓存舍入差异暂停。修复为按实际干预 batch 形状缓存，保留原误差阈值与数学协议。'
    text+='训练权重未重训，失败阶段无正式矩阵 shard；实现修订与旧代码锁均归档。每个 checkpoint 的首输入仍核验缓存/完整前向的一致性。\n\n'
    text+='复现命令与文件用途见 docs/BERT_MEDIUM_CONTROLLED_PROTOCOL.md；分析命令为 `python scripts/analyze_bert_medium.py --root RUN_ROOT`，'
    text+='本报告命令为 `python scripts/report_bert_medium_results.py --root RUN_ROOT`。原始矩阵、逐输入统计、bootstrap、CSV/JSON/LaTeX 与 PDF/PNG 均已保留。\n\n'
    for name in ['performance_and_oracle','candidate_and_subgraph_controls']:text+=f'![{name}](interpretation/{name}.png)\n\n'
    (root/'REPORT.md').write_text(text.rstrip()+'\n',encoding='utf-8')
    reply='We added seven randomly initialized BERT-style MLM encoders trained with a shared, prespecified 2B-input-token protocol. '
    reply+='We evaluate the same 256 held-out documents using the original CFS intervention and alpha grid, with input-paired bootstrap intervals. '
    reply+=f"Across width 384/512/640, Matched-2 increases ({series(AXES['width'],'matched_2')}), whereas depth 6/8/12 exhibits a non-monotonic pattern ({series(AXES['depth'],'matched_2')}). "
    reply+=f"With exactly 41,404,730 parameters, changing the head decomposition from 4 to 8 to 16 yields Raw Oracle {series(AXES['heads'],'raw')} and Matched-2 {series(AXES['heads'],'matched_2')}. "
    reply+='We also report fixed-K=4 graph controls. These results extend controlled CFS analysis to bidirectional encoders, without establishing universal monotonic scaling or causal performance benefits. '
    reply+='The common-input 1B/2B probes show residual functional drift in some models, so we describe the results as equal-budget comparisons rather than fully converged functional organization. '
    reply+='All intervals condition on one training seed; they do not measure across-training variability.\n'
    (root/'REBUTTAL_WCQn.md').write_text(reply,encoding='utf-8')
    print(json.dumps(dict(training_process_gpu_hours=float(hours),successful_cfs_gpu_hours=cost,report=str(root/'REPORT.md')),indent=2))

if __name__=='__main__':main()
