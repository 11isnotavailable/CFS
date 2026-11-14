"""Input-cluster bootstrap, paired contrasts and publication figures for gating."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def bootstrap_stat(values, draws):
    values = np.asarray(values, dtype=np.float64)
    sampled = values[draws].mean(1)
    return {'mean': float(values.mean()), 'sd_inputs': float(values.std(ddof=1)),
            'ci_low': float(np.quantile(sampled,.025)), 'ci_high': float(np.quantile(sampled,.975))}, sampled


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(args.evaluation/'per_input.csv')
    samples = sorted(frame['sample'].unique())
    assert len(samples)==448
    assert not frame.duplicated(['sample','seed','method','keep','compensation']).any()
    assert frame.groupby('sample').dense_prediction.nunique().max()==1
    draws = np.random.default_rng(42).integers(0,len(samples),(args.bootstrap,len(samples)))
    np.savez_compressed(args.output/'bootstrap_input_draws.npz',indices=draws)
    archive, summaries, seed_rows = {}, [], []
    averaged = frame.groupby(['sample','method','keep','compensation'])[['kl','correct']].mean().reset_index()
    for keys, rows in frame.groupby(['method','keep','compensation','seed']):
        method, keep, comp, seed = keys
        rows = rows.sort_values('sample'); assert len(rows)==448
        for metric in ('kl','correct'):
            stat, _ = bootstrap_stat(rows[metric], draws)
            seed_rows.append(dict(method=method,keep=keep,compensation=comp,seed=seed,metric=metric,**stat))
    seeds = pd.DataFrame(seed_rows)
    for keys, rows in averaged.groupby(['method','keep','compensation']):
        method, keep, comp = keys
        rows = rows.sort_values('sample')
        for metric in ('kl','correct'):
            stat, sampled = bootstrap_stat(rows[metric], draws)
            means = seeds.query('method==@method and keep==@keep and compensation==@comp and metric==@metric')['mean']
            summaries.append(dict(method=method,keep=keep,compensation=comp,metric=metric,
                                  sd_seed_means=float(means.std(ddof=1)),**stat))
            archive[f'{method}_{keep}_{comp}_{metric}'] = sampled
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output/'summary.csv',index=False)
    summary.to_json(args.output/'summary.json',orient='records',indent=2)
    seeds.to_csv(args.output/'per_seed_summary.csv',index=False)
    dense = pd.read_csv(args.evaluation/'dense_inputs.csv').sort_values('sample')
    dense_stat, _ = bootstrap_stat(dense.correct,draws)
    (args.output/'dense_accuracy.json').write_text(json.dumps(dense_stat,indent=2))
    contrasts, seed_contrasts = [], []
    def vector(method, keep, comp, metric='kl'):
        rows = averaged.query('method==@method and keep==@keep and compensation==@comp').sort_values('sample')
        assert rows['sample'].tolist()==samples
        return rows[metric].to_numpy()
    definitions = [
        ('Gating_minus_R2_hard','Gating','hard','R2','hard'),
        ('Gating_minus_R2_shared_merge','Gating','shared_r2_merge','R2','shared_r2_merge'),
        ('Gating_minus_R2_joint_only_hard','Gating','hard','R2_joint_only','hard'),
        ('Gating_hard_minus_R2_full','Gating','hard','R2','shared_r2_merge'),
        ('R2_compensation_gain','R2','hard','R2','shared_r2_merge'),
        ('Gating_compensation_gain','Gating','hard','Gating','shared_r2_merge'),
        ('Taylor_minus_R2_hard','Taylor','hard','R2','hard')]
    for name,left,lc,right,rc in definitions:
        for keep in (3,6,9):
            for metric in ('kl','correct'):
                diff = vector(left,keep,lc,metric)-vector(right,keep,rc,metric)
                stat, sampled = bootstrap_stat(diff,draws)
                contrasts.append(dict(contrast=name,keep=keep,metric=metric,left=left,left_comp=lc,
                                      right=right,right_comp=rc,**stat))
                archive[f'{name}_{keep}_{metric}'] = sampled
                for seed in (17,23,41):
                    a=frame.query('seed==@seed and method==@left and keep==@keep and compensation==@lc').sort_values('sample')
                    b=frame.query('seed==@seed and method==@right and keep==@keep and compensation==@rc').sort_values('sample')
                    stat_seed,_=bootstrap_stat(a[metric].to_numpy()-b[metric].to_numpy(),draws)
                    seed_contrasts.append(dict(contrast=name,keep=keep,metric=metric,seed=seed,**stat_seed))
    paired = pd.DataFrame(contrasts)
    paired.to_csv(args.output/'paired_bootstrap.csv',index=False)
    paired.to_json(args.output/'paired_bootstrap.json',orient='records',indent=2)
    pd.DataFrame(seed_contrasts).to_csv(args.output/'per_seed_paired_bootstrap.csv',index=False)
    np.savez_compressed(args.output/'bootstrap_statistics.npz',**archive)
    for comp, filename in [('hard','shared_hard'),('shared_r2_merge','shared_learned_compensation')]:
        sub = summary[(summary.compensation==comp)&(summary.metric=='kl')]
        table = sub.pivot(index='method',columns='keep',values='mean')
        table.to_csv(args.output/f'{filename}_table.csv')
        table.to_latex(args.output/f'{filename}_table.tex',float_format='%.5f')
    full = summary[((summary.method=='R2')&(summary.compensation=='shared_r2_merge')) |
                   ((summary.method.isin(['Gating','Taylor']))&(summary.compensation=='hard'))]
    full.to_csv(args.output/'full_pipeline_table.csv',index=False)
    full.to_latex(args.output/'full_pipeline_table.tex',index=False,float_format='%.5f')
    summary[summary.metric=='correct'].to_csv(args.output/'accuracy_table.csv',index=False)
    # Complete CI tables accompany the compact mean tables.
    summary.to_latex(args.output/'all_results_with_ci.tex',index=False,float_format='%.5f')
    paired.to_latex(args.output/'paired_bootstrap.tex',index=False,float_format='%.5f')
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'figure.dpi':150})
    colors={'Taylor':'#666666','Gating':'#0072B2','R2':'#D55E00','R2_joint_only':'#009E73'}
    labels={'Taylor':'Taylor','Gating':'Learned gating','R2':'Original CFS R2','R2_joint_only':'R2 structure, joint KL only'}
    fig, axes = plt.subplots(1,2,figsize=(7,2.8),sharey=True)
    for ax,comp,title in zip(axes,['hard','shared_r2_merge'],['Shared hard gating','Shared R2 compensation']):
        for method in colors:
            rows=summary.query('method==@method and compensation==@comp and metric=="kl"').sort_values('keep')
            means=rows['mean'].to_numpy()
            ax.plot(rows.keep,means,'o-',ms=3,label=labels[method],color=colors[method])
            ax.fill_between(rows.keep,rows.ci_low,rows.ci_high,color=colors[method],alpha=.14)
        ax.set(title=title,xlabel='Retained heads K',xticks=[3,6,9]);ax.grid(alpha=.15)
    axes[0].set_ylabel('KL(dense || routed)')
    axes[0].legend(fontsize=6.5,frameon=False)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(args.output/f'figure1_dynamic_routing.{suffix}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.5,2.6))
    for offset,(name,label,color) in enumerate([
        ('Gating_minus_R2_hard','Original R2, hard','#D55E00'),
        ('Gating_minus_R2_shared_merge','Original R2, shared merge','#0072B2'),
        ('Gating_minus_R2_joint_only_hard','Joint-only R2, hard','#009E73')]):
        rows=paired.query('contrast==@name and metric=="kl"').sort_values('keep')
        ax.errorbar(rows.keep+(offset-1)*.18,rows['mean'],
                    yerr=np.stack([rows['mean']-rows.ci_low,rows.ci_high-rows['mean']]),
                    fmt='o',ms=3,capsize=3,label=label,color=color)
    ax.axhline(0,color='black',lw=.7);ax.set(xticks=[3,6,9],xlabel='Retained heads K',ylabel='KL(gating) − KL(comparator)')
    ax.legend(fontsize=6.5,frameon=False);ax.grid(alpha=.15);fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(args.output/f'figure2_paired_difference.{suffix}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    print(summary[summary.metric=='kl'].to_string(index=False))
    print(paired[paired.metric=='kl'].to_string(index=False))


if __name__ == '__main__':
    main()
