"""Re-layout existing Matched-N/Fixed-K summaries without recomputing statistics.

Plot conventions are retained from summarize_matched_scaling.py: parameter axes,
Okabe-Ito colors, circular markers, dashed gray references, and original95% CIs.
Only the two requested PDF/PNG pairs and an export audit are written.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader

COLORS={'2':'#0072B2','4':'#E69F00','5':'#009E73','6':'#009E73',
        '8':'#CC79A7','11':'#D55E00','12':'#D55E00'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel(ax,summary,family,metrics,title,ylabel,legend_location):
    group=summary[summary.family==family]
    unit=1e6 if family=='ViT' else 1e9
    for metric in metrics:
        values=group[group.metric==metric].sort_values('parameters')
        assert len(values)==(3 if family=='ViT' else 4),(family,metric)
        reference=metric.endswith('full') or metric=='raw'
        value=metric.split('_')[-1]
        label='Raw Oracle' if metric=='raw' else ('Full H' if reference else
              ('N=' if metric.startswith('matched_') else 'K=')+value)
        # Same means, intervals, ordering, parameter units, colors and line styles
        # as the original PNG exporter. No bootstrap or model execution here.
        ax.errorbar(values.parameters.to_numpy()/unit,values['mean'],
            yerr=np.maximum(0,np.stack([values['mean']-values.ci_low,values.ci_high-values['mean']])),
            fmt='o--' if reference else 'o-',color='#555555' if reference else COLORS[value],
            markersize=3,capsize=2,linewidth=1.5,label=label)
    ticks=group[['parameters','model']].drop_duplicates().sort_values('parameters')
    ax.set_xticks(ticks.parameters/unit,[f'{v/unit:.3g}' for v in ticks.parameters])
    ax.set(title=title,xlabel='Parameters (M)' if family=='ViT' else 'Parameters (B)',ylabel=ylabel)
    ax.legend(frameon=False,fontsize=7,loc=legend_location,labelspacing=.3,
              handlelength=2.0,handletextpad=.5,borderaxespad=.6)
    ax.grid(alpha=.2)


def export(fig,path,expected_panels):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    boxes=[ax.get_tightbbox(renderer) for ax in fig.axes]
    page=fig.bbox
    for ax,box in zip(fig.axes,boxes):
        assert box.x0>=0 and box.y0>=0 and box.x1<=page.x1 and box.y1<=page.y1,'Clipped panel text'
        ticks=[tick.get_window_extent(renderer) for tick in ax.get_xticklabels() if tick.get_visible()]
        assert all(not left.overlaps(right) for left,right in zip(ticks[:-1],ticks[1:])),'Overlapping x ticks'
    assert all(not left.overlaps(right) for i,left in enumerate(boxes) for right in boxes[i+1:]),'Overlapping panels'
    # Avoid bbox_inches="tight": the PDF MediaBox stays exactly7 inches wide.
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'),dpi=300)
    plt.close(fig)
    pdf=PdfReader(path.with_suffix('.pdf'));assert len(pdf.pages)==1
    page=pdf.pages[0]
    assert abs(float(page.mediabox.width)/72-7)<1e-8
    assert len(page.images)==0,'PDF unexpectedly contains raster images'
    text=page.extract_text()
    assert all(f'({letter})' in text for letter in expected_panels)
    assert page.get('/Resources').get_object().get('/Font') is not None
    return dict(pdf=str(path.with_suffix('.pdf')),png=str(path.with_suffix('.png')),
        width_inches=float(page.mediabox.width)/72,height_inches=float(page.mediabox.height)/72,
        raster_images=len(page.images),subplot_labels=list(expected_panels),
        panel_bounds_and_tick_overlap_checks='passed',pdf_sha256=sha(path.with_suffix('.pdf')))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=Path('results/matched_scaling_rebuttal_20261115/all_results.csv'))
    p.add_argument('--output',type=Path,default=Path('results/matched_scaling_rebuttal_20261115'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    before=sha(args.input);summary=pd.read_csv(args.input)
    assert not summary.duplicated(['family','model','metric']).any()
    plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(7,4.8),layout='constrained')
    fig.get_layout_engine().set(w_pad=.045,h_pad=.045,wspace=.025,hspace=.045)
    panel(axes[0,0],summary,'Qwen',['raw','matched_2','matched_4','matched_8','matched_11'],
          '(a) Qwen2.5: Matched-N','Oracle recovery','upper left')
    panel(axes[0,1],summary,'ViT',['raw','matched_2','matched_4','matched_5'],
          '(b) ViT: Matched-N','Oracle recovery','upper left')
    panel(axes[1,0],summary,'Qwen',['cover_full','cover_4','cover_8','cover_12'],
          '(c) Qwen2.5: Fixed-K Cover','Cover/K','lower left')
    panel(axes[1,1],summary,'ViT',['cover_full','cover_4','cover_6'],
          '(d) ViT: Fixed-K Cover','Cover/K','upper right')
    main=export(fig,args.output/'fig_natural_scaling_controls','abcd')
    fig,axes=plt.subplots(1,2,figsize=(7,2.5),layout='constrained')
    fig.get_layout_engine().set(w_pad=.045,h_pad=.045,wspace=.025)
    panel(axes[0],summary,'Qwen',['rank_full','rank_4','rank_8','rank_12'],
          '(a) Qwen2.5: Fixed-K Rank','Rank/K','lower left')
    panel(axes[1],summary,'ViT',['rank_full','rank_4','rank_6'],
          '(b) ViT: Fixed-K Rank','Rank/K','upper right')
    appendix=export(fig,args.output/'fig_fixedk_rank_appendix','ab')
    assert sha(args.input)==before,'Source summary changed'
    audit=dict(source=str(args.input),source_sha256=before,
        original_plot_code='scripts/summarize_matched_scaling.py',statistics_recomputed=False,
        reference_lines='Raw Oracle and Full H retained; Full H uses normalization by H',
        figures=[main,appendix])
    (args.output/'natural_scaling_controls_export.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
