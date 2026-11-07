"""Render experimental figures from supplied measurement CSVs.

No experiments are executed. Measurement sources are listed in FIGURE-PROVENANCE.csv; raw algorithm IDs are unchanged.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();DATA=a.data_root;OUT=a.output_dir
INPUTS={'control_accuracy_summary.csv':'control_accuracy/raw/control_accuracy_summary.csv','summary_by_H.csv':'horizon_scaling/data/derived/summary_by_H.csv','runtime-statistics.csv':'runtime-statistics.csv','speedup-statistics.csv':'speedup-statistics.csv'}
OUT.mkdir(parents=True,exist_ok=True)
BLUE='#0072B2'; ORANGE='#D55E00'; GREY='#666666'; GREEN='#009E73'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.5,
    'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
    'grid.alpha':.16,'grid.linewidth':.4,'pdf.fonttype':42,'lines.linewidth':1.1,
    'lines.markersize':3.5})
def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),metadata={'Creator':'Matplotlib','Author':None})
    fig.savefig(OUT/(name+'.png'),dpi=200)
    plt.close(fig)

m1=pd.read_csv(DATA/'control_accuracy/raw/control_accuracy_summary.csv')
ie=m1[(m1.algorithm=='IEVI_Waterfall_common_VI')&(m1.delta_solve==.01)].sort_values('lambda')
tight=m1[(m1.algorithm=='IEVI_Waterfall_common_VI')&(m1.delta_solve==1e-8)].sort_values('lambda')
fig,ax=plt.subplots(1,2,figsize=(5.5,2.65),layout='constrained')
ax[0].plot(ie['lambda'],ie.stable_N_P,'o-',color=BLUE,label='Adaptive IE-VI')
ax[0].axhline(49,color=GREY,ls='--',label='VI (49 calls)')
ax[0].set(yscale='log',xlabel=r'Corruption $\lambda$',ylabel='True calls to stable hit',title='(a) True-model queries')
ax[0].legend(loc='upper left')
for k,c,m in [(.001,GREEN,'o'),(.01,BLUE,'s'),(.1,ORANGE,'^')]:
    ax[1].plot(ie['lambda'],(ie.stable_N_P+k*ie.stable_N_Phat)/49,marker=m,color=c,label=rf'$\kappa={k:g}$')
ax[1].axhline(1,color=GREY,ls='--',label='VI')
ax[1].set(yscale='log',xlabel=r'Corruption $\lambda$',ylabel='Weighted cost / VI cost',title='(b) Surrogate cost matters')
fig.legend(*ax[1].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False)
save(fig,'E1-weighted-cost')

ratio=tight.stable_N_Phat.to_numpy()/ie.stable_N_Phat.to_numpy()
assert np.array_equal(tight.stable_N_P,ie.stable_N_P)
fig,ax=plt.subplots(1,2,figsize=(5.5,2.65),layout='constrained')
ax[0].plot(ie['lambda'],tight.stable_N_P,'o-',color=GREY,label=r'$\delta=10^{-8}$')
ax[0].plot(ie['lambda'],ie.stable_N_P,'s--',mfc='white',color=BLUE,label=r'$\delta=10^{-2}$')
ax[0].set(xlabel=r'Corruption $\lambda$',ylabel='True calls to stable hit',yscale='log',title='(a) Counts coincide')
ax[0].legend(loc='upper left')
ax[1].plot(ie['lambda'],ratio,'o-',color=BLUE)
ax[1].axhline(np.median(ratio),ls='--',color=ORANGE,label=f'Median: {np.median(ratio):.2f}')
ax[1].axhline(1,color=GREY,lw=.7)
ax[1].set(xlabel=r'Corruption $\lambda$',ylabel='Tight / loose surrogate work',ylim=(.8,3.5),title='(b) Surrogate work falls')
ax[1].legend(loc='lower right')
save(fig,'E2-solve-tolerance')

m2=pd.read_csv(DATA/'horizon_scaling/data/derived/summary_by_H.csv')
fig=plt.figure(figsize=(5.5,4.7),layout='constrained')
gs=fig.add_gridspec(2,2,height_ratios=[.8,1])
a=fig.add_subplot(gs[0,:]); b=fig.add_subplot(gs[1,0]); c=fig.add_subplot(gs[1,1])
for schedule,color,marker,label in [('fixed80',ORANGE,'s','80 samples per row'),('linear16',BLUE,'o',r'$16H$ samples per row')]:
    d=m2[(m2.schedule==schedule)&(m2.algorithm=='IE-VI Joint')].sort_values('H')
    a.loglog(d.H,d.median_E_infinity,color=color,marker=marker,label=label)
    a.fill_between(d.H,d.q25_E_infinity,d.q75_E_infinity,color=color,alpha=.12)
a.set(xlabel=r'Effective horizon $H$',ylabel=r'Model error $E_\infty$',title='(a) Surrogate accuracy')
a.legend(loc='lower left')
for ax,schedule,title in [(b,'fixed80','(b) Fixed sampling budget'),(c,'linear16','(c) Budget grows with horizon')]:
    for alg,color,marker,ls,label in [('VI',GREY,'x','--','VI'),('IE-VI Joint',BLUE,'o','-','IE-VI (refined schedule)'),('IE-VI Eq16',ORANGE,'^',':','IE-VI (conservative schedule)'),('OS-VI',GREEN,'s','-.','OS-VI')]:
        d=m2[(m2.schedule==schedule)&(m2.algorithm==alg)].sort_values('H')
        ax.loglog(d.H,d.median_N_P,color=color,marker=marker,ls=ls,label=label)
        if alg=='IE-VI Joint': ax.fill_between(d.H,d.q25_N_P,d.q75_N_P,color=color,alpha=.12)
    ax.set(xlabel=r'Effective horizon $H$',ylabel='Median true calls',title=title,ylim=(1,50000))
fig.legend(*b.get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
save(fig,'E3-horizon-scaling')

runtime=pd.read_csv(DATA/'runtime-statistics.csv')
methods=['VI','IEVI_VI_Ks4','IEVI_VI_Ks8','IEVI_VI_Ks16','IEVI_VI_Ks32']
wt=runtime[runtime.metric=='runtime_seconds'].set_index('method').loc[methods]
ss=pd.read_csv(DATA/'speedup-statistics.csv').set_index('method').loc[methods[1:]]
fig,ax=plt.subplots(1,2,figsize=(5.5,2.65),layout='constrained')
ax[0].errorbar(range(5),wt['mean'],yerr=wt.se,fmt='o',capsize=3,color=BLUE)
ax[0].set(xticks=range(5),xticklabels=['VI','IE:4','IE:8','IE:16','IE:32'],ylabel='Solver time (seconds)',title='(a) Solver time')
ax[1].errorbar(range(4),ss['mean'],yerr=ss.se,fmt='o-',capsize=3,color=BLUE)
ax[1].axhline(1,ls='--',color=GREY)
ax[1].set(xticks=range(4),xticklabels=['4','8','16','32'],xlabel='Surrogate simulation steps',ylabel='Paired VI / IE-VI time',title='(b) Speedup over VI')
save(fig,'E4-wallclock-fidelity')
print('Rendered four vector figures from the supplied measurements.')
