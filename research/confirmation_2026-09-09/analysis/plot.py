#!/usr/bin/env python3
"""Small static figures with discovery separated from the prospective endpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();result=json.loads(args.analysis.read_text());args.out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(fig,name):
        fig.tight_layout();fig.savefig(args.out/(name+'.svg'));fig.savefig(args.out/(name+'.png'),dpi=180);plt.close(fig)
    est=np.asarray(result['estimates']);ci=np.asarray(result['family_percentile_95_intervals'])
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    axes[0].scatter([0],[0.18906941504940958],color='#7f7f7f',s=50,label='Discovery (selected band)')
    axes[0].errorbar([1],[est[0]],yerr=[[est[0]-ci[0,0]],[ci[0,1]-est[0]]],fmt='o',capsize=4,color='#176da4',label='Confirmation (95% group-bootstrap interval)')
    axes[0].set(xticks=[0,1],xticklabels=['Discovery\n90 items, fit01','Confirmation\n'+str(result['eligible_items'])+' items, fit01'],ylabel='J-Lens minus raw excess hit@10',title='Fixed loop4, physical layers26–37')
    axes[0].axhline(0,color='black',lw=.6);axes[0].margins(x=.4)
    x=np.arange(2)
    axes[1].bar(x-.17,[est[1],est[2]],width=.34,label='J-Lens fit01',color='#176da4')
    axes[1].bar(x+.17,[est[3],est[4]],width=.34,label='Raw logit lens',color='#d47a32')
    axes[1].set(xticks=x,xticklabels=['Intended concepts','Matched controls'],ylabel='Mean fixed-layer hit@10',title='Confirmation: primary-band components');axes[1].legend(frameon=False)
    save(fig,'primary_and_components')
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True,sharey=True)
    for loop,ax in enumerate(axes.flat):
        for arm,color in [('raw','#d47a32'),('fit01','#176da4')]:
            curve=np.asarray(result['descriptive_curves'][arm]['excess'])[loop*48:(loop+1)*48]
            ax.plot(np.arange(1,49),curve,label='Raw' if arm=='raw' else 'J-Lens fit01',color=color)
        if loop==3:ax.axvspan(25.5,37.5,color='#176da4',alpha=.10,label='Preselected primary band')
        ax.axhline(0,color='black',lw=.6);ax.set_title('Loop '+str(loop+1));ax.set_xlim(1,48)
        if loop>=2:ax.set_xlabel('Physical layer (one-based)')
        if loop%2==0:ax.set_ylabel('Intended minus control hit@10')
    axes[0,0].legend(frameon=False);fig.suptitle('Confirmation layer curves — descriptive',y=1.01)
    save(fig,'descriptive_layer_curves')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
    for panel,metric in enumerate(('fixed_mean','any_layer')):
        selected=[s for s in result['secondary_family'] if s['id'].startswith('early_loop') and s['metric']==metric]
        for i,s in enumerate(selected):
            lo,hi=s['simultaneous_95_interval'];e=s['estimate']
            axes[panel].errorbar([i+1],[e],yerr=[[e-lo],[hi-e]],fmt='o',capsize=4,color='#176da4')
        axes[panel].axhline(0,color='black',lw=.6);axes[panel].set(xticks=[1,2,3],xlabel='Loop',title='Fixed-layer mean' if metric=='fixed_mean' else 'Hit anywhere in full loop')
    axes[0].set_ylabel('J-Lens minus raw excess hit@10');fig.suptitle('Early-loop secondaries — shared 95% interval family',y=1.02)
    save(fig,'early_loop_secondaries')

if __name__=='__main__':main()
