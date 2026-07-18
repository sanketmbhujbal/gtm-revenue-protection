"""
Stage 3: Behavior Analysis — Script version
Generates all Stage 3 figures to outputs/figures/.
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

os.makedirs('outputs/figures', exist_ok=True)
ACCENT='#1F4E79'; BLUE2='#2E75B6'; ORANGE='#E07B39'; GREEN='#2E8B57'; RED='#C0392B'
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white',
    'axes.spines.top':False,'axes.spines.right':False,'font.family':'sans-serif'})

def main():
    print("Stage 3: Behavior Analysis")
    churn_df  = pd.read_csv('data/processed/fct_cohort_churn.csv', parse_dates=['signup_date','peak_date'])
    usage_df  = pd.read_csv('data/processed/stg_daily_usage.csv',  parse_dates=['usage_date','signup_date'])
    errors_df = pd.read_csv('data/raw/api_errors.csv',             parse_dates=['timestamp'])
    n_total   = len(churn_df)
    n_t       = int(churn_df['is_treatment'].sum())
    n_c       = n_total - n_t
    avg_onset = churn_df[churn_df['is_treatment']]['days_to_hard_churn'].mean()
    print(f"  Companies: {n_total} | Treatment: {n_t} | Control: {n_c}")
    print(f"  Treatment avg churn onset: day {avg_onset:.1f}")

    churned = churn_df[churn_df['hard_churn_label']==True].copy()

    # Fig 1: churn timeline
    fig, axes = plt.subplots(1,2,figsize=(13,4.5))
    ax=axes[0]
    ax.hist(churned['days_to_hard_churn'],bins=20,color=ACCENT,alpha=0.85,edgecolor='white')
    ax.axvline(churned['days_to_hard_churn'].median(),color=ORANGE,lw=2,linestyle='--',
               label=f'Median={churned["days_to_hard_churn"].median():.0f}d')
    ax.set_title('Distribution of Churn Onset Day'); ax.set_xlabel('Day Since Signup')
    ax.set_ylabel('Companies'); ax.legend(frameon=False)
    ax2=axes[1]
    cumulative=[(churn_df['days_to_hard_churn']<=d).sum()/n_total*100 for d in range(91)]
    ax2.plot(range(91),cumulative,color=ACCENT,lw=2.5)
    ax2.fill_between(range(91),cumulative,alpha=0.12,color=ACCENT)
    ax2.set_title('Cumulative Churn Rate'); ax2.set_xlabel('Day Since Signup')
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    for d in [30,60,90]:
        ax2.annotate(f'Day {d}: {cumulative[d]:.1f}%',xy=(d,cumulative[d]),
                     xytext=(d+2,cumulative[d]+1.5),fontsize=9,color=ACCENT)
    plt.tight_layout(); plt.savefig('outputs/figures/churn_timeline.png',dpi=150,bbox_inches='tight'); plt.close()
    print("  Saved: outputs/figures/churn_timeline.png")

    # Fig 2: pre-churn signal
    churn_timing = churn_df[churn_df['hard_churn_label']==True][['company_id','days_to_hard_churn']].copy()
    merged = usage_df.merge(churn_timing,on='company_id')
    merged['churn_onset_date'] = merged['signup_date']+pd.to_timedelta(merged['days_to_hard_churn'],unit='D')
    merged['days_to_onset'] = (merged['usage_date']-merged['churn_onset_date']).dt.days
    window    = merged[(merged['days_to_onset']>=-21)&(merged['days_to_onset']<=7)]
    avg_curve = window.groupby('days_to_onset')['total_tokens'].mean()
    ctrl_avg  = usage_df[usage_df['company_id'].isin(churn_df[~churn_df['is_treatment']]['company_id'])]['total_tokens'].mean()
    fig,ax=plt.subplots(figsize=(11,4.5))
    ax.plot(avg_curve.index,avg_curve.values,color=ACCENT,lw=2.5,marker='o',markersize=4,label='Friction group avg')
    ax.axhline(ctrl_avg,color=GREEN,lw=1.8,linestyle='--',label=f'Control avg ({ctrl_avg:,.0f})')
    ax.axvline(0,color=RED,lw=1.5,linestyle=':',alpha=0.8,label='Churn onset (day 0)')
    ax.fill_between(avg_curve.index,avg_curve.values,alpha=0.08,color=ACCENT)
    ax.set_title('Avg Daily Token Volume — Indexed to Churn Onset')
    ax.set_xlabel('Days Relative to Churn Onset'); ax.set_ylabel('Avg Total Tokens')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x/1000:.0f}K'))
    ax.legend(frameon=False); ax.set_xticks(range(-21,8,3))
    plt.tight_layout(); plt.savefig('outputs/figures/pre_churn_signal.png',dpi=150,bbox_inches='tight'); plt.close()
    print("  Saved: outputs/figures/pre_churn_signal.png")

    # Fig 3: cohort heatmap
    usage_b = usage_df.merge(churn_df[['company_id','friction_bucket']],on='company_id')
    bins=[0,14,28,42,56,70,90]
    labels=['Days 0-14','Days 15-28','Days 29-42','Days 43-56','Days 57-70','Days 71-90']
    usage_b['day_bin']=pd.cut(usage_b['day_since_signup'],bins=bins,labels=labels,right=False)
    hm=(usage_b.groupby(['friction_bucket','day_bin'],observed=True)['total_tokens']
        .mean().unstack('day_bin').reindex(['Severe','Mild','None']).reindex(columns=labels))
    hm_norm=hm.div(hm.iloc[:,0],axis=0)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,3.5))
    im1=ax1.imshow(hm.values,aspect='auto',cmap='Blues')
    ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels,rotation=35,ha='right',fontsize=9)
    ax1.set_yticks(range(3)); ax1.set_yticklabels(['Severe','Mild','None'],fontsize=10)
    ax1.set_title('Avg Daily Token Volume by Friction Bucket')
    for i in range(3):
        for j in range(len(labels)):
            v=hm.values[i,j]
            if not np.isnan(v):
                ax1.text(j,i,f'{v/1000:.0f}K',ha='center',va='center',fontsize=8.5,
                         color='white' if v>np.nanmax(hm.values)*0.6 else ACCENT)
    plt.colorbar(im1,ax=ax1)
    im2=ax2.imshow(hm_norm.values,aspect='auto',cmap='RdYlGn',vmin=0,vmax=2)
    ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels,rotation=35,ha='right',fontsize=9)
    ax2.set_yticks(range(3)); ax2.set_yticklabels(['Severe','Mild','None'],fontsize=10)
    ax2.set_title('Token Volume Relative to Onboarding Baseline')
    for i in range(3):
        for j in range(len(labels)):
            v=hm_norm.values[i,j]
            if not np.isnan(v):
                ax2.text(j,i,f'{v:.2f}x',ha='center',va='center',fontsize=8.5,color='black')
    plt.colorbar(im2,ax=ax2)
    plt.tight_layout(); plt.savefig('outputs/figures/cohort_heatmap.png',dpi=150,bbox_inches='tight'); plt.close()
    print("  Saved: outputs/figures/cohort_heatmap.png")

    # Fig 4: friction event distribution
    company_signups=usage_df[['company_id','signup_date']].drop_duplicates()
    err_m=errors_df.merge(company_signups,on='company_id')
    err_m['day_since_signup']=(err_m['timestamp'].dt.normalize()-err_m['signup_date']).dt.days
    t_ids=churn_df[churn_df['is_treatment']]['company_id'].tolist()
    c_ids=churn_df[~churn_df['is_treatment']]['company_id'].tolist()
    ob_err=err_m[err_m['company_id'].isin(t_ids)&err_m['day_since_signup'].between(0,13)]
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    ax=axes[0]
    r429=(ob_err[ob_err['error_code']==429].groupby('day_since_signup').size().reindex(range(14),fill_value=0))
    ax.bar(r429.index,r429.values,color=RED,alpha=0.8,edgecolor='white')
    ax.set_title('HTTP 429 Events by Onboarding Day (Treatment Group)')
    ax.set_xlabel('Day Since Signup'); ax.set_ylabel('Total 429 Events'); ax.set_xticks(range(14))
    ax2=axes[1]
    ob_u=usage_df[usage_df['day_since_signup'].between(0,13)]
    ttft_t=ob_u[ob_u['company_id'].isin(t_ids)].groupby('day_since_signup')['avg_ttft_ms'].mean()
    ttft_c=ob_u[ob_u['company_id'].isin(c_ids)].groupby('day_since_signup')['avg_ttft_ms'].mean()
    ax2.plot(ttft_t.index,ttft_t.values,color=RED,lw=2.2,marker='o',ms=4,label='Treatment')
    ax2.plot(ttft_c.index,ttft_c.values,color=BLUE2,lw=2.2,marker='s',ms=4,label='Control')
    ax2.set_title('Avg TTFT During Onboarding — Treatment vs Control')
    ax2.set_xlabel('Day Since Signup'); ax2.set_ylabel('Avg TTFT (ms)')
    ax2.set_xticks(range(14)); ax2.legend(frameon=False)
    plt.tight_layout(); plt.savefig('outputs/figures/friction_event_distribution.png',dpi=150,bbox_inches='tight'); plt.close()
    print("  Saved: outputs/figures/friction_event_distribution.png")

    # Fig 5: churn by segment
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    ax=axes[0]
    ic=(churn_df.groupby(['industry','is_treatment'])['hard_churn_label']
        .mean().unstack('is_treatment').rename(columns={True:'Treatment',False:'Control'})*100)
    x=np.arange(len(ic)); w=0.35
    ax.bar(x-w/2,ic.get('Treatment',pd.Series(dtype=float)),w,label='Treatment',color=RED,alpha=0.8)
    ax.bar(x+w/2,ic.get('Control',pd.Series(dtype=float)),  w,label='Control',  color=BLUE2,alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(ic.index,fontsize=10)
    ax.set_ylabel('Hard Churn Rate (%)'); ax.set_title('Churn Rate by Industry')
    ax.set_ylim(0,115); ax.legend(frameon=False)
    ax2=axes[1]
    td=(churn_df[churn_df['is_treatment']].groupby('contract_tier')['days_to_hard_churn']
        .agg(['mean','std','count']))
    bars=ax2.bar(td.index,td['mean'],color=[ACCENT,ORANGE],alpha=0.85,edgecolor='white')
    ax2.errorbar(td.index,td['mean'],yerr=td['std'],fmt='none',color='#333',capsize=5,lw=1.5)
    for bar,(idx,row) in zip(bars,td.iterrows()):
        ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,
                 f'n={int(row["count"])}',ha='center',fontsize=9,color='#444')
    ax2.set_ylabel('Avg Days to Hard Churn')
    ax2.set_title('Time to Churn by Contract Tier (Treatment Only)'); ax2.set_ylim(0,90)
    plt.tight_layout(); plt.savefig('outputs/figures/churn_by_segment.png',dpi=150,bbox_inches='tight'); plt.close()
    print("  Saved: outputs/figures/churn_by_segment.png")
    print(f"\nStage 3 complete. Treatment churn: {churn_df[churn_df['is_treatment']]['hard_churn_label'].mean():.1%}  Control: {churn_df[~churn_df['is_treatment']]['hard_churn_label'].mean():.1%}")

if __name__ == '__main__':
    main()
