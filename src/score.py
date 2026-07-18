import os, pickle, warnings, argparse
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap, xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

os.makedirs('outputs/figures',exist_ok=True); os.makedirs('models',exist_ok=True)

FEATURE_COLS=['count_429_onboarding','count_500_onboarding','count_503_onboarding','total_errors_onboarding',
    'avg_ttft_ms_onboarding','max_ttft_ms_onboarding','avg_ttft_days_0_14','max_ttft_days_0_14',
    'avg_context_saturation_onboarding','max_context_saturation_onboarding','avg_cache_ratio_onboarding',
    'avg_cache_ratio_0_14','avg_ctx_sat_0_14','max_ctx_sat_0_14','avg_tokens_days_0_14',
    'avg_wow_pct_change_week2','token_growth_slope_onboarding','growth_rate']

PRESCRIPTIVE_RULES=[
    {'trigger':'High rate limit frequency','condition':lambda r:r['count_429_onboarding']>=3,'action':'Increase RPM quota — escalate to Solutions Engineering','owner':'AE + SE'},
    {'trigger':'High TTFT (latency)','condition':lambda r:r['avg_ttft_ms_onboarding']>=400,'action':'Route to lower-latency deployment or regional endpoint','owner':'Infra + AE'},
    {'trigger':'Context window saturation','condition':lambda r:r['max_context_saturation_onboarding']>=0.90,'action':'Recommend upgrade to larger context model','owner':'AE'},
    {'trigger':'Elevated server error rate','condition':lambda r:r['count_500_onboarding']+r['count_503_onboarding']>=3,'action':'Flag for infrastructure investigation','owner':'Engineering'},
    {'trigger':'Low prompt cache utilisation','condition':lambda r:r['avg_cache_ratio_onboarding']<0.10,'action':'Proactive prompt caching workshop','owner':'Customer Success'},
]
DEFAULT={'trigger':'Declining velocity','action':'Proactive check-in call — use-case review','owner':'AE'}

def get_prescription(row):
    for rule in PRESCRIPTIVE_RULES:
        try:
            if rule['condition'](row): return rule['trigger'],rule['action'],rule['owner']
        except: continue
    return DEFAULT['trigger'],DEFAULT['action'],DEFAULT['owner']

def load_features():
    df=pd.read_csv('data/processed/fct_account_features.csv')
    le_ind=LabelEncoder(); le_tier=LabelEncoder()
    df['industry_enc']=le_ind.fit_transform(df['industry'])
    df['contract_tier_enc']=le_tier.fit_transform(df['contract_tier'])
    X=df[FEATURE_COLS+['industry_enc','contract_tier_enc']].fillna(0)
    y=df['hard_churn_label'].astype(int)
    return X,y,df

def train_model(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42,stratify=y)
    spw=(y_train==0).sum()/max((y_train==1).sum(),1)
    model=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.1,subsample=0.8,
        colsample_bytree=0.8,scale_pos_weight=spw,random_state=42,eval_metric='logloss',verbosity=0)
    model.fit(X_train,y_train,eval_set=[(X_test,y_test)],verbose=False)
    y_proba=model.predict_proba(X_test)[:,1]
    y_pred=(y_proba>=0.5).astype(int)
    roc=roc_auc_score(y_test,y_proba); pr=average_precision_score(y_test,y_proba)
    print(f'\n── Model evaluation\n  ROC-AUC: {roc:.4f}  PR-AUC: {pr:.4f}')
    print(classification_report(y_test,y_pred,target_names=['No Churn','Churn'],digits=3))
    cv=cross_val_score(model,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring='roc_auc')
    print(f'  5-fold CV ROC-AUC: {cv.mean():.4f} ± {cv.std():.4f}')
    cm=confusion_matrix(y_test,(y_proba>=0.75).astype(int))
    print(f'  Confusion matrix (0.75): TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}')
    # ROC/PR plot
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    RocCurveDisplay.from_predictions(y_test,y_proba,ax=axes[0],name=f'XGBoost (AUC={roc:.3f})')
    axes[0].lines[0].set_color('#1F4E79'); axes[0].plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
    axes[0].set_title('ROC Curve')
    PrecisionRecallDisplay.from_predictions(y_test,y_proba,ax=axes[1],name=f'XGBoost (AP={pr:.3f})')
    axes[1].lines[0].set_color('#1F4E79'); axes[1].set_title('Precision-Recall Curve')
    for ax in axes: ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig('outputs/figures/model_roc_pr.png',dpi=150,bbox_inches='tight'); plt.close()
    return model

def compute_shap(model,X):
    explainer=shap.TreeExplainer(model)
    shap_values=explainer.shap_values(X)
    mean_abs=np.abs(shap_values).mean(axis=0)
    shap_df=pd.DataFrame({'feature':X.columns,'mean_abs_shap':mean_abs}).sort_values('mean_abs_shap').tail(10)
    fig,ax=plt.subplots(figsize=(9,5))
    bars=ax.barh(shap_df['feature'],shap_df['mean_abs_shap'],color='#1F4E79',alpha=0.85)
    ax.set_xlabel('Mean |SHAP value|'); ax.set_title('Top 10 Features — XGBoost Churn Risk Model (SHAP)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for bar,val in zip(bars,shap_df['mean_abs_shap']):
        ax.text(val+0.001,bar.get_y()+bar.get_height()/2,f'{val:.3f}',va='center',fontsize=8.5)
    plt.tight_layout(); plt.savefig('outputs/figures/shap_barchart.png',dpi=150,bbox_inches='tight'); plt.close()
    print('  Saved: outputs/figures/shap_barchart.png')
    return shap_values

def score_and_prescribe(model,X,df):
    proba=model.predict_proba(X)[:,1]
    pres=df.apply(get_prescription,axis=1,result_type='expand')
    pres.columns=['top_friction_trigger','recommended_action','action_owner']
    scored=pd.DataFrame({
        'company_id':df['company_id'],'company_name':df['company_name'],
        'industry':df['industry'],'contract_tier':df['contract_tier'],
        'friction_bucket':df['friction_bucket'],'churn_probability':proba.round(4),
        'risk_tier':pd.cut(proba,bins=[-0.01,0.50,0.75,0.85,1.01],labels=['Low','Medium','High','Critical']),
        'top_friction_trigger':pres['top_friction_trigger'],'recommended_action':pres['recommended_action'],
        'action_owner':pres['action_owner'],'hard_churn_label':df['hard_churn_label'].astype(int),
        'days_to_hard_churn':df['days_to_hard_churn'],
        'count_429_onboarding':df['count_429_onboarding'],'avg_ttft_ms_onboarding':df['avg_ttft_ms_onboarding'],
        'avg_cache_ratio_onboarding':df['avg_cache_ratio_onboarding'],
        'max_context_saturation_onboarding':df['max_context_saturation_onboarding'],
        'avg_tokens_days_0_14':df['avg_tokens_days_0_14'],'token_growth_slope_onboarding':df['token_growth_slope_onboarding'],
    }).sort_values('churn_probability',ascending=False).reset_index(drop=True)
    scored.to_csv('data/processed/scored_accounts.csv',index=False)
    return scored

def generate_counterfactuals(model,X,df):
    ARR={'Enterprise':120_000,'Growth':24_000}
    scenarios=[
        {'name':'Baseline (no change)','ttft_mult':1.0,'rl_red':0.0},
        {'name':'TTFT reduced by 25%','ttft_mult':0.75,'rl_red':0.0},
        {'name':'TTFT reduced by 50%','ttft_mult':0.50,'rl_red':0.0},
        {'name':'Rate limits eliminated','ttft_mult':1.0,'rl_red':1.0},
        {'name':'TTFT -50% + rate limits fixed','ttft_mult':0.50,'rl_red':1.0},
    ]
    base_proba=model.predict_proba(X)[:,1]
    rows=[]
    for s in scenarios:
        Xcf=X.copy()
        for c in [col for col in Xcf.columns if 'ttft' in col.lower()]: Xcf[c]*=s['ttft_mult']
        for c in [col for col in Xcf.columns if '429' in col]: Xcf[c]*=(1-s['rl_red'])
        cf_proba=model.predict_proba(Xcf)[:,1]
        saved_mask=(base_proba>=0.75)&(cf_proba<0.75)
        n_saved=saved_mask.sum()
        arr_impact=int(sum(ARR.get(df.iloc[i]['contract_tier'],24_000) for i in np.where(saved_mask)[0]))
        rows.append({'scenario':s['name'],'ttft_reduction_pct':int((1-s['ttft_mult'])*100),
                     'rate_limit_reduction_pct':int(s['rl_red']*100),
                     'accounts_moved_to_safe':int(n_saved),'estimated_arr_retained':arr_impact})
    cf_df=pd.DataFrame(rows)
    cf_df.to_csv('data/processed/counterfactual_scenarios.csv',index=False)
    return cf_df

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--no-retrain',action='store_true'); args=parser.parse_args()
    MODEL_PATH='models/xgb_churn_model.pkl'
    print('Stage 5: Prediction + Prescription')
    X,y,df=load_features()
    print(f'  Features: {X.shape[0]} companies, {X.shape[1]} cols  |  Churn rate: {y.mean():.1%}')
    if args.no_retrain and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH,'rb') as f: model=pickle.load(f)
    else:
        model=train_model(X,y)
        with open(MODEL_PATH,'wb') as f: pickle.dump(model,f)
        print(f'  Model saved.')
    compute_shap(model,X)
    scored=score_and_prescribe(model,X,df)
    print(f'\n── Scoring\n  {len(scored)} accounts scored')
    print(scored['risk_tier'].value_counts().sort_index().to_string())
    cf=generate_counterfactuals(model,X,df)
    print('\n── Counterfactuals')
    print(cf[['scenario','accounts_moved_to_safe','estimated_arr_retained']].to_string(index=False))
    print('\nStage 5 complete.')

if __name__=='__main__': main()
