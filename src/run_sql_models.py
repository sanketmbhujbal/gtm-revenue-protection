import os, sys, duckdb, pandas as pd
from pathlib import Path
SQL_DIR=Path('src/sql'); OUT_DIR=Path('data/processed')
MODEL_ORDER=['stg_daily_usage','int_usage_velocity','int_friction_flags','int_cohort_health','fct_cohort_churn','fct_account_features']
PERSIST={'fct_cohort_churn','fct_account_features','stg_daily_usage','int_cohort_health'}
def main():
    os.makedirs(OUT_DIR,exist_ok=True)
    for f in ['companies.csv','daily_usage.csv','api_errors.csv']:
        if not (Path('data/raw')/f).exists(): print(f'ERROR: data/raw/{f} not found.'); sys.exit(1)
    print('Stage 2: Analytics Engineering')
    con=duckdb.connect()
    for model in MODEL_ORDER:
        sql=(SQL_DIR/f'{model}.sql').read_text()
        print(f'  Running {model}...', end=' ')
        con.execute(f'CREATE OR REPLACE VIEW {model} AS ({sql})')
        if model in PERSIST:
            df=con.execute(f'SELECT * FROM {model}').df()
            df.to_csv(OUT_DIR/f'{model}.csv',index=False)
            print(f'✓  ({len(df):,} rows)')
        else:
            n=con.execute(f'SELECT COUNT(*) FROM {model}').fetchone()[0]
            print(f'✓  ({n:,} rows) [view]')
    # Quality checks
    checks=[
        ('No duplicate company_id in fct_cohort_churn','SELECT COUNT(*)-COUNT(DISTINCT company_id) FROM fct_cohort_churn',0),
        ('Treatment/control mutually exclusive','SELECT COUNT(*) FROM fct_cohort_churn WHERE is_treatment=TRUE AND is_control=TRUE',0),
        ('fct_account_features row count matches companies','SELECT ABS(COUNT(*)-(SELECT COUNT(*) FROM read_csv_auto(\'data/raw/companies.csv\'))) FROM fct_account_features',0),
        ('No nulls in hard_churn_label','SELECT COUNT(*) FROM fct_cohort_churn WHERE hard_churn_label IS NULL',0),
    ]
    print('\n── Quality checks')
    for name,q,expected in checks:
        r=con.execute(q).fetchone()[0]
        print(f'  {name}: {"PASS" if r==expected else f"FAIL ({r})"}')
    summary=con.execute("""SELECT CASE WHEN is_treatment THEN 'Treatment' ELSE 'Control' END AS grp,
        COUNT(*) AS n, SUM(CASE WHEN hard_churn_label THEN 1 ELSE 0 END) AS churned,
        ROUND(AVG(CASE WHEN hard_churn_label THEN 1.0 ELSE 0 END)*100,1) AS churn_pct
        FROM fct_cohort_churn GROUP BY is_treatment ORDER BY is_treatment DESC""").df()
    print('\n── Churn by group:')
    print(summary.to_string(index=False))
    print('\nStage 2 complete.')
if __name__=='__main__': main()
