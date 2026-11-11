"""
Loan Default Risk Analysis  (Give Me Some Credit dataset)
Run:  python loan_default_analysis.py cs-training.csv
Creates an  outputs/  folder with:
  cleaning_log.txt, loans_clean.csv (use in Power BI), q*.csv (SQL results),
  stats_results.txt, model_results.txt, feature_importance.csv, loans.db
Read the printed output top to bottom -- every section maps to one resume bullet.
"""
import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.inspection import permutation_importance

try:  # use XGBoost if you have it (you do in your EV project), else sklearn's boosting
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

csv_path = sys.argv[1] if len(sys.argv) > 1 else "cs-training.csv"
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

RENAME = {
    "SeriousDlqin2yrs": "defaulted",
    "RevolvingUtilizationOfUnsecuredLines": "utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "late_90",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89",
    "NumberOfDependents": "dependents",
}

# ---------------------------------------------------------------- 1. LOAD
df = pd.read_csv(csv_path)
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
df = df.rename(columns=RENAME)
missing_cols = set(RENAME.values()) - set(df.columns)
if missing_cols:
    sys.exit(f"Columns not found after rename: {missing_cols}. Check your CSV header.")

# ---------------------------------------------------------------- 2. VALIDATE + CLEAN
log = []
def note(msg):
    log.append(msg)
    print(msg)

print("\n===== 2. DATA VALIDATION =====")
note(f"Rows loaded: {len(df):,}")
note(f"Default rate (raw): {df['defaulted'].mean():.2%}")
for c, n in df.isna().sum()[lambda s: s > 0].items():
    note(f"Missing values in {c}: {n:,} ({n/len(df):.1%})")

dups = df.duplicated().sum()
df = df.drop_duplicates()
note(f"Dropped {dups:,} duplicate rows")

bad_age = (df["age"] < 18).sum()
df = df[df["age"] >= 18].copy()
note(f"Dropped {bad_age:,} rows with age < 18 (invalid for a borrower)")

late_cols = ["late_30_59", "late_60_89", "late_90"]
for c in late_cols:
    coded = df[c].isin([96, 98]).sum()
    df.loc[df[c].isin([96, 98]), c] = np.nan
    note(f"{c}: {coded:,} values coded 96/98 (placeholder codes, not real counts) set to missing")
    df[c] = df[c].fillna(df[c].median())

df["income_missing"] = df["income"].isna().astype(int)
note(f"Added income_missing flag ({df['income_missing'].sum():,} rows) before imputing income with the median")
df["income"] = df["income"].fillna(df["income"].median())
df["dependents"] = df["dependents"].fillna(df["dependents"].median())

for c in ["utilization", "debt_ratio", "income"]:
    cap = df[c].quantile(0.99)
    n_cap = (df[c] > cap).sum()
    df[c] = df[c].clip(upper=cap)
    note(f"{c}: capped {n_cap:,} extreme values at the 99th percentile ({cap:,.2f})")

note(f"Rows after cleaning: {len(df):,}")
note(f"Default rate (clean): {df['defaulted'].mean():.2%}  <- class imbalance, so accuracy is NOT a useful metric")
(OUT / "cleaning_log.txt").write_text("\n".join(log))
df.to_csv(OUT / "loans_clean.csv", index=False)

# ---------------------------------------------------------------- 3. SQL ANALYSIS
print("\n===== 3. SQL ANALYSIS =====")
db_path = OUT / "loans.db"
con = sqlite3.connect(db_path)
df.to_sql("loans", con, if_exists="replace", index=False)

AGE = """CASE WHEN age < 30 THEN '1) 18-29' WHEN age < 40 THEN '2) 30-39'
             WHEN age < 50 THEN '3) 40-49' WHEN age < 60 THEN '4) 50-59' ELSE '5) 60+' END"""
INC = """CASE WHEN income < 3000 THEN '1) <3k' WHEN income < 5000 THEN '2) 3-5k'
             WHEN income < 8000 THEN '3) 5-8k' ELSE '4) 8k+' END"""
UTIL = """CASE WHEN utilization < 0.3 THEN '1) <30%' WHEN utilization < 0.6 THEN '2) 30-60%'
              WHEN utilization < 0.9 THEN '3) 60-90%' ELSE '4) 90%+' END"""

QUERIES = {
    "q01_overall": """
        SELECT COUNT(*) AS borrowers, SUM(defaulted) AS defaulters,
               ROUND(100.0 * AVG(defaulted), 2) AS default_rate_pct
        FROM loans""",
    "q02_by_age_band": f"""
        SELECT {AGE} AS age_band, COUNT(*) AS borrowers,
               ROUND(100.0 * AVG(defaulted), 2) AS default_rate_pct
        FROM loans GROUP BY 1 ORDER BY 1""",
    "q03_by_income_band": f"""
        SELECT {INC} AS monthly_income_band, COUNT(*) AS borrowers,
               ROUND(100.0 * AVG(defaulted), 2) AS default_rate_pct
        FROM loans GROUP BY 1 ORDER BY 1""",
    "q04_by_utilization": f"""
        SELECT {UTIL} AS credit_utilization, COUNT(*) AS borrowers,
               ROUND(100.0 * AVG(defaulted), 2) AS default_rate_pct
        FROM loans GROUP BY 1 ORDER BY 1""",
    "q05_by_90day_lates": """
        SELECT CAST(MIN(late_90, 4) AS INT) AS times_90_days_late_capped_at_4, COUNT(*) AS borrowers,
               ROUND(100.0 * AVG(defaulted), 2) AS default_rate_pct
        FROM loans GROUP BY 1 ORDER BY 1""",
    "q06_by_dependents": """
        SELECT MIN(CAST(dependents AS INT), 4) AS dependents_capped_at_4, COUNT(*) AS borrowers,
               ROUND(100.0 * AVG(defaulted), 2) AS default_rate_pct
        FROM loans GROUP BY 1 ORDER BY 1""",
    # CTE + window functions: rank segments and show how much of all defaults each one holds
    "q07_risk_segments_ranked": f"""
        WITH seg AS (
            SELECT {AGE} AS age_band,
                   CASE WHEN late_30_59 + late_60_89 + late_90 > 0 THEN 'Has late payments'
                        ELSE 'No late payments' END AS payment_history,
                   COUNT(*) AS borrowers, SUM(defaulted) AS defaulters
            FROM loans GROUP BY 1, 2
        )
        SELECT age_band, payment_history, borrowers, defaulters,
               ROUND(100.0 * defaulters / borrowers, 2) AS default_rate_pct,
               RANK() OVER (ORDER BY 1.0 * defaulters / borrowers DESC) AS risk_rank,
               ROUND(100.0 * defaulters / SUM(defaulters) OVER (), 2) AS share_of_all_defaults_pct
        FROM seg ORDER BY risk_rank""",
    # lift vs the overall default rate
    "q08_lift_vs_overall": f"""
        WITH overall AS (SELECT AVG(defaulted) AS r FROM loans),
        seg AS (
            SELECT {UTIL} AS credit_utilization, {INC} AS income_band,
                   COUNT(*) AS borrowers, AVG(defaulted) AS rate
            FROM loans GROUP BY 1, 2 HAVING COUNT(*) >= 200
        )
        SELECT credit_utilization, income_band, borrowers,
               ROUND(100.0 * rate, 2) AS default_rate_pct,
               ROUND(rate / (SELECT r FROM overall), 2) AS lift_vs_overall
        FROM seg ORDER BY lift_vs_overall DESC LIMIT 10""",
}
for name, sql in QUERIES.items():
    res = pd.read_sql_query(sql, con)
    res.to_csv(OUT / f"{name}.csv", index=False)
    print(f"\n--- {name} ---")
    print(res.to_string(index=False))
con.close()

# ---------------------------------------------------------------- 4. STATISTICS
print("\n===== 4. HYPOTHESIS TESTS =====")
stats_lines = []
d, nd = df[df.defaulted == 1], df[df.defaulted == 0]
for col in ["income", "age", "debt_ratio"]:
    u, p = mannwhitneyu(d[col], nd[col], alternative="two-sided")
    stats_lines.append(
        f"Mann-Whitney {col}: median defaulters = {d[col].median():,.2f}, non-defaulters = {nd[col].median():,.2f}, p = {p:.3g}"
    )
df["any_90_late"] = (df["late_90"] > 0).astype(int)
tab = pd.crosstab(df["any_90_late"], df["defaulted"])
chi2, p, _, _ = chi2_contingency(tab)
r = df.groupby("any_90_late")["defaulted"].mean()
stats_lines.append(
    f"Chi-square, any 90+ day late vs default: default rate {r.get(1, np.nan):.1%} (late) vs {r.get(0, np.nan):.1%} (never late), p = {p:.3g}"
)
stats_lines.append(
    "Note: with ~150k rows almost everything is 'significant'. Look at the size of the difference, not just the p-value."
)
print("\n".join(stats_lines))
(OUT / "stats_results.txt").write_text("\n".join(stats_lines))

# ---------------------------------------------------------------- 5. MODEL
print("\n===== 5. MODEL =====")
features = ["utilization", "age", "late_30_59", "debt_ratio", "income", "open_credit_lines",
            "late_90", "real_estate_loans", "late_60_89", "dependents", "income_missing"]
X, y = df[features], df["defaulted"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
if HAVE_XGB:
    gb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(),
                       eval_metric="auc", random_state=42)
    gb_name = "XGBoost"
else:
    gb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                        class_weight="balanced", random_state=42)
    gb_name = "Gradient Boosting (sklearn)"

out_lines = []
def top_decile_capture(y_true, prob):
    k = int(len(prob) * 0.10)
    idx = np.argsort(prob)[::-1][:k]
    return np.asarray(y_true)[idx].sum() / np.asarray(y_true).sum()

for name, model in [("Logistic Regression", lr), (gb_name, gb)]:
    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    out_lines += [
        f"\n{name}",
        f"  ROC-AUC = {roc_auc_score(yte, prob):.3f}   PR-AUC = {average_precision_score(yte, prob):.3f}"
        f"   (a model that guesses randomly gets PR-AUC ~ {yte.mean():.3f})",
        f"  Top 10% highest-risk borrowers contain {top_decile_capture(yte, prob):.1%} of all defaulters",
        classification_report(yte, (prob >= 0.5).astype(int), digits=3,
                              target_names=["no default", "default"]),
    ]
print("\n".join(out_lines))
(OUT / "model_results.txt").write_text("\n".join(out_lines))

imp_idx = Xte.sample(min(20000, len(Xte)), random_state=42).index
pi = permutation_importance(gb, Xte.loc[imp_idx], yte.loc[imp_idx], scoring="roc_auc",
                            n_repeats=5, random_state=42)
imp = pd.DataFrame({"feature": features, "importance_auc_drop": pi.importances_mean}) \
        .sort_values("importance_auc_drop", ascending=False)
imp.to_csv(OUT / "feature_importance.csv", index=False)
print(f"\nFeature importance ({gb_name}, permutation):")
print(imp.to_string(index=False))
print("\nDone. Open the outputs/ folder. Use loans_clean.csv (+ the q*.csv files) in Power BI.")
