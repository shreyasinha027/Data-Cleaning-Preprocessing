import os
import re
import zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


INPUT_ZIP = r"C:\Users\Shreya\Documents\data cleaning\Uncleaned_DS_jobs.csv.zip"
OUTPUT_CLEANED = r"C:\Users\Shreya\Documents\cleaned_ds_jobs.csv"
OUTPUT_ML_READY = r"C:\Users\Shreya\Documents\cleaned_ds_jobs_ml_ready.csv"

def find_column(df, names):
    """
    Return the real column name from df that matches any name in names.
    Matching is case-insensitive and ignores extra spaces.
    """
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        key = name.strip().lower()
        if key in normalized:
            return normalized[key]
    return None

def parse_salary(s):
    """
    Parse salary like '$137K-$171K (Glassdoor est.)'
    Returns: min_salary, max_salary, avg_salary
    """
    if pd.isna(s):
        return np.nan, np.nan, np.nan

    text = str(s)
    nums = re.findall(r"\d+", text)

    if len(nums) >= 2:
        low, high = int(nums[0]), int(nums[1])
        if "K" in text.upper():
            low *= 1000
            high *= 1000
        return low, high, (low + high) / 2

    return np.nan, np.nan, np.nan

def split_company_name(x):
    """
    Example:
    'Healthfirst\\n3.1' -> ('Healthfirst', 3.1)
    """
    if pd.isna(x):
        return np.nan, np.nan

    parts = str(x).split("\n")
    company = parts[0].strip()

    rating = np.nan
    if len(parts) > 1:
        try:
            rating = float(parts[1].strip())
        except:
            rating = np.nan

    return company, rating

def extract_state(x):
    """
    Example:
    'New York, NY' -> 'NY'
    """
    if pd.isna(x):
        return np.nan
    text = str(x).strip()
    if "," in text:
        return text.split(",")[-1].strip()
    return np.nan

def make_job_simp(title):
    if pd.isna(title):
        return "unknown"

    t = str(title).lower()

    if "director" in t:
        return "director"
    elif "vice president" in t or "vp" in t:
        return "vp"
    elif "senior" in t or "sr" in t:
        return "senior"
    elif "junior" in t or "jr" in t or "entry" in t:
        return "junior"
    elif "lead" in t:
        return "lead"
    elif "manager" in t:
        return "manager"
    else:
        return "data_scientist"

def make_seniority(title):
    if pd.isna(title):
        return "unknown"

    t = str(title).lower()

    if "director" in t or "vice president" in t or "vp" in t:
        return "executive"
    elif "senior" in t or "sr" in t:
        return "senior"
    elif "junior" in t or "jr" in t or "entry" in t:
        return "junior"
    elif "lead" in t:
        return "lead"
    elif "manager" in t:
        return "manager"
    else:
        return "not_specified"

def keyword_flag(text, keyword):
    if pd.isna(text):
        return 0
    return int(keyword.lower() in str(text).lower())

def iqr_filter(df, columns):
    """
    Remove outliers using IQR on selected columns.
    """
    mask = pd.Series(True, index=df.index)

    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= df[col].between(lower, upper)

    return df.loc[mask].copy()


if not os.path.exists(INPUT_ZIP):
    raise FileNotFoundError(f"File not found: {INPUT_ZIP}")

with zipfile.ZipFile(INPUT_ZIP) as zf:
    csv_name = zf.namelist()[0]
    df = pd.read_csv(zf.open(csv_name))

clean = df.copy()


clean.columns = (
    clean.columns.astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

print("Columns found in file:")
print(clean.columns.tolist())


salary_col = find_column(clean, ["Salary Estimate", "Salary", "salary estimate"])
job_title_col = find_column(clean, ["Job Title", "job title"])
job_desc_col = find_column(clean, ["Job Description", "job description"])
company_name_col = find_column(clean, ["Company Name", "company name"])
location_col = find_column(clean, ["Location", "location"])
hq_col = find_column(clean, ["Headquarters", "headquarters"])
founded_col = find_column(clean, ["Founded", "founded"])
competitors_col = find_column(clean, ["Competitors", "competitors"])
rating_col = find_column(clean, ["Rating", "rating"])
index_col = find_column(clean, ["index"])


if index_col is not None:
    clean = clean.drop(columns=[index_col])



if salary_col is not None:
    clean[["min_salary", "max_salary", "avg_salary"]] = clean[salary_col].apply(
        lambda x: pd.Series(parse_salary(x))
    )
else:
    print("Warning: Salary column not found. Creating empty salary features.")
    clean["min_salary"] = np.nan
    clean["max_salary"] = np.nan
    clean["avg_salary"] = np.nan

if company_name_col is not None:
    clean[["company_name", "company_rating_from_name"]] = clean[company_name_col].apply(
        lambda x: pd.Series(split_company_name(x))
    )
else:
    clean["company_name"] = np.nan
    clean["company_rating_from_name"] = np.nan


if job_title_col is not None:
    clean["job_simp"] = clean[job_title_col].apply(make_job_simp)
    clean["seniority"] = clean[job_title_col].apply(make_seniority)
else:
    clean["job_simp"] = "unknown"
    clean["seniority"] = "unknown"

if location_col is not None:
    clean["job_state"] = clean[location_col].apply(extract_state)
else:
    clean["job_state"] = np.nan

if hq_col is not None:
    clean["hq_state"] = clean[hq_col].apply(extract_state)
else:
    clean["hq_state"] = np.nan

clean["same_state"] = (clean["job_state"] == clean["hq_state"]).astype(int)


if founded_col is not None:
    clean["company_age"] = clean[founded_col].replace(-1, np.nan)
    clean["company_age"] = clean["company_age"].apply(
        lambda x: np.nan if pd.isna(x) else 2020 - x
    )
else:
    clean["company_age"] = np.nan


if competitors_col is not None:
    def count_competitors(x):
        if pd.isna(x) or str(x).strip() in ["-1", ""]:
            return 0
        return len([c for c in str(x).split(",") if c.strip()])
    clean["num_competitors"] = clean[competitors_col].apply(count_competitors)
else:
    clean["num_competitors"] = 0


keywords = ["python", "excel", "hadoop", "spark", "aws", "tableau", "big data"]
if job_desc_col is not None:
    for kw in keywords:
        col_name = kw.replace(" ", "_")
        clean[col_name] = clean[job_desc_col].apply(lambda x, k=kw: keyword_flag(x, k))
else:
    for kw in keywords:
        clean[kw.replace(" ", "_")] = 0


clean = clean.replace("-1", np.nan)


for c in clean.select_dtypes(include=["object", "string"]).columns:
    clean[c] = clean[c].replace(
        {
            "Unknown / Non-Applicable": np.nan,
            "Unknown": np.nan,
            "N/A": np.nan,
            "None": np.nan,
            "": np.nan,
        }
    )


clean = clean.drop_duplicates().reset_index(drop=True)


num_cols = clean.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    if clean[c].isna().any():
        clean[c] = clean[c].fillna(clean[c].median())


cat_cols = clean.select_dtypes(include=["object", "string"]).columns.tolist()
for c in cat_cols:
    if clean[c].isna().any():
        mode_vals = clean[c].mode(dropna=True)
        fill_value = mode_vals.iloc[0] if len(mode_vals) > 0 else "unknown"
        clean[c] = clean[c].fillna(fill_value)


clean = iqr_filter(clean, ["avg_salary", "company_age", "Rating"])


clean.to_csv(OUTPUT_CLEANED, index=False)
print(f"Saved cleaned file to: {OUTPUT_CLEANED}")


# ML PREPROCESSING

ml_df = clean.copy()


drop_cols = [
    salary_col,
    job_desc_col,
    company_name_col,
    job_title_col,
    location_col,
    hq_col,
    competitors_col,
    founded_col,
]
drop_cols = [c for c in drop_cols if c is not None]

ml_df = ml_df.drop(columns=drop_cols, errors="ignore")


cat_cols_ml = ml_df.select_dtypes(include=["object", "string"]).columns.tolist()
ml_df = pd.get_dummies(ml_df, columns=cat_cols_ml, drop_first=True)


bool_cols = ml_df.select_dtypes(include=["bool"]).columns
ml_df[bool_cols] = ml_df[bool_cols].astype(int)


binary_cols = [
    c for c in ml_df.columns
    if pd.api.types.is_numeric_dtype(ml_df[c]) and set(pd.unique(ml_df[c])).issubset({0, 1})
]

scale_cols = [
    c for c in ml_df.columns
    if pd.api.types.is_numeric_dtype(ml_df[c]) and c not in binary_cols
]

if scale_cols:
    scaler = StandardScaler()
    ml_df[scale_cols] = scaler.fit_transform(ml_df[scale_cols])

# Final check
if ml_df.isna().sum().sum() != 0:
    raise ValueError("Missing values still exist in ML-ready data.")

ml_df.to_csv(OUTPUT_ML_READY, index=False)
print(f"Saved ML-ready file to: {OUTPUT_ML_READY}")

print("Original shape:", df.shape)
print("Cleaned shape:", clean.shape)
print("ML-ready shape:", ml_df.shape)