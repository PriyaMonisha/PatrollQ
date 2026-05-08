---
paths:
  - "src/data/**/*.py"
  - "scripts/**/*.py"
---

# Data Pipeline Rules — PatrolIQ

## The Cardinal Rule
data/raw/ is READ ONLY — always and forever.
Outputs go to data/processed/ only.
If writing to data/raw/, STOP immediately — you are wrong.

## Data Format
- Processed data: CSV with gzip compression (.csv.gz)
- pandas reads: pd.read_csv('file.csv.gz') — works transparently
- pandas writes: df.to_csv(path, index=False, compression='gzip')
- Artifacts: plain .csv (small enough), .json for metrics

## Sampling Protocol
- Always take MOST RECENT 500K records (sort by Date DESC)
- Rationale: recent crimes are more actionable for current policing
- Record this in sample_metadata.json: {n_records, date_min, date_max, sampled_at}

## Known Data Quirks (Never "Fix" These)
- Some records have null Latitude/Longitude — drop these (they can't be clustered geographically)
- Beat/District/Ward may have missing values — mode impute is correct
- Date column may parse with timezone info — strip tz for consistency
- Primary_Type has 33 categories — all are valid, none should be merged

## Loading Standards
- Always validate shape after loading: assert len(df) > 0
- Always log: rows, columns, null counts, memory usage
- Use dtype hints when loading for performance (avoid pandas guessing)
- Never modify DataFrame in place without comment

## Feature Engineering Boundary
- preprocessor.py: cleaning, type fixing, basic extraction (hour, day, etc.)
- engineer.py: derived features (cyclical encoding, severity scores, normalization)
- NEVER mix these — preprocessing must run before feature engineering
