---
name: review-section
argument-hint: [section-number]
---

Full quality review of Section $ARGUMENTS before marking complete:

1. Read CLAUDE.md — get expected file list for section $ARGUMENTS
2. For each source file:
   a. File exists in src/ or scripts/
   b. Header comment present: filename + purpose + version
   c. No print() statements: grep for print( in file
   d. No hardcoded paths: grep for "data/" or "artifacts/" inline
   e. Type hints on all function signatures
   f. No bare except: clauses
3. Run tests if they exist: pytest tests/ -v -k "relevant" --tb=short
4. Check notebook exists in notebooks/ (if section has one)
5. ML checks (sections 4–7 only):
   - RANDOM_STATE = 42 present in all model code
   - MLflow logging present in training code
   - config.py constants used (not hardcoded hyperparams)
   - No model.fit() in Streamlit pages
   - @st.cache_data on data loading in pages/
6. Artifact check (sections 5–8): verify expected CSVs and JSONs exist in artifacts/
7. Report: ✓ PASS or ✗ FAIL for every single check
8. If all PASS → "Section $ARGUMENTS ready. Run /checkpoint to mark complete."
9. If any FAIL → "Fix these before proceeding: [exact list]"
