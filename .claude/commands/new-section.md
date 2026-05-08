---
name: new-section
argument-hint: [section-number] [section-name]
---

Initialize Section $ARGUMENTS for PatrolIQ:

1. Read CLAUDE.md — confirm previous section is marked complete
2. Check previous section has: src file + notebook (if applicable) + tests (if applicable)
   If anything missing — STOP and report before proceeding
3. Create any new subdirectories needed for this section
4. Create notebook skeleton: notebooks/0N_sectionname.py with:
   - Header comment (filename, purpose, section, version)
   - Setup cell: imports + logging config + RANDOM_STATE
   - Load data cell: from data/processed/ only
   - Summary cell placeholder: findings + decisions + next steps
5. Update CLAUDE.md:
   - Move previous section to Completed ✅
   - Set new section as In Progress 🔄
   - List all files to be created this section
6. Say: "Ready to start Section N. Please share the equivalent file(s) from your EMI project as reference."
7. WAIT for user to share reference files before writing any production code
8. Print: "Section N initialized. Files to create: [full list]"
