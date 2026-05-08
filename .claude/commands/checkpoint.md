---
name: checkpoint
argument-hint: [what-was-just-completed]
---

Save progress checkpoint: $ARGUMENTS

1. Read current CLAUDE.md
2. Mark the completed item in Progress Tracker:
   Change "- [ ]" to "- [x]" for: $ARGUMENTS
3. Update "Active Section" with current status
4. Update "Last Working File" with file just completed
5. Update "Last Decision Made" with any new design decisions this session
6. Check: are ALL files for current section now complete?
   If yes → print: "Section N looks complete. Run /review-section N before marking done."
   If no  → print: "Remaining in this section: [list remaining files]"
7. Git commit: "section-N: $ARGUMENTS complete"
8. Print full summary: what's done, what's next in this section
