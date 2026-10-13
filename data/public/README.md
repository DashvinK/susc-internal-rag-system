# data/public — committed, safe-to-share corpus

Everything in this folder is bundled into the public index (`index/`) and ships
in the demo. **Only put non-confidential, shareable material here.**

> ⚠️ The documents currently in this folder are **SAMPLE PLACEHOLDERS** —
> fictional, generic content written only so the pipeline runs end-to-end.
> Replace them with real, cleared SUSC documents, then rebuild:
>
> ```bash
> python -m src.ingest.pipeline --full
> ```

## Before you commit a real corpus

1. Keep anything confidential (internal minutes, member contact details, or
   anything you hold as Director of Extracurricular that isn't public) in
   `data/private/` — it is gitignored and never enters the index.
2. After building, **manually read `index/chunks.json`** — it is the one file
   that actually ships. Confirm nothing sensitive leaked in.
3. Filenames are used as citation keys, so keep them human-readable and unique.
