"""Stage 0/1: turns raw or third-party data into the clean
manifest.csv/biological_data.csv contract every downstream pipeline stage
reads (see core.dataset.load_dataset).

    ingest_raw.py            raw image root(s) -> raw manifest.csv (identity-free)
    export_clean_dataset.py  raw manifest.csv + raw identification CSV ->
                              clean, canonically-named dataset.csv (optional,
                              only needed when the raw data is genuinely messy)
    build_manifest.py        any per-photo CSV -> manifest.csv/biological_data.csv
                              (always run, the sole entry point into stage 2+)
    combine_manifests.py     several already-built manifest.csv/biological_data.csv
                              roots -> one combined root
    prepare_dataset.py       Step 0 orchestrator chaining the four scripts
                              above from a single JSON config

See PIPELINE.md section 1 for the full stage-by-stage contract.
"""
