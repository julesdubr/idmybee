"""Standalone CLI entry points, grouped by responsibility (see CONVENTIONS.md
"Placement d'un fichier"):

    tools.ingestion    raw/messy data -> a clean manifest.csv/biological_data.csv
                        (see tools.ingestion.prepare_dataset for the Step 0
                        orchestrator chaining the whole stage).
    tools.pipeline      dataset-agnostic orchestrators (detection -> crop ->
                        landmarks -> renumbering -> export -> train/predict)
                        plus the review export/reconciliation tools.
    tools.maintenance   one-off, ad hoc scripts (TPS cleanup, image repair,
                        format conversion) -- not part of the regular pipeline.

Every script here is launched with `python -m tools.<package>.<script>` and
is never imported by another module except tools.pipeline.export_final_landmarks
(reused in-process by utils.landmarking_pipeline.run_export).
"""
