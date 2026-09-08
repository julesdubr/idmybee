"""Dataset-agnostic landmarking/classification orchestrators, plus the
validation-review tools that sit between them (see CONVENTIONS.md "Statuts
harmonisés" and PIPELINE.md "Orchestrator scripts").

    export_final_landmarks.py   stage 6: R-facing TPS+CSV package for one
                                 already-processed dataset root
    export_review.py            writes crops/landmarks review CSVs + overlay
                                 images for manual inspection (CLI counterpart
                                 of the Scenario 1 UI's validation steps)
    reconcile_review.py         re-applies a hand-edited review CSV as an
                                 override consumed by train/predict/export
    train_dataset.py            stages 2-6 then classifiers.train -> model.joblib
    predict_dataset.py          stages 2-6 then classifiers.predict batch

train_dataset.py/predict_dataset.py/export_final_landmarks.py delegate the
shared stages 2-5 to utils.landmarking_pipeline rather than duplicating them.
export_final_landmarks.py is the one script in this package imported by
another module (utils.landmarking_pipeline.run_export) -- every other file
here is launched standalone only.
"""
