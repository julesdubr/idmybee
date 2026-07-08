https://github.com/GabrielPatry/BumblebeeWingsMeasurement

- Le code pour la génération GT est la fonction `creation_relief_ulti_v2` qui se trouve dans le fichier `UNet_class_and_functions.py`

- Le code pour le prétraitement de la base  1 est dans le notebook `umons_dataset_openingandprocessing.ipynb`

- Le code pour le train est dans le notebook `UNet_training.ipynb`

- Le code pour l'inférence est dans le notebook `base_2_predictions.ipynb` (pour base non annotée) et `Model_evaluation.ipynb` (pour base annotée)

- Le meilleur modèle est dans le dossier best_model

- Le code pour le prétraitement de la base d'Adrien est dans le notebook `Wing_segmentation_clean.ipynb`

Quelques commentaires : 

    J'ai tous fait tourner en local, donc il y a sans doute des choses é changer pour pouvoir faire tourner les codes sur collab (les chemins d'accés notamment et peut-étre d'autres points). En régle générale il faut toujours créer des dossiers manuellement et changer les chemins d'accés, je n'ai pas automatiser la création de dossier au lancement des codes de sauvegarde des images aprés traitement.
    J'ai systématiquement utilisé cuda quand cela était possible, mais je pense que l'inférence du UNet doit pouvoir se faire sans dans un temps raisonnable. Pour l'entrainement de modèle, et l'utilisation de SAM en local, ça peut-étre plus compliqué.
    Tout ce qui concerne les modèles et l'entrainement est codé avec PyTorch.


