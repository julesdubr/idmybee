# Conventions

Ce document fixe les conventions du projet. Il sert de référence unique ;
tout code nouveau ou retouché doit s'y conformer. Statut : **validé**
(voir chat).

## Langue

Tout passe en anglais : code, docstrings, commentaires, messages de log,
aide CLI, et les valeurs de champ type `error_reason`/`status` écrites dans
les CSV (ex: `image_illisible_ou_format_non_supporte` ->
`unreadable_image_or_unsupported_format`). Les CSV déjà produits avec des
valeurs en français ne sont pas migrés rétroactivement -- seules les
NOUVELLES lignes écrites après la bascule utilisent les nouveaux libellés.
Ça crée une incohérence entre anciennes et nouvelles données sur ce point
précis ; à garder en tête si un script filtre sur la valeur exacte d'un
`error_reason`. Les identifiants de domaine (espèce, caste...) restent tels
quels (ex: noms latins, `rupestris`).

## Docstrings

Le module `utils/tps_io.py`, `utils/alignment.py`, `utils/run_io.py` et
`utils/pipeline_io.py` sont la référence de style à suivre partout.

**Module** : description courte du rôle, puis seulement ce qui est
non-évident (convention de fichier, piège connu, référence à un bug déjà
corrigé). Pas de rationale historique complète -- un lien vers un commit ou
une phrase suffit.

**Fonction/classe** : une ligne de description, puis inputs/outputs si pas
évidents de la signature. Un exemple d'usage seulement si l'API n'est pas
auto-descriptive. Detail algorithmique uniquement pour les étapes
réellement non triviales (ex. le correctif Kabsch-Umeyama documenté dans
`alignment.py`).

**À éviter** : paraphraser la signature, expliquer *pourquoi* une fonction
existe si ce n'est pas surprenant, dupliquer un `# commentaire` qui répète
juste le nom de la variable suivante.

## Nommage CLI

- Argument requis unique et central du script (dataset, tps, model_path...) :
  **positionnel**. Tout le reste : `--flag` en kebab-case.
  → généraliser le style déjà en place dans `classifiers/train.py`,
  `classifiers/predict.py`, `analysis/variance_report.py` à
  `extraction/*.py` et `landmarks/*.py` (aujourd'hui en `--dataset`
  obligatoire).
- Variable du parser : `parser` (pas `ap`, `p`, etc.).
- Entrée de script :
  ```python
  def main(argv: list[str] | None = None) -> None:
      parser = argparse.ArgumentParser(...)
      ...
      args = parser.parse_args(argv)

  if __name__ == "__main__":
      main()
  ```
  Systématique -- même pour un script "jetable" dans `tools/` -- pour rester
  testable/appelable en dehors du CLI (utile pour la future UI Streamlit).

## Logging

- `logger = logging.getLogger(__name__)` en tête de chaque module qui
  produit des messages de statut/progression.
- `print()` réservé à la sortie destinée à l'utilisateur final en mode CLI
  (un rapport, un tableau récap) -- jamais pour du statut interne
  ("traitement de X...", "N images ok").
- Un seul point de configuration du logging, `setup_console_logging(verbose:
  bool = False)` (déjà présent dans `utils/run_io.py`, à généraliser/déplacer
  au niveau du `cli.py` commun) -- `--verbose`/`--quiet` partagés par tous
  les scripts plutôt que redéfinis à la main.

## Imports

- Le projet est installé en mode éditable (`pip install -e .`, voir
  `pyproject.toml`) -- imports absolus depuis la racine `src/`
  (`from utils.tps_io import ...`, `from landmarks.methods.base import ...`).
- Plus de `sys.path.insert(...)`, plus d'import "nu" qui ne marche qu'en
  exécution directe du script (`from constants import ...` depuis un
  fichier du même dossier). Si un module a besoin d'un autre module du même
  package, import qualifié complet.

## Fonctions core réutilisables (pas seulement CLI)

*(Ajouté session du 2 sept. 2026, suite à la conception de l'outil 1 --
voir `RESUME.md` "Détails outil 1".)*

- Toute étape de pipeline appelée par plusieurs outils (ex. la pose de
  landmarks, consommée à la fois par le mode dataset de l'outil 1 et par
  son mode terrain/single) doit exister comme **fonction Python pure**
  (entrée -> sortie en mémoire, ex. `place_landmarks(image) -> Landmarks`)
  en plus de son entrée CLI/fichier. Le CLI/l'UI appellent cette fonction
  et gèrent l'I/O disque autour -- la fonction elle-même ne lit/écrit
  jamais de fichier.
- Même principe pour le dessin d'overlay annoté (landmarks numérotés sur
  l'image, utilisé par l'étape de validation) :
  `draw_landmarks_overlay(image, landmarks) -> Image`, partagée telle
  quelle entre CLI (export d'overlays sur disque) et UI (affichage direct).
- Objectif : éviter un aller-retour disque (écrire un TPS à une ligne puis
  le relire) pour des usages "ad hoc" comme la prédiction terrain sur une
  photo isolée.

## Tests

- `pytest`, un fichier `tests/test_<module>.py` par module de `src/` qui
  contient de la logique pure (géométrie, parsing, nommage). Pas besoin de
  données réelles : données synthétiques minimales dans le test lui-même.
- Toute fonction dans `utils/gpa.py`, `utils/alignment.py`, `utils/tps_io.py`,
  `utils/outliers.py`, `utils/run_io.py`, `utils/pipeline_io.py` (le futur
  `core/`) doit avoir un test qui ne dépend pas du dataset -- c'est le filet
  de sécurité du refactor à venir.
- Un bug corrigé une fois (ex. le facteur d'échelle Kabsch-Umeyama) doit
  laisser un test de non-régression, pas seulement une note dans le
  docstring.

## Ce qui n'est *pas* couvert ici (volontairement, à traiter dans le
découpage à venir)

- Convention finale de sortie de run (`run_io.py` vs `landmarks_trainer/
  checkpoint.py` vs convention native Ultralytics) -- attend la
  réorganisation en `landmarking/` / `classification/` / `training/`.
- Emplacement final de `utils/` (deviendra probablement `core/`) --
  idem, pas de renommage avant le découpage pour éviter de bouger deux fois.
