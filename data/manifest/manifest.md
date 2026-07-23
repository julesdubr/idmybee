# Manifest idmybee — Phase 0 (indexation en lecture seule)

## Ce que fait `build_manifest.py`

Scanne les racines d'images brutes (locales + disque externe si monte),
sans jamais deplacer/renommer/modifier un fichier. Ecrit 4 CSV dans
`data/manifest/` :

- **images.csv** — une ligne par photo trouvee. `image_id` = hash du
  contenu (pas du nom de fichier), donc stable meme si le fichier est
  copie/deplace, et sert directement de cle de dedup.
- **specimens.csv** — une ligne par `num_inv`, jointe au CSV d'identification.
  `is_labeled=False` => pool de prediction. `in_identification_csv=False`
  avec `in_images=True` => specimen photographie mais jamais identifie.
- **unparsed.csv** — noms de fichiers qui ne suivent aucun des deux schemas
  connus (organized/vrac : `num_inv_[S|P]<n>` ; terrain : `num_inv_<n>`).
  A revoir a la main, rien n'est perdu, juste isole.
- **duplicates.csv** — groupes de fichiers au contenu strictement identique
  (utile pour reperer les copies disque externe / disque local en double).

## Points de vigilance vus sur les tests

- Un nom de fichier qui, par coincidence, matche le schema `num_inv_[S|P]<n>`
  sans etre un vrai `num_inv` connu sera quand meme "parsed_ok" — c'est
  attendu (le script ne peut pas deviner), mais `specimens.csv` donne un
  garde-fou : croiser `in_identification_csv=False` avec un `specimen_id`
  qui a une forme suspecte (trop court, alphabetique, etc.) permet de
  reperer ces faux positifs a la main.
- Une racine externe absente (disque non monte) est simplement ignoree avec
  un avertissement — le script ne plante pas, on peut le relancer avec ou
  sans `--external-roots` selon que le disque est branche.

## Prochaine etape (pas encore fait)

Une fois `images.csv`/`specimens.csv` valides sur le vrai jeu de donnees :
regarder `unparsed.csv` (probablement l'essentiel du dossier `vrac/`, dont
le nommage n'a pas ete precise) pour decider d'une regle de parsing dediee,
ou d'un renommage manuel assiste.

La Phase 1 (`crops.csv`, alimente par `crop_wings.py`) et la Phase 3
(`landmarks.csv`, statuts de numerotation) viendront se brancher sur
`images.csv`/`specimens.csv` via `image_id`/`specimen_id` — rien a refaire
ici pour ca.