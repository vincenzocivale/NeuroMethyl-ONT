# MethyLYZR label mapping (R15)

Maps our 91 GSE90496 classes to the WHO-level labels of the MethyLYZR Zenodo cohort (R9 75 runs, R10 207
libraries), so that 91-class probabilities can be evaluated on it by summing merged classes. This is a documented,
defensible mapping, not a medical decision. Files:

- `configs/taxonomies/methylyzr_who_level.yaml` (version 1): every class with `coarse`, `confidence`
  (clear / ambiguous), `alt_coarse` (the other plausible target, not applied) and `rationale`; plus the truth-side
  `cohort_labels`. The key layout is meant to be reused for `sturgeon_general_87.yaml`.
- `src/neuromethyl_ont/evaluation/taxonomy_mapping.py`: `load_taxonomy`, `map_to_coarse(probs, class_names, tax)`
  (sums mass, raises unless the 91 classes are each present exactly once), `coarse_truth(label, tax)`.
- `registry/methylyzr_cohort_groups.tsv` (built by `scripts/data/build_methylyzr_manifest.py`) and
  `src/neuromethyl_ont/evaluation/cohort_groups.py` (`build_manifest`, `grouped_folds`).

```python
tax = load_taxonomy("configs/taxonomies/methylyzr_who_level.yaml")
coarse_probs, labels = map_to_coarse(probs91, class_names91, tax)   # columns follow tax["coarse_labels"]
truth, abstain = coarse_truth(label_short, tax)                      # abstain=True for metastases
folds = grouped_folds(manifest.split_group, manifest.coarse_label, n_splits=5, seed=0)
```

## Mapping (default decisions)

Prediction side has 14 coarse columns: the 13 cohort CNS labels plus `OTHER_91`, the mass of the 56 classes that no
cohort label covers (43 clear, 13 ambiguous, below). Basis: class names of Capper 2018 and MethyLYZR's own
Capper-to-MZ table (Brandl et al. Supp. Table 1), where it exists.

| Coarse label | Clear classes | Ambiguous classes (default shown here, alternative in the YAML) |
|---|---|---|
| A IDH | A IDH; A IDH, HG | - |
| O IDH | O IDH | - |
| GBM | GBM, MES; GBM, MYCN; GBM, RTK I; GBM, RTK II; GBM, RTK III | GBM, MID |
| PXA | PXA | - |
| MNG | MNG | - |
| MB | MB, G3; MB, G4; MB, SHH CHL AD; MB, SHH INF; MB, WNT | - |
| EPN | EPN, MPE; EPN, PF A; EPN, PF B; EPN, RELA; EPN, SPINE; EPN, YAP | - |
| LYMPHO | LYMPHO | - |
| PA | LGG, PA MID; LGG, PA PF | ANA PA; LGG, PA/GG ST |
| PLEX | PLEX, AD; PLEX, PED A; PLEX, PED B | - |
| ATRT | ATRT, MYC; ATRT, SHH; ATRT, TYR | - |
| DMG, K27 | DMG, K27 | - |
| CONTR, REACT | CONTR, REACT | - |

`OTHER_91`, ambiguous (alternative target in brackets): GBM, G34 (GBM); IHG (GBM); SUBEPN, PF / SPINE / ST (EPN);
CONTR, ADENOPIT / CEBM / HEMI / HYPTHAL / INFLAM / PINEAL / PONS / WM (CONTR, REACT).
`OTHER_91`, clear: the remaining 43 classes (pituitary, pineal, embryonal other than MB/ATRT, other LGG,
melanoma, plasmacytoma, schwannoma, ...), see the YAML.

### Metastasis libraries
27 R10 libraries (lung 13, breast 6, melanoma 4, colorectal 3, endometrial 1) get truth label `OUT_OF_SCOPE`
(truth side only; never a prediction column) with `abstention_test = 1` in the manifest. Treat them as an
abstention test: any confident call is a false positive. Exception to note: the 91 classes do contain MELAN
(melanoma), so a MELAN call on the 4 melanoma metastases is arguably correct; the YAML carries a note.

## Patient groups (manifest)
`split_group = patient_id` (the cohort's own pseudonymous `IEGxx`; no age, sex or site). 282 libraries,
**192 groups**; the 33 patients in both R9 and R10 and all repeat runs (R9 up to 6) and `_2` libraries (34) share one group.
Group size: 1 library 126 groups, 2: 51, 3: 9, 4: 5, 7: 1 (IEG114). No patient has conflicting labels.

| Coarse label | groups (patients) | libraries |
|---|---|---|
| A IDH | 44 | 70 |
| GBM | 63 | 85 |
| O IDH | 38 | 58 |
| MB | 10 | 13 |
| EPN | 5 | 7 |
| PXA | 2 | 6 |
| LYMPHO | 2 | 4 |
| MNG | 2 | 4 |
| PA | 2 | 3 |
| PLEX | 1 | 2 |
| ATRT | 1 | 1 |
| CONTR, REACT | 1 | 1 |
| DMG, K27 | 1 | 1 |
| OUT_OF_SCOPE (metastases) | 20 | 27 |

`grouped_folds` uses sklearn `StratifiedGroupKFold(shuffle=True, random_state=seed)` on the coarse label. Classes
with fewer groups than folds (PLEX, ATRT, CONTR REACT, DMG K27 have one) cannot appear in every test fold; the
rare-class warning is suppressed on purpose. Use per-class results over the pooled folds, not per fold.

## Open questions for Vincenzo
1. **GBM, G34**: keep outside GBM (MethyLYZR and WHO CNS5 treat it as a separate type) or merge into GBM because
   the cohort's pre-2021 "GBM" labels may include it? Changes the GBM column only.
2. **IHG** (infantile hemispheric glioma): same question, default outside GBM.
3. **GBM, MID** is in GBM (by name and MethyLYZR). Midline overlaps DMG, K27; keep, or treat as ambiguous in reports?
4. **EPN vs subependymoma**: does the cohort's EPN label include SUBEPN? Default no (separate WHO type).
5. **PA**: keep `ANA PA` and `LGG, PA/GG ST` in PA (MethyLYZR's choice) or move to OTHER_91?
6. **Controls**: only CONTR, REACT counts as the cohort label `CONTR, REACT`; should all 9 control classes merge
   (one library only, so low stakes)?
7. **Melanoma metastases**: abstention test (default) or count a MELAN call as correct?
8. **PXA** is kept as a plain Capper code (clear); R10 EPIC calls include `X PXA`, whose relation to our PXA class was not checked.
9. Should classes without a cohort label stay as one `OTHER_91` bucket (default), or be scored as an explicit
   "none of the cohort classes" prediction? The mapping function supports both (it is one column).

## Known limits of the cohort
- Labels are WHO-level postoperative diagnoses, about 13 CNS classes (most with 1 to 3 patients); only 22 R9 patients
  have a Heidelberg v11b4 subclass. A coarse-label accuracy is not comparable with a 91-class accuracy.
- Patients overlap with the Capper 2018 training set was not excluded (not verified); the v11b4 labels come from a
  classifier trained on the same data.
- The MethyLYZR authors tuned their 0.6 threshold on these samples: do not copy it, and treat any comparison on this cohort as
  not independent of their design choices (relevant for success criterion 5).
- 33 patients are in both R9 and R10: never split them (use `split_group`).
- Class sizes are so small that per-class confidence intervals are wide; pool over seeds and folds.

## Not verified
Class names were read from the GSE90496 labels (91 classes, identical to the YAML) and the Capper-to-MZ CSV; the
mapping of cohort labels onto Capper classes was not checked against pathology reports, and no model was evaluated here.
