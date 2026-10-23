# IARI label-mapping verification

Verified 23 October 2026. This is documentation-level verification, not independent diagnosis of individual images.

## Original dataset

Sunny Arya and Biswabiplab Singh (2020), *Wheat nitrogen deficiency and leaf rust image dataset*, Mendeley Data, version 1, DOI [10.17632/th422bg4yd.1](https://data.mendeley.com/datasets/th422bg4yd/1).

The original record distinguishes nitrogen-deficiency and wheat leaf-rust sub-datasets. It describes 368 diseased and 491 control images in the leaf-rust subset and a 70:15:15 train/test/validation division. These totals agree with the previously reported local inventories: diseased 258 + 55 + 55 = 368; control 343 + 74 + 74 = 491. This count agreement supports provenance consistency; it is not a byte-for-byte comparison with a freshly downloaded original archive.

The record describes IARI-field acquisition during 2019–20, an RGB Sony IMX363 camera, booting-stage leaves and Otsu-based background masking. Acquisition/segmentation differences are candidate explanations for source shift, not mechanisms established by this study.

## Disease-name equivalence

An ICAR-IARI publication, *Nilgiri Wheat News*, states on printed page 31 (PDF page 31) that leaf or brown rust is caused by *Puccinia triticina*:

https://iari.res.in/files/Publication/Nilgiri_Wheat_News/Nilgiri_Wheat_News_02012023.pdf

The Tamil Nadu Agricultural University wheat-disease reference also identifies the category as Leaf Rust / Brown Rust:

https://agritech.tnau.ac.in/crop_protection/crop_prot_crop%20diseases_cereals_wheat.html

## Mapping and limits

| Original subset/folder | Study label | Basis |
|---|---|---|
| Wheat leaf-rust subset / `diseased` | `Brown_rust` | Creator-described wheat leaf rust plus authoritative leaf-rust/Brown-rust terminology |
| Wheat leaf-rust subset / `control` | `Healthy` | Creator-described healthy control leaves |
| Nitrogen-deficiency subset | Excluded | Different stress, outside the rust classification task |

The folder name `diseased` alone would not establish this mapping. The conclusion depends on its belonging to the documented wheat leaf-rust subset, consistent with the recorded study provenance and counts. It does not establish pathogen confirmation, annotation correctness for every photo, or plant/field/session independence. Dataset-author confirmation or expert reannotation would strengthen image-level validity. No author confirmation was obtained in this check.
