# Uncovering Shortcut Learning in ISIC Dermoscopic Classifiers
Evaluates shortcut learning and artifact reliance (ink marks, rulers, hair) in ISIC-trained skin lesion classifiers to measure real-world diagnostic robustness.

---

## The problem

A skin lesion classifier reports 90% accuracy. That tells you it separates malignant from benign images in your test set. It doesn't tell you *what it used* to do it.

Dermoscopic images contain things that aren't skin: ruler markers, surgical ink, hair, lens vignetting, colored calibration patches. None are diagnostic. None are randomly distributed either — a clinician inks a lesion because they're about to biopsy it, and biopsied lesions skew malignant. Artifact and label become linked through clinical workflow rather than biology.

A network doesn't know which correlations are meaningful. If an artifact predicts the label, gradient descent will find it. And standard evaluation cannot catch this: when the test set carries artifacts at the same rate as training, the shortcut works there too and accuracy reports it as success.

This project asks a narrower, answerable question about one specific model:

> **Does a classifier trained on public ISIC data actually depend on the lesion — and how much of its decision rests on everything else in the frame?**

## Approach

Observation isn't enough. Saliency heatmaps are pictures, and pictures invite confirmation bias — Adebayo et al. (2018) found several attribution methods produce plausible-looking maps even under randomized model weights.

So the method is interventional. Destroy a region, re-run inference, measure the shift:

```
delta = p(original) − p(modified)
```

Two arms on every test image:

| Arm | What survives | What it measures |
|---|---|---|
| **Lesion masked** | background only | dependence on the diagnostic region |
| **Background masked** | lesion only | dependence on everything else |

A clinically grounded model should collapse when the lesion is removed and shrug when the background goes. Comparable deltas mean it's reading something else.

Masked regions are filled with Gaussian blur rather than a solid color. A black rectangle is out-of-distribution input, so a confidence drop would be ambiguous between "lost needed information" and "confused by a bizarre image." Blur preserves local color statistics while destroying structure. Three fill modes are implemented so the conclusion can be checked for stability across them.

## Findings so far

**Artifact–label correlations in ISIC 2018 are weak, and several run backwards.** Base malignancy rate is 20.0% across 2,594 annotated images.

| Artifact | n | Malignant | Lift |
|---|---:|---:|---:|
| dark_corner | 960 | 24.4% | +4.4 |
| ruler | 1284 | 24.1% | +4.1 |
| gel_bubble | 1267 | 20.3% | +0.3 |
| hair | 1518 | 17.4% | −2.6 |
| ink | 442 | 14.3% | −5.8 |
| gel_border | 679 | 13.1% | −6.9 |
| patches | 187 | 1.1% | −18.9 |

Chi-square tests of independence accompany each row in the notebook output. Seven artifacts
tested at α = 0.05 yields roughly one expected false positive, so Bonferroni-corrected
thresholds are reported alongside uncorrected ones.

**Ink is anti-correlated with malignancy here.** This contradicts the framing the project started with. Winkler et al. (2019) demonstrated that surgical markings degrade a market-approved commercial classifier — but that finding does not transfer to ISIC, whose images come from a different clinical pipeline. The hypothesis was revised accordingly: the question is no longer "does the model key off ink," but whether a model exploits artifact correlations *even when they are individually weak and sometimes point away from malignancy*.

Dark corners and rulers are the positive-lift candidates. Patches, at 1.1% malignant, is the strongest single association in the dataset — and is the confound Nauta et al. (2021) built their study around.

## Data

| Source | Role | Access |
|---|---|---|
| ISIC 2018 Task 1–2 (2,594 images) | evaluation set | [challenge site](https://challenge.isic-archive.com/data/#2018) |
| ISIC 2018 Task 1 ground truth | lesion segmentation masks | same |
| `alceubissoto/debiasing-skin` | artifact annotations | [GitHub](https://github.com/alceubissoto/debiasing-skin) |

## Running it

Open `shortcut_learning_setup.ipynb` in Colab. Runtime → Change runtime type → **T4 GPU**.

Section 1 and 3 runs immediately with no download — it clones the annotations and regenerates the correlation table.

Section 2 requires manual download of the two ISIC 2018 zips into `MyDrive/shortcut-learning/raw/`.

## Design notes

**Evaluation transforms are deterministic.** No augmentation at inference. The delta compares two predictions on the same image; stochastic transforms would inject noise indistinguishable from the masking effect.

**Class weighting.** At 20% positives, an all-benign predictor scores 80% accuracy. `pos_weight ≈ 4` in the loss prevents that local minimum. AUC is reported rather than accuracy for the same reason.

**Nearest-neighbor mask resizing.** Masks are binary; bilinear interpolation produces intermediate values at boundaries that threshold into edge artifacts.

**Wilcoxon rather than paired t-test.** Delta distributions are bounded in [−1, 1] and skewed, so normality shouldn't be assumed.

**Grad-CAM is illustration, not evidence.** It generates figures. The argument rests on the delta measurements.

## Positioning

| Prior work | Contribution | Gap |
|---|---|---|
| Winkler et al. (2019, 2021) | clinical demonstration that markings degrade a classifier | single proprietary model, manipulated test set |
| Bissoto et al. (2019–2023) | artifact annotations, trap sets, debiasing evaluation | correlations modest; debiasing methods largely fail to beat ERM |
| Nauta et al. (2021) | inpainting-based quantification of shortcut reliance on ISIC | measures artifact sensitivity in absolute terms |

The closest precedent is Nauta et al. The contribution here is the *relative* comparison — artifact/background dependence measured against lesion dependence on the same images — which converts an absolute sensitivity into an interpretable ratio, and yields an audit other researchers can run on their own models with only segmentation masks.

## Selected references

Adebayo, J., et al. (2018). Sanity checks for saliency maps. *NeurIPS*.

Bissoto, A., Fornaciali, M., Valle, E., & Avila, S. (2019). (De)Constructing bias on skin lesion datasets. *CVPRW*.

Bissoto, A., Barata, C., Valle, E., & Avila, S. (2022). Artifact-based domain generalization of skin lesion models. *ECCV Workshops*.

Cassidy, B., et al. (2022). Analysis of the ISIC image datasets: Usage, benchmarks and recommendations. *Medical Image Analysis*, 75, 102305.

Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665–673.

Nauta, M., Walsh, R., Dubowski, A., & Seifert, C. (2021). Uncovering and correcting shortcut learning in machine learning models for skin cancer diagnosis. *Diagnostics*, 12(1), 40.

Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.

Winkler, J. K., et al. (2019). Association between surgical skin markings in dermoscopic images and diagnostic performance of a deep learning CNN for melanoma recognition. *JAMA Dermatology*, 155(10), 1135–1141.

---

Research conducted under the Lumiere Research Scholar Program (Individual Research Program).
