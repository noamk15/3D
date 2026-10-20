# Préparation orale — Mini-SfM (Prof. Werman)

Pour chaque brique, tu trouveras dans l'ordre :

1. **Intuition** en français.
2. **À essayer d'abord** : une ou deux questions pour toi. La réponse est cachée dans un bloc repliable.
3. **Formalisation** en anglais : *Construction / Correctness / Runtime / Conclusion*, avec un pourcentage de confiance.
4. **Ce qu'on a mesuré** : les chiffres viennent des scripts de `experiments/`, rien n'est inventé.
5. **Questions que Werman pourrait poser**, avec une réponse courte.

Conventions du code : caméra `P = K[R | t]`, point caméra `X_c = R X + t`, centre `C = −Rᵀt`, contrainte épipolaire `x₂ᵀ F x₁ = 0`.

---

## Brique 1 — Matrice fondamentale, 8 points normalisé

### Intuition

Un point x₁ dans l'image 1 correspond à tout un **rayon** 3D. Vu depuis la caméra 2, ce rayon est une **droite** : la droite épipolaire `l₂ = F x₁`. Son correspondant x₂ est forcément dessus, d'où `x₂ᵀ F x₁ = 0`.

Toutes ces droites passent par un même point, l'**épipole** e₂, qui est l'image du centre de la caméra 1. C'est pour ça que F est de rang 2.

Cette équation est **linéaire en les 9 coefficients de F**. Chaque correspondance donne une ligne d'un système `A f = 0`, qu'on résout par SVD.

Problème : en pixels, une ligne de A contient à la fois des termes en `u·u' ≈ 10⁶` et un `1`. Le système est très mal conditionné. La **normalisation de Hartley** recentre et remet les points à l'échelle avant de résoudre, puis on « dé-normalise » le résultat.

### À essayer d'abord

1. Combien de degrés de liberté a F ? Pourquoi l'algorithme utilise 8 points, alors que le minimum théorique est 7 ?
2. Pourquoi la solution de `min ‖Af‖ avec ‖f‖ = 1` est-elle le dernier vecteur singulier droit de A ?

<details><summary>Réponses</summary>

1. 9 coefficients, moins 1 pour l'échelle (F et λF définissent la même contrainte), moins 1 pour `det F = 0`, soit **7 ddl**.
   - Le 8 points **ignore** la contrainte non linéaire `det F = 0`. Il reste un problème linéaire à 8 inconnues, d'où 8 points, et on impose le rang 2 après coup.
   - Le 7 points l'utilise : avec 7 équations, l'espace nul de A est de dimension 2, F = αF₁ + (1−α)F₂, et `det F = 0` donne un polynôme cubique en α, donc 1 ou 3 solutions.
2. ‖Af‖² = fᵀ(AᵀA)f est un quotient de Rayleigh. Sur la sphère unité, son minimum est la plus petite valeur propre de AᵀA, atteinte au vecteur propre associé, c'est-à-dire au dernier vecteur singulier droit de A.
</details>

### Formalisation

**Construction.** Given N ≥ 8 correspondences (x₁ᵢ, x₂ᵢ):
1. Compute similarities T₁, T₂: translate the centroid to 0 and scale so the mean distance to the origin is √2. Set x̂ = T x.
2. Build A ∈ ℝ^{N×9} with row `[u₂u₁, u₂v₁, u₂, v₂u₁, v₂v₁, v₂, u₁, v₁, 1]` (row-major vec of F).
3. Take f = last right singular vector of A. Reshape it into F̂.
4. Enforce rank 2: F̂ = U diag(σ₁, σ₂, σ₃) Vᵀ is replaced by U diag(σ₁, σ₂, 0) Vᵀ.
5. De-normalize: F = T₂ᵀ F̂ T₁.

**Correctness.**
- The epipolar constraint: x₁ back-projects to the ray C₁ + λ M₁⁻¹x₁. Its image in camera 2 is the line through e₂ = P₂C₁ and through the image of the point at infinity of the ray, so `l₂ = [e₂]ₓ H x₁ = F x₁`. Since x₂ lies on l₂, we get x₂ᵀ F x₁ = 0. With calibration, F = K₂⁻ᵀ [t]ₓ R K₁⁻¹.
- De-normalization: if x̂₂ᵀ F̂ x̂₁ = 0 with x̂ = Tx, then x₂ᵀ (T₂ᵀ F̂ T₁) x₁ = 0.
- Rank 2: F = [e₂]ₓ H and [e₂]ₓ has rank 2. Geometrically, all epipolar lines F x₁ must meet at e₂, which needs a non-trivial null vector, F e₁ = 0. By Eckart–Young, zeroing σ₃ gives the closest rank-2 matrix in Frobenius norm.
- Why normalize: the 8-point method minimizes an *algebraic* error ‖Af‖, whose weighting depends on the coordinate frame. In pixels, the columns of A differ in scale by ~10⁶, so A is badly conditioned and noise is amplified anisotropically. After normalization all columns are O(1). Hartley (1997) showed this makes the linear method competitive with iterative ones.

**Runtime.** Building A takes O(N). The SVD of an N×9 matrix takes O(N·9²) = O(N). Normalization and rank projection take O(N) and O(1). Total: **O(N)**.

**Conclusion.** The normalized 8-point algorithm returns a rank-2 F satisfying the epipolar constraint exactly for noise-free data, and is well conditioned with noise. *Confidence: 97%.*

### Ce qu'on a mesuré (`experiments/conditioning.py`, figure `figures/normalization.png`)

| Bruit σ (px) | Erreur de Sampson de F, normalisé | Erreur de Sampson de F, pixels bruts | Erreur de rotation, normalisé | Erreur de rotation, bruts |
|---|---|---|---|---|
| 0,5 | 0,138 px | 1,39 px | 0,21° | 0,60° |
| 1,0 | 0,276 px | 6,01 px | 0,42° | 2,05° |
| 4,0 | 1,10 px | 68,9 px | 1,63° | 28,2° |

- Le conditionnement σ₁/σ₈ de A passe de **≈ 1,6·10⁵** (pixels bruts) à **≈ 6** (normalisé).
- Sans bruit, `|x₂ᵀFx₁| < 10⁻⁸` et le rang vaut 2 (tests `tests/test_two_view.py`).

### Questions que Werman pourrait poser

- **« Pourquoi x₂ᵀFx₁ = 0 ? Que représente F géométriquement ? »** F envoie un point de l'image 1 sur sa droite épipolaire dans l'image 2. e₁ est le noyau de F (F e₁ = 0) et e₂ le noyau de Fᵀ. e₁ est l'image du centre de la caméra 2 dans l'image 1, et e₂ l'image du centre de la caméra 1 dans l'image 2.
- **« Pourquoi normaliser ? »** Pour le conditionnement : on mesure 1,6·10⁵ contre 6. L'erreur algébrique minimisée dépend du repère ; après normalisation, toutes les coordonnées pèsent pareil. À σ = 1 px, l'erreur de rotation passe de 2,05° à 0,42°.
- **« Pourquoi imposer le rang 2, et comment ? »** Sans ça, les droites épipolaires ne se coupent pas en un point et il n'y a pas d'épipole. On met σ₃ à 0 dans la SVD : c'est la projection de Frobenius la plus proche (Eckart–Young).
- **« Que se passe-t-il si tous les points sont coplanaires ? »** Voir la brique 3 / dégénérescences : l'espace nul de A devient de dimension 3, donc F n'est pas unique.
- **« Le 8 points minimise-t-il l'erreur géométrique ? »** Non, il minimise une erreur algébrique. Pour l'erreur géométrique, on utilise Sampson dans RANSAC, et le bundle adjustment à la fin.

---

## Brique 2 — Matrice essentielle, 4 solutions, cheirality

### Intuition

Si on connaît K, on peut travailler en coordonnées normalisées `y = K⁻¹x`. F devient alors **E = [t]ₓR**, qui ne dépend que du mouvement.

E a seulement 5 degrés de liberté : 3 pour la rotation, 2 pour la direction de t, puisque **l'échelle est perdue**.

À partir de E, la SVD donne **4 couples (R, t)** :
- deux signes possibles pour t ;
- deux rotations possibles : la « twisted pair », qui diffère d'un demi-tour autour de la ligne de base.

Une seule de ces 4 configurations met les points **devant les deux caméras**. C'est le test de **cheirality**.

### À essayer d'abord

1. Montre que E a deux valeurs singulières égales et une nulle. Indice : calcule E Eᵀ.
2. Pourquoi ne peut-on jamais retrouver la norme de t avec deux images ?

<details><summary>Réponses</summary>

1. E Eᵀ = [t]ₓ R Rᵀ [t]ₓᵀ = [t]ₓ[t]ₓᵀ = ‖t‖² I − t tᵀ. Ses valeurs propres sont ‖t‖² (deux fois, sur t^⊥) et 0 (sur t). Les valeurs singulières de E sont donc (‖t‖, ‖t‖, 0).
2. Si on remplace (X, t) par (sX, st), on a s(RX + t), qui donne les mêmes pixels après division par la profondeur. Plus généralement, toute similitude globale (7 ddl) laisse les images inchangées.
</details>

### Formalisation

**Construction.**
- E = K₂ᵀ F K₁, projected onto the essential manifold: E = U diag(1,1,0) Vᵀ.
- With det U = det V = +1 and W = [[0,−1,0],[1,0,0],[0,0,1]], the candidates are R ∈ {UWVᵀ, UWᵀVᵀ} and t = ±u₃ (third column of U).
- For each candidate, triangulate the matches and count the points with positive depth in both cameras. Keep the candidate with the largest count.

**Correctness.**
- [t]ₓ is skew-symmetric with t in its null space. Hartley–Zisserman (Result 9.18–9.19) shows that every factorization E = SR, with S skew-symmetric and R a rotation, is one of these four (S = U Z Uᵀ, with Z = [[0,1,0],[−1,0,0],[0,0,0]]).
- The two rotations differ by a 180° rotation about the baseline. Flipping the sign of t reverses the baseline direction: the reconstructed point then lies behind both cameras (HZ Fig. 9.12).
- Exactly one of the four configurations puts a generic point in front of both cameras. The other three put it behind one or both. Counting over all points makes this robust to noise.

**Runtime.** One 3×3 SVD, plus 4 triangulations of N points: O(N).

**Conclusion.** E determines the relative pose up to the scale of t, and cheirality removes the 4-fold ambiguity. *Confidence: 95%* (the SVD sign conventions are the delicate part; the test `test_decomposition_four_candidates_and_cheirality` checks them).

### Ce qu'on a mesuré

- Sans bruit, la bonne solution a 100 % des points devant les deux caméras, et l'erreur de rotation est inférieure à 10⁻⁶ degré.
- Sur les 51 paires vérifiées de fountain-P11, l'erreur médiane de pose relative est de **0,051° en rotation** et **0,13° sur la direction de t**.
- La baseline OpenCV (`findEssentialMat`, 5 points + RANSAC, sans ré-estimation sur les inliers) donne **0,23°** et **0,26°**.
- **Attention à la queue de distribution.** La médiane nous est favorable, mais notre méthode est moins bonne qu'OpenCV sur 10 paires sur 51. Elle dépasse 1° sur 8 paires, contre 3 pour OpenCV, et la moyenne est de 0,73° contre 0,36°.
  - Les mauvaises paires impliquent presque toutes la vue 10 : grande baseline, peu de matches (par exemple (3,10) avec 109 matches, (2,10) avec 46). Le pire cas est (5,10) : 13° contre 0,11°.
  - Hypothèse (non vérifiée par une expérience dédiée) : le 8 points estime F (7 ddl, sans utiliser K). Il est donc plus sensible au bruit et à une scène en partie plane (la façade) que le 5 points calibré (5 ddl), quand il y a peu de matches.
  - J'ai vérifié une corrélation avec la planarité : elle n'est que faible (fraction d'inliers d'une homographie, corrélation ≈ −0,3 avec le log de l'erreur).
  - Ce n'est pas bloquant pour le pipeline, qui choisit la paire initiale (5,6) avec beaucoup de matches, puis place les autres vues par PnP et non par paires.
- Notre version est plus lente : 61 ms contre 10 ms en médiane (OpenCV est en C++ et son 5 points demande moins d'itérations).
- Voir `experiments/out/fountain_summary.json`.

### Questions que Werman pourrait poser

- **« Différence entre E et F ? »**
  - F agit sur les pixels : 7 ddl, rang 2.
  - E agit sur les coordonnées normalisées et nécessite K : 5 ddl, valeurs singulières (σ, σ, 0).
  - Lien : E = K₂ᵀ F K₁.
- **« Pourquoi 4 solutions ? »** Il y a deux choix de signe pour t et deux rotations (UWVᵀ, UWᵀVᵀ). Ces dernières sont reliées par un demi-tour autour de la baseline.
- **« Pourquoi seulement une échelle ? Et l'ambiguïté projective / affine / métrique ? »**
  - Sans calibration, la reconstruction n'est définie qu'à une homographie 4×4 près (15 ddl) : c'est l'ambiguïté projective.
  - Connaître le plan à l'infini la réduit à l'ambiguïté affine (12 ddl).
  - Connaître K (la conique absolue) la réduit à l'ambiguïté métrique, une similitude à 7 ddl.
  - Nous connaissons K, donc notre reconstruction est métrique : correcte à une rotation, une translation et une échelle près.
- **« Pourquoi le 5 points ? »** E n'a que 5 ddl, donc des échantillons de 5 au lieu de 8. Dans RANSAC, ça divise énormément le nombre d'itérations : pour 50 % d'outliers et p = 0,99, il faut 146 itérations au lieu de 1177. Je ne l'ai pas implémenté (il faut les racines d'un polynôme de degré 10) ; la baseline OpenCV l'utilise.

---

## Brique 3 — Triangulation et cas dégénérés

### Intuition

Deux rayons, un dans chaque image, devraient se croiser au point 3D. Avec du bruit, ils ne se croisent pas exactement.

- **DLT** : on écrit x × (PX) = 0, ce qui donne deux équations linéaires par vue, et on résout par SVD. C'est rapide, mais l'erreur minimisée est algébrique.
- **Raffinement** : on minimise ensuite la vraie erreur de reprojection, en pixels, par Gauss-Newton, sur 3 inconnues par point.

Plus les deux caméras sont proches (petite baseline), plus les rayons sont presque parallèles, et plus la profondeur est mal déterminée.

### À essayer d'abord

Comment varie l'erreur de profondeur en fonction de la baseline b, de la profondeur Z et de la focale f ?

<details><summary>Réponse</summary>

Pour une stéréo rectifiée, Z = f b / d, où d est la disparité. Donc δZ ≈ Z² δd / (f b) : l'erreur est **∝ 1/b** et **∝ Z²**.
</details>

### Formalisation

**Construction.**
- For each view i, the equations are u (p₃·X) − p₁·X = 0 and v (p₃·X) − p₂·X = 0. Stack them into A X = 0 and take X = last right singular vector.
- Then run Gauss–Newton on Σᵢ ‖xᵢ − π(PᵢX)‖². It uses the analytic 2×3 Jacobian of the projection and solves a 3×3 normal system per point (vectorized over all points).

**Correctness.**
- Noise-free, X is in the null space of A, so DLT is exact (test: error below 10⁻⁸).
- The GN refinement minimizes the geometric (reprojection) cost, which gives the maximum-likelihood estimate under i.i.d. Gaussian pixel noise. A per-point descent check keeps a GN step only where it lowers that point's cost.

**Runtime.** O(N·V) for N points and V views. Each point costs a 2V×4 SVD plus a few 3×3 solves.

**Conclusion.** Linear initialization plus non-linear refinement gives the ML point estimate, given the cameras. *Confidence: 95%.*

### Ce qu'on a mesuré (`figures/baseline.png`, `figures/degeneracies.png`)

**Erreur 3D médiane en fonction de l'angle entre les deux caméras** (σ = 0,5 px, objet de taille 3 à distance 6) :

| Angle | 0,25° | 0,5° | 1° | 2° | 5° | 10° | 20° | 40° |
|---|---|---|---|---|---|---|---|---|
| Erreur médiane | 1,16 | 0,46 | 0,21 | 0,10 | 0,041 | 0,021 | 0,011 | 0,0067 |

On voit bien la loi en 1/b.

**Dégénérescences**, lues dans les valeurs singulières de A (sans bruit) :
- **Scène générale** : une seule valeur singulière nulle, donc F est unique.
- **Scène plane** : **trois** valeurs nulles. F n'est pas unique : toute F = [e₂]ₓH convient, où H est l'homographie induite par le plan.
- **Rotation pure** (t = 0) : **trois** valeurs nulles également. E = [0]ₓR = 0. Les images sont reliées par l'homographie K R K⁻¹, et la translation n'est pas observable, donc pas de profondeur.

### Questions que Werman pourrait poser

- **« Cas dégénérés ? »**
  - Points coplanaires : l'espace nul de A est de dimension 3. On le détecte en comparant un modèle d'homographie à un modèle F (GRIC, critère de Torr).
  - Rotation pure : E = 0.
  - Baseline trop petite : l'erreur explose en 1/b. C'est pourquoi le choix de la paire initiale impose un angle de triangulation médian d'au moins 3°.
- **« Pourquoi pas le point milieu des deux rayons ? »** Il n'est pas invariant projectivement, et ce n'est pas l'estimateur du maximum de vraisemblance. L'erreur de reprojection l'est, sous un bruit gaussien en pixels.
- **« Que se passe-t-il pour un point très loin ? »** Les rayons sont presque parallèles, donc l'angle de triangulation est petit et la profondeur incertaine. On filtre ces points avec un angle minimum de 1,5°.

---

## Brique 4 — Estimation robuste : RANSAC, MSAC, LO-RANSAC, erreur de Sampson

### Intuition

Dans les moindres carrés, l'influence d'un point (ψ(r) = r) **croît sans borne** avec son résidu : un seul faux match peut tirer F n'importe où. Leur breakdown point est **0**.

**RANSAC** inverse la logique. On tire un petit échantillon (8 points), on calcule F, puis on **compte qui est d'accord** avec elle. On garde la F qui a le plus large consensus, et on la ré-estime sur tous ses inliers.

Le nombre de tirages nécessaires dépend de la proportion w d'inliers et de la taille s de l'échantillon.

**MSAC** remplace le comptage par une perte quadratique tronquée : à nombre d'inliers égal, le modèle le plus « serré » gagne.

**LO-RANSAC** affine localement chaque nouveau meilleur modèle. Le modèle à 8 points est bruité ; le ré-estimer sur ses inliers donne un consensus meilleur.

L'**erreur de Sampson** est une approximation au premier ordre de la vraie distance géométrique (en pixels) d'une paire de points à la contrainte épipolaire. Le seuil a donc une unité physique : des pixels.

### À essayer d'abord

1. Dérive le nombre d'itérations N pour avoir, avec probabilité p, au moins un échantillon sans outlier.
2. Pourquoi l'erreur algébrique x₂ᵀFx₁ n'est-elle pas une bonne mesure pour décider qui est inlier ?

<details><summary>Réponses</summary>

1. Un échantillon de s points est entièrement inlier avec probabilité wˢ. La probabilité que N échantillons échouent tous est (1 − wˢ)ᴺ. On veut (1 − wˢ)ᴺ ≤ 1 − p, donc **N = log(1 − p) / log(1 − wˢ)**. Exemple : s = 8, w = 0,5, p = 0,99 donne 1177.
2. Elle n'a pas d'unité. Elle change si on multiplie F par λ, et pour une même distance en pixels elle varie selon la position du point dans l'image. On ne peut donc pas lui fixer un seuil.
</details>

### Formalisation

**Construction.**
- Repeat the following:
  1. Sample s = 8 matches and fit F (normalized 8-point).
  2. Compute the squared Sampson errors eᵢ².
  3. Score the model. RANSAC: #{eᵢ² < T²}. MSAC: −Σ min(eᵢ², T²). LO: MSAC, plus re-fitting each new best model on its inliers while the score improves (at most 5 times).
  4. When a new best model appears, set w = its inlier ratio and update N = log(1−p)/log(1−w⁸).
- Stop at N iterations (cap 20 000).
- Finally re-fit F on all inliers, re-classify, and repeat once.

The Sampson error is

  e² = (x₂ᵀFx₁)² / ((Fx₁)₁² + (Fx₁)₂² + (Fᵀx₂)₁² + (Fᵀx₂)₂²).

**Correctness.**
- The Sampson error is the first-order approximation of the geometric distance. The constraint c(x̃) = x₂ᵀFx₁, linearized at the measured 4-vector x̃, gives the distance |c| / ‖∇c‖, and ∇c is exactly the four entries in the denominator. The error is therefore invariant to the scale of F and measured in pixels.
- Threshold choice: under Gaussian noise σ, the distance to the epipolar line has 1 degree of freedom, so T² = χ²₁(0.95)·σ² = 3.84σ². For σ ≈ 0.5–0.8 px this gives T ≈ 1–1.5 px.
- The probabilistic guarantee follows from the derivation of N, assuming independent samples and a correct w.

**Runtime.** Each iteration costs O(N). The expected number of iterations is ≈ log(1−p)/log(1−w⁸), which grows like w⁻⁸ as w drops. That is the practical limit.

**Conclusion.** RANSAC turns a non-robust estimator (breakdown 0) into one that tolerates a majority of outliers, as long as the inliers form the largest consistent set and the iteration budget allows it. *Confidence: 93%.*

### Ce qu'on a mesuré (`experiments/robustness.py`)

Voir `figures/two_view_outliers_synthetic.png` et `figures/two_view_outliers_fountain.png`.

**Synthétique** (300 matches, σ = 0,5 px, 30 tirages). Chaque case donne l'erreur de rotation médiane, et entre parenthèses le taux de succès (erreur < 1°).

| Méthode | 0 % | 10 % | 30 % | 50 % | 60 % | 70 % |
|---|---|---|---|---|---|---|
| Moindres carrés | 0,11° (100 %) | 28,1° (0 %) | 49,1° (0 %) | 49,0° (0 %) | 54,4° (0 %) | 53,7° (0 %) |
| RANSAC | 0,12° (100 %) | 0,14° (100 %) | 0,15° (100 %) | 0,14° (100 %) | 0,23° (97 %) | 0,67° (57 %) |
| MSAC | 0,12° (100 %) | 0,14° (100 %) | 0,16° (100 %) | 0,14° (100 %) | 0,23° (97 %) | 0,75° (53 %) |
| LO-RANSAC | 0,12° (100 %) | 0,15° (100 %) | 0,13° (100 %) | 0,12° (100 %) | 0,19° (100 %) | 0,31° (70 %) |

**Réel** : fountain-P11, paire (4,5), 1000 vrais matches vérifiés, plus des faux matches tirés au hasard (15 tirages).

| Méthode | 0 % | 10 % | 30 % | 50 % | 60 % | 70 % |
|---|---|---|---|---|---|---|
| Moindres carrés | 0,05° (100 %) | 9,5° (0 %) | 13,2° (0 %) | 17,1° (0 %) | 52,3° (0 %) | 164° (0 %) |
| RANSAC | 0,05° (100 %) | 0,05° (100 %) | 0,05° (100 %) | 0,07° (100 %) | 0,08° (100 %) | 0,61° (60 %) |
| MSAC | 0,05° (100 %) | 0,05° (100 %) | 0,05° (100 %) | 0,07° (87 %) | 0,06° (100 %) | 1,42° (47 %) |
| LO-RANSAC | 0,05° (100 %) | 0,05° (100 %) | 0,05° (100 %) | 0,10° (93 %) | 0,08° (100 %) | 1,55° (40 %) |

Lecture des tableaux :
- **Moindres carrés** : ils cassent dès **10 %** d'outliers (28° d'erreur en synthétique). Breakdown point de 0, comme prévu.
- **Variantes RANSAC** : précision quasi constante jusqu'à 50–60 %.
- **À 70 %** : w = 0,3, donc N = log(0,001)/log(1 − 0,3⁸) ≈ **105 000** itérations seraient nécessaires. Notre plafond est 20 000 (atteint en médiane dès 60 % pour RANSAC et MSAC en synthétique ; LO en fait environ 10 200 à 60 %), donc le taux de succès chute. C'est le **point de rupture pratique**. Il ne vient pas de la statistique ; il vient du budget de calcul, qui explose en w⁻⁸.
- **Itérations RANSAC en médiane** (synthétique) : 50, 50, 98, 206, 768, 3208, 20 000, 20 000 pour 0 → 70 %. Le « 50 » vient d'un minimum de 50 itérations imposé dans le code (`min_iters`), par sécurité quand w est estimé à partir de très peu de tirages ; la formule seule en donnerait moins. Temps LO : 23 ms à 0 %, 0,47 s à 50 %, 5,6 s à 70 %.
- **LO-RANSAC** : il gagne nettement en synthétique à 70 % (70 % de succès contre 57 %). Sur la paire réelle, la différence entre variantes est dans le bruit (15 tirages seulement). Je ne prétends pas que LO soit toujours meilleur.

### Questions que Werman pourrait poser

- **« Pourquoi RANSAC plutôt que les moindres carrés ? »** Le breakdown point des moindres carrés est 0 : une fraction 1/n d'outliers suffit à les rendre arbitrairement faux. Mesuré : avec 10 % de faux matches, l'erreur de rotation passe à 28° (synthétique) et 9,5° (paire réelle).
- **« Quel est le breakdown point de RANSAC ? »** Il n'a pas de breakdown point formel à 50 % comme la médiane. Il fonctionne tant que les inliers forment le **plus grand ensemble cohérent** et que N tirages suffisent. En pratique, la limite est le coût : avec w = 0,3 et s = 8, il faut environ 105 000 itérations pour p = 0,999. D'où l'intérêt des solveurs minimaux (7 points, 5 points) et du *guided sampling*.
- **« Pourquoi Sampson plutôt que l'erreur algébrique ? »** Sampson est géométrique (en pixels), invariant à l'échelle de F, et donne un seuil interprétable via le χ².
- **« Comment choisis-tu le seuil ? »** Avec le χ² à 1 ddl (distance à la droite épipolaire) : T² = 3,84σ², soit T ≈ 1,96σ.
  - σ = 0,5 px donne T ≈ 1 px (utilisé pour la comparaison avec OpenCV). T = 1,5 px (matching, robustesse, paire initiale) correspond à σ ≈ 0,77 px, une marge prudente pour SIFT.
  - Pour PnP, le résidu est 2D (2 ddl), donc T² = 5,99σ². T = 3 px correspond à σ ≈ 1,2 px : c'est plus large exprès, car les points 3D sont eux-mêmes bruités.
- **« Pourquoi ré-estimer après RANSAC ? »** Le modèle ne vient que de 8 points, donc il est bruité. Les moindres carrés sur des centaines d'inliers réduisent la variance. LO-RANSAC fait ça pendant la recherche, ce qui améliore aussi le choix du consensus.
- **« Et les M-estimateurs ? »** Voir la brique 7 (Huber, Cauchy dans le BA). Ils supposent une bonne initialisation, alors que RANSAC n'en a pas besoin. Les deux sont complémentaires : RANSAC pour démarrer, une perte robuste pour raffiner.

---

## Brique 5 — PnP et SfM incrémental

### Intuition

Une fois quelques points 3D reconstruits, une **nouvelle image** se place en utilisant les points 3D qu'elle voit. Avec des correspondances 2D-3D, on cherche R et t telles que la projection retombe sur les pixels : c'est le problème **PnP**.

On procède en trois étapes :
1. Une version linéaire (DLT, au moins 6 points), dans un RANSAC.
2. Un raffinement Gauss-Newton directement sur SE(3).
3. On triangule les nouveaux points et on refait un bundle adjustment.

### À essayer d'abord

Combien de correspondances 2D-3D faut-il au minimum, et pourquoi le DLT en demande-t-il plus ?

<details><summary>Réponse</summary>

La pose a 6 ddl et chaque point donne 2 équations, donc **3 points** au minimum (P3P, jusqu'à 4 solutions). Le DLT estime les 12 entrées de [R|t] (11 ddl à l'échelle près) sans imposer RᵀR = I. Il faut donc ⌈11/2⌉ = **6 points**, et on projette ensuite sur SO(3).
</details>

### Formalisation

**Construction.**
- Normalized coordinates y = K⁻¹x. Each point gives two rows of A m = 0, with m = vec([R|t]) (3D points pre-normalized).
- SVD, then fix the sign so that det(M₃ₓ₃) > 0. M₃ₓ₃ = U S Vᵀ gives R = UVᵀ and t = m₄ / mean(S).
- RANSAC (MSAC, 6-point samples, squared reprojection error, plus a cheirality check).
- Gauss–Newton on the inliers with left updates R ← exp([δω]ₓ)R, t ← t + δt.

**Incremental loop** (`sfm/incremental.py`):
1. Build tracks (union-find over the verified matches).
2. Choose the initial pair: largest number of inliers, with a median triangulation angle of at least 3°.
3. Repeat: pick the view that sees the most already-reconstructed tracks, run PnP-RANSAC, triangulate the new tracks (n-view DLT + GN, angle ≥ 1.5°, reprojection < 4 px), run a Huber BA, and remove observations above 4 px.

**Correctness.** In the noise-free case the DLT is exact (test: error below 10⁻⁶). The SO(3) projection is the Frobenius-closest rotation.

**Runtime.**
- PnP: O(n) per RANSAC iteration.
- The incremental loop runs one BA per added view. This is the cost of incremental SfM: roughly O(V) BAs.

**Conclusion.** *Confidence: 90%.* The pipeline is standard and tested on synthetic and real data, but the heuristics (thresholds, view order) are choices, not theorems.

### Ce qu'on a mesuré

- **Synthétique** (6 vues, σ = 0,5 px) : toutes les vues sont enregistrées. Le test vérifie que l'erreur de position des centres après alignement est < 0,05 unités et l'erreur de rotation relative < 0,3° (bornes d'assertion, pas des valeurs mesurées).
- **fountain-P11** (images à 1536×1024, SIFT : 12 600 à 15 600 points par image) :
  - 51 paires vérifiées, les **11 vues** enregistrées, **28 687 points**, 101 601 observations, **RMS de reprojection 0,247 px**.
  - Par rapport à la vérité terrain, après alignement par similitude : erreur de rotation médiane **0,037°** (max 0,052°), erreur de position des centres médiane **0,03 %** de la taille de la scène (max 0,05 %).
  - Temps : environ 50–70 s pour le SfM selon la charge de la machine, 174 s pour SIFT + matching de toutes les paires.
  - Initialisation sur la paire (5,6), puis ordre d'ajout 4, 3, 2, 1, 7, 0, 8, 9, 10. PnP garde chaque fois 97,6–99,2 % d'inliers.

### Questions que Werman pourrait poser

- **« Pourquoi choisir la paire initiale avec une grande baseline ? »** Une petite baseline donne une triangulation instable (brique 3), et toute l'erreur se propage aux vues suivantes.
- **« Que se passe-t-il si une vue est mal placée ? »** L'erreur se propage : c'est la **dérive** de l'incrémental. On la limite par des BA fréquents et des pertes robustes. Une alternative est le SfM global (rotation averaging), en extension.
- **« Pourquoi ne pas utiliser OpenCV solvePnP ? »** Le cœur doit être écrit à la main. OpenCV ne sert que de baseline de comparaison.

---

## Brique 6 — Données réelles : SIFT, ratio test, vérification géométrique

### Intuition

SIFT trouve des points distinctifs et les décrit par un vecteur de 128 nombres (histogrammes de gradients), invariant à l'échelle et à la rotation. Pour apparier deux images, on cherche le plus proche voisin de chaque descripteur.

Problème : sur une façade répétitive, un descripteur peut avoir **deux** voisins presque aussi proches. Le match est alors ambigu.

On applique trois filtres :
- Le **ratio test de Lowe** garde un match seulement si le meilleur voisin est nettement meilleur que le deuxième (d₁/d₂ < 0,8).
- Une **vérification croisée** : le match doit être réciproque.
- **Notre F-RANSAC** élimine ce qui reste d'incohérent avec la géométrie épipolaire.

### Questions que Werman pourrait poser

- **« Pourquoi 0,8 ? »** C'est la valeur de Lowe (2004) : elle élimine environ 90 % des faux matches pour environ 5 % des vrais perdus, sur sa base de test. Ce n'est pas un théorème : c'est un réglage empirique.
- **« Le ratio test suffit-il ? »** Non. Il élimine les ambiguïtés, pas les erreurs cohérentes visuellement mais fausses géométriquement. C'est RANSAC qui les élimine.
- **« Pourquoi FLANN ? »** Avec environ 15 000 descripteurs par image et 55 paires, la force brute prenait plus de 25 minutes. Les kd-trees randomisés (approximatifs) ramènent ça à environ 3 minutes. Une approximation sur quelques voisins est sans conséquence, puisque RANSAC filtre derrière.

---

## Brique 7 — Bundle adjustment

### Intuition

À la fin, on a des caméras et des points 3D estimés **séparément**, chacun avec ses erreurs. Le BA ajuste **tout en même temps** pour minimiser l'erreur de reprojection totale, en pixels.

C'est non linéaire pour deux raisons : la division par la profondeur, et la rotation.

- **Levenberg-Marquardt** se comporte comme Gauss-Newton quand tout va bien, et comme une descente de gradient prudente quand l'approximation linéaire est mauvaise (paramètre λ).
- **Structure creuse** : chaque observation ne dépend que d'**une** caméra et d'**un** point. Le bloc « points » du système est donc diagonal par blocs 3×3. On élimine les points (**complément de Schur**) et on résout un petit système sur les caméras seulement.

### À essayer d'abord

1. Combien de degrés de liberté de jauge a le BA ? Que se passe-t-il si on n'en fixe aucun ?
2. Pourquoi paramétrer les mises à jour de rotation par un vecteur de so(3) plutôt que par les 9 entrées de R ou par des angles d'Euler ?

<details><summary>Réponses</summary>

1. **7** ddl : 3 de rotation, 3 de translation, 1 d'échelle. Une similitude globale ne change aucune reprojection. Sans jauge fixée, JᵀJ est singulière (noyau de dimension 7) et le solveur dérive dans ces directions. On fixe une caméra (6 ddl) et une coordonnée de translation d'une autre caméra (l'échelle). Dans le pipeline, la caméra fixée est la première de la paire initiale (la vue 5 sur fountain).
2. Avec 9 entrées, il y a 6 contraintes RᵀR = I à maintenir, et R sort de SO(3). Les angles d'Euler ont des singularités (blocage de cardan). L'incrément δω ∈ ℝ³ est minimal, sans contrainte, et R ← exp([δω]ₓ)R reste exactement dans SO(3). Localement, la paramétrisation est régulière.
</details>

### Formalisation

**Construction.**
- Unknowns: for each camera, a 6-vector (δω, δt); for each point, a 3-vector.
- For each observation (c, p), the residual is rₒ = uₒ − π(K(R_c X_p + t_c)), with Jacobian blocks
  - J_c = J_π · [−[R_c X_p]ₓ | I] (2×6),
  - J_p = J_π · R_c (2×3).
- Normal equations [[U, W],[Wᵀ, V]] [δc; δp] = [g_c; g_p], with
  - U = blockdiag over cameras (6×6),
  - V = blockdiag over points (3×3),
  - W = cross terms.
- LM damping: U, V ← U + λ diag(U), V + λ diag(V).
- **Schur complement**: S = U − W V⁻¹ Wᵀ (6n_c × 6n_c). Solve S δc = g_c − W V⁻¹ g_p, then back-substitute δp = V⁻¹(g_p − Wᵀ δc).
- Accept the step if the cost decreases (λ /= 3), otherwise reject it (λ *= 5).
- Robust losses through IRLS: each residual gets a weight w(‖r‖). Huber: w = min(1, c/‖r‖). Cauchy: w = 1/(1 + ‖r‖²/c²).

**Correctness.**
- Block elimination is exact: the Schur step equals the step of the full system. This is checked numerically by `test_schur_step_equals_full_step`.
- LM with accept-if-decrease control (λ /= 3 on success, λ *= 5 on failure) decreases the cost monotonically by construction.
- IRLS is a valid majorize-minimize scheme for the Huber and Cauchy costs: each weighted least-squares step decreases the robust cost.
- The analytic Jacobians are checked against central finite differences (`test_projection_jacobians_finite_differences`).

**Runtime.**
- Building the blocks: O(M) for M observations.
- Inverting V: O(n_p) (one 3×3 per point).
- Forming S: as implemented, each point's block W_p is stored densely (6n_c × 3), so this costs O(n_p·n_c²). A fully sparse implementation would cost O(Σ_p m_p²), where m_p = track length.
- Solving: O(n_c³).
- Total: **O(M + n_p·n_c² + n_c³)**, still linear in the number of points, compared with O((6n_c + 3n_p)³) for the full system.

**Conclusion.** BA computes the maximum-likelihood estimate of all cameras and points under Gaussian pixel noise (or a robust M-estimate with Huber/Cauchy). The Schur trick makes it linear in the number of points. *Confidence: 92%.*

### Ce qu'on a mesuré

**Convergence sur fountain-P11** (`figures/ba_convergence_fountain.png`).
- Problème : 11 caméras, 28 687 points, 101 601 observations, soit 86 127 inconnues.
- Départ : rotation de chaque caméra perturbée d'environ 1°. Bruit gaussien d'écart-type 2 % de la taille de la scène par axe sur les translations, et 1 % par axe sur les points (taille = distance moyenne au centroïde). RMS initial 32,5 px.
- Notre LM + Schur : 32,5, puis 0,76, 0,53, 0,47, 0,38, 0,29, 0,250, 0,2471 px. Convergence en **10 itérations, environ 5 s**.
- scipy `least_squares` (TRF, Jacobien creux par différences finies) : même minimum (0,2471 px) en 11–25 s selon la charge de la machine.

**Coût du Schur** (`figures/schur_scaling.png`, 10 caméras).

| Points | Inconnues | Résolution dense du système complet | Notre itération LM complète avec Schur |
|---|---|---|---|
| 400 | 1 260 | 0,036 s | 0,015 s |
| 800 | 2 460 | 0,19 s | 0,029 s |
| 1 600 | 4 860 | 1,01 s | 0,060 s |
| 3 200 | 9 660 | (non lancé) | 0,11 s |

Le dense croît en n³, le Schur environ linéairement. Pour fountain (86 127 inconnues), un solveur dense est hors de portée : sa matrice ferait à elle seule 59 Go.

**Pertes robustes dans le BA** (`figures/ba_robust_losses.png`).
- Setup : 8 caméras, 250 points, σ = 0,5 px. Une fraction des observations reçoit une erreur grossière ~N(0, 40 px). 10 tirages. Pour chaque tirage, on prend la moyenne de l'erreur de rotation sur les caméras ; le tableau donne la médiane sur les 10 tirages.

| Observations corrompues | 0 % | 5 % | 10 % | 20 % | 30 % | 50 % |
|---|---|---|---|---|---|---|
| L2 | 0,054° | 1,06° | 1,06° | 1,25° | 1,83° | 2,94° |
| Huber (c = 2 px) | 0,054° | 0,058° | 0,070° | 0,094° | 0,15° | 0,36° |
| Cauchy (c = 2 px) | 0,052° | 0,046° | 0,045° | 0,055° | 0,057° | 0,079° |

Lecture :
- **L2** : 5 % d'outliers suffisent à multiplier l'erreur par 20.
- **Huber** : il borne l'influence, mais chaque outlier pèse encore c/‖r‖. L'erreur augmente donc doucement avec le taux.
- **Cauchy** : il est redescendant, et reste quasi insensible jusqu'à 50 % (ici avec une bonne initialisation).
- Le RMS sur les observations *propres* monte quand même à partir de 20 %. Les points dont la plupart des observations sont corrompues (par exemple 6 sur 8) sont tirés vers les outliers : c'est un breakdown **par point**, pas global.

### Questions que Werman pourrait poser

- **« Pourquoi le BA est-il non linéaire ? »** Il y a la division perspective π(X) = (X/Z, Y/Z), et la rotation (variété SO(3)).
- **« Pourquoi LM plutôt que Gauss-Newton ? »**
  - GN peut diverger si l'approximation linéaire est mauvaise, ou si JᵀJ est mal conditionnée (jauge, points lointains).
  - LM ajoute λ diag(JᵀJ) : c'est une région de confiance implicite, robuste loin du minimum. Près du minimum il se comporte comme Gauss-Newton : convergence quadratique si les résidus sont nuls au minimum, seulement linéaire ou superlinéaire sinon.
- **« Pourquoi la structure creuse accélère-t-elle ? »** V est diagonale par blocs 3×3, donc V⁻¹ coûte O(n_p). On résout ensuite un système de taille 6n_c au lieu de 6n_c + 3n_p. Mesuré : voir `figures/schur_scaling.png`.
- **« Jauge ? »** 7 ddl (similitude). On fixe une caméra et une coordonnée de translation d'une autre caméra. Sinon JᵀJ a un noyau de dimension 7.
- **« Huber ou Cauchy ? »**
  - Huber est convexe : son influence est bornée mais jamais nulle, donc les outliers biaisent encore un peu.
  - Cauchy est redescendante (l'influence tend vers 0) : les outliers sont quasi ignorés, mais la fonction est non convexe et a besoin d'une bonne initialisation.
  - Mesuré : voir `figures/ba_robust_losses.png`.
- **« Pourquoi so(3) ? »** Voir « À essayer d'abord ». Le Jacobien de la mise à jour à gauche est simplement −[RX]ₓ.

---

## Brique 8 — Viewer 3D

Le viewer (`viewer/`, Three.js) affiche :
- le nuage coloré (couleur du pixel SIFT dans la première vue qui observe le point) ;
- les frustums estimés (orange) et vérité terrain (bleu) ;
- la trajectoire des caméras.

Le bouton « View through camera » place la caméra virtuelle exactement au centre C = −Rᵀt, regardant selon la 3e ligne de R. C'est une vérification visuelle directe de la pose.

Le tout est exprimé dans le repère de la vérité terrain après alignement par similitude (Umeyama). C'est nécessaire, car notre reconstruction n'est définie qu'à une similitude près (brique 2).

---

## Brique 9 — Synthèse : le « point de rupture »

| Estimateur | Breakdown point (théorique) | Ce qu'on observe |
|---|---|---|
| Moindres carrés (8 points sur tout) | 0 | Faux dès 10 % d'outliers |
| RANSAC / MSAC / LO | Pas de borne à 50 % : marche tant que les inliers forment le plus grand consensus | Stable jusqu'à 60 %. À 70 %, le budget de 20 000 itérations ne suffit plus (il en faudrait ~10⁵) |
| BA L2 | 0 | 5 % d'observations grossières, erreur ×20 |
| BA Huber | Influence bornée mais non nulle, convexe | Dégradation progressive |
| BA Cauchy | Redescendant, non convexe | Quasi insensible jusqu'à 50 % si l'initialisation est bonne |

**Message pour l'oral :** RANSAC pour **trouver** le bon modèle sans initialisation (combinatoire), puis M-estimateur pour **raffiner** (continu). C'est exactement l'ordre du pipeline : RANSAC sur F et sur PnP, puis BA avec Huber.

---

## Plan de démo (3–5 minutes)

1. **(30 s) Le problème.** Montrer 3 images de fountain-P11. Question : « à partir de ces photos seulement, peut-on retrouver la scène 3D et où étaient les caméras ? »
2. **(60 s) Le pipeline**, au tableau ou sur une slide : SIFT, puis F (8 points normalisé + RANSAC/Sampson), puis E, puis (R, t) avec cheirality, puis triangulation, puis PnP pour chaque nouvelle vue, puis bundle adjustment sur SO(3) avec Schur. Dire que le cœur est écrit à la main (NumPy).
3. **(60 s) Viewer 3D.** Ouvrir le viewer : nuage coloré et frustums. Cliquer sur une caméra pour « regarder à travers elle ». Activer les caméras vérité terrain : elles se superposent. Annoncer les chiffres : 11/11 vues, 28 687 points, RMS 0,25 px, erreur de rotation médiane 0,037° contre la vérité terrain.
4. **(60 s) Robustesse.** Montrer la figure « erreur vs taux d'outliers » : les moindres carrés cassent dès 10 %, RANSAC/MSAC/LO tiennent jusqu'à 60 % ; à 70 %, le budget d'itérations casse (40–70 % de succès seulement). Puis la figure du BA : L2 contre Huber contre Cauchy.
5. **(30 s) Limites et suite.** Dérive de l'incrémental, donc rotation averaging. Le 5 points pour réduire le nombre d'itérations. La densification (MVS / NeRF à partir de nos poses).

Questions probables juste après : voir les sections « Questions que Werman pourrait poser » ci-dessus.
