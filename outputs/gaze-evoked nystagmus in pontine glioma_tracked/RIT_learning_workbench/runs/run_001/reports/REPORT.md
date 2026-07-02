# RIT Learning Workbench -- run_001

Active path: RIT Orbit Lock + clinician ground truth + learned pixel segmenter. OpenCV rule detector is DEPRECATED (historical benchmark only).

## 1. Ground-truth validation
- units found: **52**, valid: **52** (L 26 / R 26)
- excluded: 0
- iris/sclera disjoint: True (max overlap 0.01); almond = by construction (union of iris+sclera) unless GREEN outline supplied

## 2. Model
- **ExtraTreesClassifier**, 300 trees, max_depth 18, class_weight balanced; 5-fold held-out-FRAME CV
- features (16): B, G, R, H, S, V, L*, a*, b*, gray, blur_s, blur_l, |lap|, dx, dy, radius
- top features: dy=0.1558, dx=0.1384, radius=0.0756, R=0.0644, blur_l=0.0643, L*=0.0603

## 3. Held-out validation (learned)
- iris IoU **0.514**, sclera IoU **0.604**, almond IoU **0.661**
- iris centre error **34.44 px**, iris-outside-almond units 52, pass 11/52
  - L: iris 0.462 sclera 0.565 almond 0.625 ctrErr 39.931px pass 1/26
  - R: iris 0.566 sclera 0.643 almond 0.698 ctrErr 28.95px pass 10/26

## 5. Full-clip application
- records: 606 (303 frames x 2 eyes)
- status: {'learned_ok': 606}
- mean confidence: 0.672
- overlay: `outputs\gaze-evoked nystagmus in pontine glioma_tracked\RIT_learning_workbench\runs\run_001\overlays\iris_sclera_overlay.mp4`

## 6. Worst frames selected for correction

- correction set: `outputs\gaze-evoked nystagmus in pontine glioma_tracked\RIT_learning_workbench\runs\run_001\correction_set` (0 units, paint red/blue/green then re-run for run_002)