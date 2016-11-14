======Work Process=====
1. Preprocess.py subject_id gesture_id image_id
2. handSizeDataCollector.py (no need if there's already hand_size.txt)
3. handDetector.py  -> generates cropped candidate images
4. featureExtractor.py  -> generates features for each image
5. AdaBoost.py  (no need if adaboost_model.p ??? already exists)


======Work Flow=====

1. Preprocess image to remove background
2. HandDetector -> generate collection of possible image cuts
3. for each candidate image, use adaboostClassifier to determine if T
4. merge set of slices labeled T