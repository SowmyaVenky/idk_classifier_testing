# IDK Classifier Iterations

### Run the IDK classifier as specified in Ronit's paper. (paper_imagenet_idk_cascade.py)
* This sets up the resnet classifiers in a cascade just like the paper has documented. There is no skipping optimization applied. The images flow from resnet-18, resnet-34 and resnet-152. Here are the results of the run.

<pre>
Top-1 Accuracy: 0.6673
Processed by ResNet18 only:  63.71%
Processed by ResNet34:       3.24%
Processed by ResNet152:      33.05%
Total evaluation time: 771.79 seconds
</pre>

### Run the IDK classifier as specified in Ronit's paper with the Random Forest Classifier. (train_random_forest_regulators.py)
* This sets up the resnet classifiers in a cascade just like the paper has documented. There is a Random Forest skipping optimization applied. The images flow from resnet-18. Based on the RandomForest regulator, the skip of resnet-34 can happen to delegate to  resnet-152. Here are the results of the run.

<pre>
Extracting features from calibration data...
Training Random Forest 1 (ResNet18 regulator)...
Training Random Forest 2 (ResNet34 regulator)...
Total Random Forest Training time: 88.93 seconds

Final Cascade Top-1 Accuracy: 0.6038
Skipped to Stage 2 (ResNet34):  34.09%
Skipped to Stage 3 (ResNet152): 14.24%
Total evaluation time: 429.86 seconds
</pre>

### Run the IDK classifier as specified in Ronit's paper with the light weight gatekeeper Classifier. (train_gatekeeper_3_classifier.py)
* This sets up the resnet classifiers in a cascade just like the paper has documented. There is a gatekeeper skipping optimization applied. The images flow from resnet-18. Based on the gatekeeper regulator, the skip of resnet-34 can happen to delegate to  resnet-152. Here are the results of the run.

<pre>
Extracting gatekeeper features...

Gatekeeper Cascade Top-1 Accuracy: 0.6764
Skipped to ResNet34:  47.52%
Skipped to ResNet152: 34.79%
Total evaluation time: 570.56 seconds
</pre>

### Run the IDK classifier as specified in Ronit's paper with the light weight gatekeeper Classifier all 5 resnet stages are used. (train_gatekeeper_allresnet.py)
* This sets up all resnet classifiers in a cascade adding 2 more resnet classifiers than paper has documented. There is a gatekeeper skipping optimization applied. The images flow from resnet-18. Based on the gatekeeper regulator, the skip of higher level classifiers can happen to delegate to  resnet-152. Here are the results of the run.

<pre>

</pre>