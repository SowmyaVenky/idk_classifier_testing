# IDK Classifier Iterations

## Testing # 1 : Run the IDK classifier as specified in Ronit's paper. (paper_imagenet_idk_cascade.py)
* This sets up the resnet classifiers in a cascade just like the paper has documented. There is no skipping optimization applied. The images flow from resnet-18, resnet-34 and resnet-152. Here are the results of the run.

<pre>
Top-1 Accuracy: 0.6673
Processed by ResNet18 only:  63.71%
Processed by ResNet34:       3.24%
Processed by ResNet152:      33.05%
Total evaluation time: 771.79 seconds
</pre>

---

## Testing # 2 : Run the IDK classifier as specified in Ronit's paper with the Random Forest Classifier. (train_random_forest_regulators.py)
* This sets up the resnet classifiers in a cascade just like the paper has documented. There is a Random Forest skipping optimization applied. The images flow from resnet-18. Based on the RandomForest regulator, the skip of resnet-34 can happen to delegate to  resnet-152. Here are the results of the run.

This code implements a highly efficient machine learning pattern often called **Dynamic Routing** or **Cascading Models**.

Instead of passing every single image through a massive, computationally expensive model (like ResNet152), this script sets up a "triage" system. It tries to solve the problem with a fast, lightweight model first. If it's confident, it stops there. If it's unsure, it escalates the problem to a slightly larger model, and only uses the heaviest model as a last resort.

Here is a step-by-step breakdown of how the code actually achieves this.

### 1. Training the "Random Forest Classifier" (`train_routing_classifiers`)

Before the cascade can run, it needs to know *when* to trust the smaller models. It does this by training **Random Forest Classifiers** to act as gatekeepers.

* **Feature Extraction:** The script passes a calibration dataset through both ResNet18 (small) and ResNet34 (medium). It saves the raw output scores (**logits**) and checks if the model's prediction matched the actual label (1 for Correct, 0 for Incorrect).
* **Training the Forests:** It trains one Random Forest on ResNet18's outputs and another on ResNet34's outputs.
* **The Goal:** These Random Forests are learning the "shape" of a correct prediction versus an incorrect one. (e.g., "If the highest probability is only slightly higher than the second-highest, the ResNet is probably guessing, so I will output a 0 for 'Incorrect'").

### 2. The Core Logic (`RandomForestResNetCascade`)

This is the custom PyTorch module that handles the live inference. It uses **boolean masking** to efficiently filter a batch of images through the stages without losing track of their original order.

* **Stage 1 (ResNet18):** * The whole batch goes through ResNet18.
* The `rf_stage1` (Random Forest) evaluates the outputs and predicts which ones are likely wrong (`0`).
* A mask (`to_s2_mask`) is created for all the images flagged as `0`. If everything was predicted correctly, the function returns early, saving massive compute time.


* **Stage 2 (ResNet34):** * Only the "hard" images—the ones masked `True` in `to_s2_mask`—are passed to ResNet34.
* Again, `rf_stage2` evaluates these new predictions. It creates a sub-mask (`to_s3_sub_mask`) for the images ResNet34 likely got wrong.
* The script patches ResNet34's answers back into the `final_outputs` tensor so the batch remains in the correct order.


* **Stage 3 (ResNet152):** * The absolute hardest images—the ones that failed both previous checks—are passed to the massive ResNet152.
* Because this is the end of the line, ResNet152's answers are accepted as the final say and patched into the output tensor.



### 3. The Execution Loop (`__main__`)

This section proves the concept works by running it on standard data.

1. **Prep:** It loads the ImageNetV2 dataset and splits it into a calibration set (for the Random Forests) and an evaluation set (to test the final system).
2. **Train:** It calls the function from Step 1 to train the Random Forests.
3. **Build & Run:** It creates the `RandomForestResNetCascade` system and feeds the evaluation data through it.
4. **Analytics:** Finally, it calculates the overall accuracy and prints out how many images were "escalated" to Stages 2 and 3.

### Why this is a great approach:

In a real-world production environment, you might find that 70% of your images are "easy" and can be accurately classified by ResNet18. By using this cascade, you save the immense computational cost of running ResNet152 on that 70%, reserving your heavy artillery only for the 30% of edge cases that actually need it.

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

---

## Testing # 3 : Run the IDK classifier as specified in Ronit's paper with the light weight gatekeeper Classifier. (train_gatekeeper_3_classifier.py)
* This sets up the resnet classifiers in a cascade just like the paper has documented. There is a gatekeeper skipping optimization applied. The images flow from resnet-18. Based on the gatekeeper regulator, the skip of resnet-34 can happen to delegate to  resnet-152. Here are the results of the run.

This is a well-written script that implements a technique often referred to as **Adaptive Inference** or a **Cascade Classifier**.

At a high level, this code is designed to save computational power and speed up image classification. Instead of running every single image through a massive, slow neural network (like ResNet152), it first tries a small, fast network (ResNet18). If the fast network is highly confident in its answer, the system accepts it and stops. If the fast network is confused, the system passes the image to a medium network (ResNet34), and finally to the heavy network if needed.

Here is a step-by-step breakdown of how your code achieves this.

### **1. Measuring Confidence (`extract_gatekeeper_features`)**

Before the system can decide whether to skip to a larger model, it needs to know how "confident" the current model is. The script extracts three specific features from the raw output predictions (logits) to measure this:

* **Top-1 Probability:** The highest probability score assigned to any single class. A higher score generally implies higher confidence.
* **Margin:** The difference between the highest probability and the second-highest probability. If a model predicts "Dog" at 45% and "Cat" at 44%, the margin is tiny, indicating ambiguity.
* **Entropy:** A mathematical measure of overall uncertainty across all possible classes. It is calculated using the formula $H = -\sum p \log(p)$. High entropy means the model's predictions are scattered across many classes; low entropy means it's focused on one.

### **2. Learning When to Skip (`train_gatekeepers`)**

The system doesn't guess what "confident" looks like; it learns it.

* It passes a subset of data (the calibration set) through the ResNet18 and ResNet34 models.
* For each image, it extracts the three confidence features mentioned above and checks whether the model actually got the answer right (`correct18` and `correct34`).
* It then trains lightweight **Logistic Regression** models using Scikit-Learn. These "gatekeepers" learn the decision boundary: based on the top-1, margin, and entropy, they predict whether the current model is likely correct. If predicted correct, the system stops. If predicted incorrect, the system flags the image to be skipped to the next stage.

### **3. The Native GPU Pipeline (`GatekeeperCascade`)**

This is the core PyTorch module where the inference actually happens. There are two particularly clever design choices here:

* **Zero CPU Bottlenecks:** Scikit-Learn runs on the CPU. To prevent the system from constantly moving data between the GPU (where the images are) and the CPU (where the gatekeeper logic is), the `__init__` function extracts the learned weights (`gate1.coef_`) from the Scikit-Learn models and converts them into native PyTorch tensors (`self.w1`, `self.b1`). This allows the gatekeeper logic to execute blazing fast directly on the GPU.
* **Batch Masking:** In the `forward` pass, the system processes a batch of images at once. Instead of looping through images one by one (which is slow), it uses boolean masks (`to_s2_mask`). It identifies exactly which images in the batch are uncertain, extracts only those specific images using index slicing, and passes *only* that subset to the heavier models.

### **4. Execution and Analytics (The Main Block)**

The runtime execution loop puts it all together:

* It loads the **ImageNetV2** dataset and applies standard preprocessing (resizing, cropping, normalizing).
* It splits the data, using 2000 images to train the linear gatekeepers.
* It runs the evaluation loop, tracking total accuracy and timing.
* Finally, it prints out the analytics, specifically showing what percentage of images were hard enough to require the ResNet34 (`s2_count`) or the massive ResNet152 (`s3_count`).

**Summary:** This code builds a pipeline that dynamically balances speed and accuracy. It dynamically routes "easy" images to cheap models and reserves expensive computation only for the "hard" images.

<pre>
Extracting gatekeeper features...
Total training time: 83.22 seconds

Gatekeeper Cascade Top-1 Accuracy: 0.6793
Skipped to ResNet34:  48.68%
Skipped to ResNet152: 36.98%
Total evaluation time: 591.07 seconds
</pre>

---

## Testing # 4 : Run the IDK classifier with the light weight gatekeeper Classifier all 5 resnet stages are used. (train_gatekeeper_allresnet.py)
* This sets up all resnet classifiers in a cascade adding 2 more resnet classifiers than paper has documented. There is a gatekeeper skipping optimization applied. The images flow from resnet-18. Based on the gatekeeper regulator, the skip of higher level classifiers can happen to delegate to  resnet-152. Here are the results of the run.

<pre>
Training Gate 1 (deciding resnet18 -> resnet34)...
Training Gate 2 (deciding resnet34 -> resnet50)...
Training Gate 3 (deciding resnet50 -> resnet101)...
Training Gate 4 (deciding resnet101 -> resnet152)...

Cascade deployed. Metrics will track flow through all 5 stages.
Total Gatekeeper Training time with all 5 stages: 1641.10 seconds

Gatekeeper Cascade Top-1 Accuracy: 0.6804

--- Cascade Traffic Distribution ---
Stage 1 (resnet18): 10000 images (100.00%)
Stage 2 (resnet34): 4725 images (47.25%)
Stage 3 (resnet50): 3521 images (35.21%)
Stage 4 (resnet101): 2617 images (26.17%)
Stage 5 (resnet152): 2075 images (20.75%)
Total evaluation time: 2420.42 seconds
</pre>

---

## Testing # 5 : Run the IDK classifier with the optimized light weight gatekeeper Classifier with features extracted - all 5 resnet stages are used. (train_gatekeeper_allresnet_featurepyramid.py)

This script is a clever implementation of **Cascaded Inference** (also known as "early exiting").

In simple terms, it creates a smart, cost-saving pipeline for image classification. Instead of running every single image through a massive, slow model like ResNet152, it tries to classify images using a small, fast model first (ResNet18). If the small model is confident, it accepts the answer and stops. If the small model is uncertain, a "gatekeeper" passes the image to the next, slightly larger model, and so on.

Here is a breakdown of how the code accomplishes this:

### **1. The Core Concept: Feature Extraction & Gatekeepers**

To decide if a model's prediction is trustworthy, the script looks at two things:

* **Top Probabilities:** How confident the model is between its #1 and #2 choices (extracted in `extract_gatekeeper_features`).
* **Global Average Pooling (GAP) Features:** The raw, internal representations the model built right before making its final classification.

The `train_gatekeepers` function uses a subset of data (the calibration set) to train simple **Logistic Regression** models from scikit-learn. These serve as the "gatekeepers." They learn to look at the features and probabilities and predict a simple binary outcome: *Did this model get the answer right or wrong?*

### **2. Building the PyTorch Cascade**

The `GatekeeperCascade` class is where the deployment magic happens.

Instead of keeping the gatekeepers in scikit-learn, the `__init__` function cleverly extracts the weights and biases (`g.coef_` and `g.intercept_`) from the trained Logistic Regression models and converts them into native PyTorch parameters. This allows the gatekeeper decisions to run entirely on the GPU, vastly speeding up inference.

### **3. The Forward Pass & Batch Masking**

The `forward` method is the most complex part of the script because it has to handle batches of images efficiently. You can't easily run different models on different parts of a batch simultaneously, so it uses **Boolean Masking**:

* It starts with a `mask` of all `True` values, meaning all images in the batch are passed to Stage 1 (ResNet18).
* It runs the model on the active images (`subset_x = x[mask]`).
* It computes the gatekeeper's decision using a dot product: `torch.matmul(feats, self.weights[i]) + self.biases[i]`.
* If the decision score is `< 0.0`, the gatekeeper believes the current model is wrong, and sets `next_mask` to `True` for those specific images.
* The overall `mask` is updated. In the next loop iteration (ResNet34), *only* the images that failed the previous gatekeeper are processed.

### **4. Evaluation**

Finally, the `__main__` block runs 10,000 images from the ImageNetV2 dataset through this cascade. It calculates the overall accuracy and uses `print_stats` to show the "Traffic Distribution."

If successful, you will see a distribution where the vast majority of easy images are solved by ResNet18 or ResNet34, while only a tiny percentage of the hardest images ever reach the computationally expensive ResNet152, saving massive amounts of compute time without sacrificing much accuracy.

<pre>
Training Gate 1 (deciding resnet18 -> resnet34)...
Training Gate 2 (deciding resnet34 -> resnet50)...
Training Gate 3 (deciding resnet50 -> resnet101)...
Training Gate 4 (deciding resnet101 -> resnet152)...

Total training time: 841.17 seconds
Cascade deployed. Evaluating on unseen validation data...

Overall Accuracy: 0.6442

--- Cascade Traffic Distribution ---
Stage 1 (resnet18): 5000 images (100.00%)
Stage 2 (resnet34): 2733 images (54.66%)
Stage 3 (resnet50): 1979 images (39.58%)
Stage 4 (resnet101): 1143 images (22.86%)
Stage 5 (resnet152): 770 images (15.40%)
Total evaluation time: 1247.38 seconds
</pre>

---
