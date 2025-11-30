# Histopathologic-Cancer-Detection
Histopathologic Cancer Detection Project

Background
In this data competition, we need to identify metastatic cancer in small image patches taken from larger digital pathology scans.
The train_labels.csv file provides the ground truth for the images in the train folder. We will use the train_labels.csv to train our models. Our goal is to predict the labels for the images in the test folder.
A positive label indicates that the center 32x32px region of a patch contains at least one pixel of cancer tissue.
The training data includes 220025 images; their ID is something that looks like this: f38a6374c348f90b587e046aac6079959adf3835. And each image is labeled with 0 or 1 for cancer detection results.

Exploratory Data Analysis (EDA)
Data Structure
Firstly, let us check the training data to see the class distribution.
From the visualization, we can see there are more data labeled in no tumor (0) than tumor (1). Total number is 130908 VS 89117. But overall, I think the contribution is ok for training.


Deep Learning Models
We start preparing the cancer dataset, get RGB image and id for both training and test folders' images.
Compare 3 Models
1. TinyCNN
TinyCNN is a lightweight CNN with 3 convolution levels. Using ReLu for activation and max pooling.
2. Logistic CNN
Logistic CNN is using only 1 convolution level. No activation and using global average pooling.
3. ResNet18
ResNet18 is a more advanced model with residual connections. The skip connection solves vanishing gradients and allows deeper feature learning.

Results and Analysis
Let's check the performance result AUC for each model.

TinyCNN VAL AUC: 0.96970
LogisticCNN VAL AUC: 0.63776
ResNet18 VAL AUC: 0.97673
Overall, ResNet18 performs best. It has a deeper architecture and can capture more complex features.

For TinyCNN and LogisticCNN, they are more straightforward, but may be because of limited convolution levels, they may lost more information during the process.
Especially, the logisticCNN with only 1 convolution level, its performance is poor.

Furthermore, I optimize ResNet with different learning rates and epochs.

Optimization
Let's optimize ResNet18 by changing the learning rates and epochs. For ResNet18, it requires more epochs to reach a good result comparing with TinyCNN or LogisticCNN. So I am increase the epochs (6, 8, 10) and test with different learning rates (1e-4, 5e-4, 1e-3).
The result below shows when learning rate is 5e-4 and epochs is 10, it got the best AUC result:
Best model by VAL AUC: ResNet18_lr=0.0005_ep=10 AUC = 0.9889461398124695.
But when I applied this best model to testing, the final result for test dataset is worse. Score is 0.9084. It is less than the previous v2, ResNet with 0.0001 and 5 epochs. So it is clearly overfitting.

Conclusion
In this project, we compared several convolutional neural network architectures—including TinyCNN, LogisticCNN, and ResNet18 and evaluated how different hyperparameters, particularly the learning rate and number of training epochs, influenced model performance.
Overall, ResNet18 provided the strongest performance. But more epochs doesn't help a lot for ResNet18 Model, maybe because it is a deeper architecture and the epochs doesn't helps a lot for this stable architecture. Instead, it create an overfitting issue.Future work could extend this analysis using automated hyperparameter search methods to further optimize model performance.


Reference¶
ResNet18: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
AUROC: https://lightning.ai/docs/torchmetrics/stable/classification/auroc.html
