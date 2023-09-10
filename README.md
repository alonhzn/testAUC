# testAUC

Examples of the limitations of ROC AUC on a test set

Example 1: [AUC_issue_example1.py](https://github.com/alonhzn/testAUC/blob/main/AUC_issue_example1.py "AUC_issue_example1.py")

You can see the predictions of a (simulated) model on the Validation set and the Test set. The ROC AUC is identical. However, if you choose a threshold on the Validation set, aiming for a Sensitivity/Specificity operation point, you would get completely different operation point on the Test set

![alt text](https://github.com/alonhzn/testAUC/blob/main/images/example1.png)
