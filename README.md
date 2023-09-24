# testAUC

Examples of the limitations of ROC AUC on a test set

Example 1: [example1.py](https://github.com/alonhzn/testAUC/blob/main/example1.py "example1.py")

Demonstrates the shifting of the operation point, despite the favourable AUC. 
We show predictions of a (simulated) model on the Validation set and the Test set. 
The ROC AUC "happens" to be identical. 
However, if you choose a threshold on the Validation set, aiming for a Sensitivity/Specificity operation point, you would get completely different operation point on the Test set

![Example1](https://github.com/alonhzn/testAUC/blob/main/images/example1.png?raw=true)


Example 2: [example2.py](https://github.com/alonhzn/testAUC/blob/main/example2.py "example2.py")

Demonstrates a measure of robustness to noise.
We show two models, Initially it seems that Model 1 is far better, offering ROC AUC above 91%
However, evaluating the robustness to uniform noise shows Model 2 is more robust

![Example2](https://github.com/alonhzn/testAUC/blob/main/images/example2.png?raw=true)

Example 3: [example3.py](https://github.com/alonhzn/testAUC/blob/main/example3.py "example3.py")

Demonstrates a measure of DRIFT between the validation set and the test set
We show two models, both of them "happen" to have the exact same AUC on the validation and test sets.
However, evaluating the DRIFT of the Sensitivity/Specificity operation point reveals that Model 1 is better (although in this extreme case, both models exhibit significant drifts) 

Note that at some operation point, Model 2 actually exceeds a drift of -50% from the Sensitivity expected at Validation time, compared to the Sensitivity received at test time - without affecting the total AUC. 

![Example3](https://github.com/alonhzn/testAUC/blob/main/images/example3.png?raw=true)
