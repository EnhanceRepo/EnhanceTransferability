# Enhancing Transferability of Adversarial Attacks for End-to-End Autonomous Driving Systems
This repository contains experiments conducted in the submission "Enhancing Transferability of Adversarial Attacks for End-to-End Autonomous Driving Systems"

**Abstract:** Adversarial attacks exploit the vulnerabilities of deep neural networks (DNNs). Most existing attacks for autonomous driving systems (ADSs) demonstrate strong performance under the white-box setting but struggle with black-box transferability, while black-box attacks are more practical in real-world scenarios as they operate without model access. Numerous transferability-enhancement techniques have been proposed in other fields (e.g., image classification), however, they remain unexplored for end-to-end (E2E) ADSs.  

Our study fills the gap by conducting the first comprehensive empirical analysis of nine transferability-enhancement methods on E2E ADSs, covering two types: three input transformation enhancements and six attack objective enhancements. We evaluate their effectiveness on two datasets with four steering models. Our findings reveal that, out of nine enhancements, Resizing+Translation delivers the best black-box transferability, producing up to 9.39 degrees increase in MAE. Pred+Attn serves as the best objective enhancement, producing a maximum of 5.55 degrees (white-box) and 6.21 degrees (black-box) increase in MAE.  Through attention heatmap visualizations, we discover that different models focus on similar regions when predicting, thereby enhancing the transferability of attention-based attacks.

In conclusion, our study provides valuable results and insights into the transferability-enhancement techniques for E2E ADSs, which also serve as a robust benchmark for further advancements in the autonomous driving field.

# Requirements
1. Python 3.9
2. numpy 1.26.4
3. tqdm 4.64.1
4. Torch 1.12.1
5. Torchvision 0.13.1
6. Scipy 1.10.1

# Demo
For example, to run the resizing transformation on udacity datast with adversarial perturbations crafted on the model Dave2V1. The user can run:
```
cd clean
python3 input_resizing_udacity.py --model 'dave2v1'
```

# Dave Runtime
Runtime results for Dave dataset:
![alt text](https://github.com/EnhanceRepo/EnhanceTransferability/blob/main/Dave_runtime.png)

