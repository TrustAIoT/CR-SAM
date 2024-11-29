This repo contains the source code and evaluation scripts for our AAAI 2024 paper:

## CR-SAM: Curvature Regularized Sharpness-Aware Minimization

[Link to paper on publisher site](https://ojs.aaai.org/index.php/AAAI/article/view/28431)<br>
[arXiv](https://arxiv.org/abs/2312.13555)

### Abstract

The capacity to generalize to future unseen data stands as one of the utmost crucial attributes of deep neural networks. Sharpness-Aware Minimization (SAM) aims to enhance the generalizability by minimizing worst-case loss using one-step gradient ascent as an approximation. However, as training progresses, the non-linearity of the loss landscape increases, rendering one-step gradient ascent less effective. On the other hand, multi-step gradient ascent will incur higher training cost. In this paper, we introduce a normalized Hessian trace to accurately measure the curvature of loss landscape on {\em both} training and test sets. In particular, to counter excessive non-linearity of loss landscape, we propose Curvature Regularized SAM (CR-SAM), integrating the normalized Hessian trace as a SAM regularizer. Additionally, we present an efficient way to compute the trace via finite differences with parallelism. Our theoretical analysis based on PAC-Bayes bounds establishes the regularizer's efficacy in reducing generalization error. Empirical evaluation on CIFAR and ImageNet datasets shows that CR-SAM consistently enhances classification performance for ResNet and Vision Transformer (ViT) models across various datasets.

### Citation:
```
@inproceedings{aaai2024crsam,
  title={{CR-SAM}: Curvature Regularized Sharpness-Aware Minimization},
  author={Wu, Tao and Luo, Tie and Wunsch II, Donald C},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  volume={38},
  number={6},
  pages={6144--6152},
  year={2024}
}
```
