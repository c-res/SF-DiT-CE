## Sampling-Free Diffusion Transformers for Low-Complexity MIMO Channel Estimation

[![icon](https://img.shields.io/badge/ArXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2602.02202) [![python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/) [![pytorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/) 




***Zhixiong Chen, Hyundong Shin, Arumugam Nallanathan***

***Queen Mary University of London***

⭐ **If SF-DiT-CE is helpful to you, please star this repo. Thanks!** 🤗


## 📝 Abstract

Diffusion model-based channel estimators have shown impressive performance but suffer from high computational complexity because they rely on iterative reverse sampling. This paper proposes a sampling-free diffusion transformer (DiT)-based channel estimator, termed SF-DiT-CE, for low-complexity MIMO channel estimation. Exploiting angular-domain sparsity of MIMO channels, we train a lightweight DiT to directly predict the \tobu{true channels} from their perturbed observations and noise levels. At inference, we first obtain an initial channel estimate using the least-squares (LS) method, which can be viewed as the true channel corrupted by Gaussian noise. The DiT then takes this estimate and its corresponding noise scale as inputs to recover the channel in a single forward pass, eliminating iterative sampling. Numerical results demonstrate that our method achieves superior estimation accuracy and robustness with significantly lower complexity than state-of-the-art baselines. 


## 😍 Main Results

Comparison of different channel estimators on CDL-C dataset:

[<img src="assets/Fig2.jpg" width="800"/>](https://imgsli.com/MzkzNjU5)

Comparison of diffusion-based channel estimators on CDL-D dataset:

[<img src="assets/Fig3.jpg" width="800"/>](https://imgsli.com/MzkzNjU5)

Impact of (a) prediction objective and (b) loss function on NMSE of
the proposed approach.

[<img src="assets/Fig4a.jpg" height="300px"/>](https://imgsli.com/MzkzNjU5)[<img src="assets/Fig4b.jpg" height="300px"/>](https://imgsli.com/MzkzNjY5)


## ⚙ Preparation

1. Installation
```
conda create -n SFDiTCE python=3.12
conda activate SFDiTCE
pip install -r requirements.txt
```
2. Download the datasets and pretrained checks from [here](https://drive.google.com/drive/folders/10wZHkuYvWO1cA1WCrLpPkgZcyGYrO2HH).



## ⚡ Inference for MIMO channel Estimation

**Step 1: Put the dataset files under the path of /Channel_datasets**

**Step 2: Put VE_checkpoint under the path of /VE_DiT**

**Step 3: Run /VE_DiT/eval_VE_DiT.py to test the channel estimation performance**
```bash
python VE_DiT/eval.py \
    --is_angular=True \
    --predict_obj='X_predict' \
    --loss_type='V_loss' 
```
* To evaluate the variants of SF-DiT-CE, change the parameter settings:

|Variants|is_angular| predict_obj       | loss_type |
|--------|--------|-------------------|-----------|
|Ours (in spatial-domain)| False| 'X_predict'       | 'V_loss'  |
|$`\mathbf{V}`$-prediction| True| 'V_predict'       | -         |
|$`\mathbf{\epsilon}`$-prediction| True | 'epsilon_predict' | -         |
|$`\mathbf{X}`$-loss | True| 'X_predict'| 'X_loss'  | 
|$`\mathbf{\epsilon}`$_loss | True| 'X_predict'| 'epsilon_loss' | 

* To evaluate SF-DIT-CE that operated in VP perturbation ("Ours (with VP perturbation)"):

1. Put VP_checkpoint under the path of /VP_DiT
2. Run:
```bash
python VP_DiT/eval_VP_DiT.py
```


## 🔥 Training

* For training the VE version of SF-DiT-CE, run:
```bash
python VE_DiT/train_VE_DiT.py \
    --is_angular=True \
    --predict_obj='X_predict' \
    --loss_type='V_loss' 
```
Please modify these parameters to obtain different variants of SF-DiT-CE.


* For training the VP version of SF-DiT-CE, run:
```bash
python VE_DiT/train_VP_DiT.py \
    --is_angular=True \
    --predict_obj='X_predict' \
    --loss_type='V_loss' 
```

* Note: You can change other parameters to obtain more variants of our SF-DiT-CE.



## 🍭 Train and inference other diffusion models (Optional)

#### Run Score model on our dataset:

Please refer the [score code](https://github.com/utcsilab/score-based-channels), you can use our dataset and pretrained checkpoints [score_checkpoints].

#### Run DMCE on our dataset:

Please refer the [DMCE code](https://github.com/benediktfesl/Diffusion_channel_est), you can use our dataset and pretrained checkpoints [DMCE_checkpoints].

Note: for DMCE, we provide train and evaluation scripts, Put the dataset files under the path of /Channel_datasets, then put DMCE_checkpoints under the path of /Run_DMCE,
Then, down DMCE code, copy the DMME and modules directories into Run_DMCE
- run eval_DMCE.py to evaluate DMCE, 
- run eval_DMCE_with_DiT.py to evaluate DMCE with DiT
- run train_DMCE.py to train DMCE
- run train_DMCE_with_DiT.py


## :book: Citation

If you find our work inspiring, please consider citing:
```bibtex
@article{chen2026sampling,
  title={Sampling-Free Diffusion Transformers for Low-Complexity MIMO Channel Estimation},
  author={Chen, Zhixiong and Shin, Hyundong and Nallanathan, Arumugam},
  journal={arXiv preprint arXiv:2602.02202},
  year={2026}
}
```

## 🥰 Acknowledgement

This work is implemented based on [JiT](https://github.com/LTH14/JiT?tab=readme-ov-file), [score-based-channels](https://github.com/utcsilab/score-based-channels). Thanks for their awesome work!












