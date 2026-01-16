# MPCNet: Multi-scale Perception and Cross-attention Feature Fusion Network for Infrared Small Target Detection

On January 5, 2026, our paper was officially accepted by the ***IEEE Transactions on Geoscience and Remote Sensing***. We sincerely thank all the reviewers and editors for their valuable comments and patient guidance during the review process, which played a crucial role in improving the quality of the paper. We are deeply honored and express our heartfelt gratitude for their support and assistance. [[paper]](https://ieeexplore.ieee.org/document/11346810)

## Network
![outline](image/img1.jpg)

## Datasets
**Our project has the following structure:**
  ```
  ├───dataset/
  │    ├── NUAA-SIRST
  │    │    ├── image
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  │    │    │    ├── ...
  │    │    ├── mask
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  │    │    │    ├── ...
  │    │    ├── train_NUAA-SIRST.txt
  │    │    │── train_NUAA-SIRST.txt
  │    ├── IRSTD-1K
  │    │    ├── image
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── mask
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── train_IRSTD-1K.txt
  │    │    ├── train_IRSTD-1K.txt
  │    ├── ...  
  ```
<be>

## Results
#### Qualitative Results

![outline](image/img2.png)

#### Precision–Recall Curve

![outline](image/img3.png)

#### Supplementary noise experiment
| Methods | σ=10 IoU | σ=10 Fm | σ=20 IoU | σ=20 Fm | σ=30 IoU | σ=30 Fm | σ=1 IoU | σ=1 Fm | σ=3 IoU | σ=3 Fm | σ=5 IoU | σ=5 Fm | σ=1 IoU | σ=1 Fm | σ=3 IoU | σ=3 Fm | σ=5 IoU | σ=5 Fm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | 67.36 | 80.38 | 68.55 | 81.24 | 62.18 | 76.56 | 70.97 | 82.91 | 68.35 | 81.07 | 67.07 | 80.17 | 64.34 | 78.18 | 61.71 | 76.16 | 54.35 | 70.34 |
| RDIAN | 70.91 | 82.86 | 69.27 | 81.72 | 65.99 | 79.37 | 73.57 | 84.66 | 71.14 | 82.99 | 70.81 | 82.79 | 69.25 | 81.71 | 66.21 | 79.59 | 64.64 | 78.40 |
| DNANet | 77.01 | 86.87 | 70.56 | 82.64 | 68.59 | 81.26 | 75.53 | 85.95 | 72.66 | 84.09 | 72.20 | 83.78 | 72.32 | 83.84 | 71.25 | 83.12 | 70.90 | 82.85 |
| UIUNet | <span style="color:blue">77.77</span> | <span style="color:blue">87.58</span> | <span style="color:blue">74.53</span> | <span style="color:blue">85.31</span> | <span style="color:blue">68.70</span> | <span style="color:blue">81.34</span> | 80.08 | <span style="color:blue">88.79</span> | <span style="color:blue">77.43</span> | <span style="color:blue">87.10</span> | 74.21 | <span style="color:blue">85.06</span> | <span style="color:blue">76.23</span> | <span style="color:blue">87.00</span> | 74.05 | 84.95 | 72.64 | 84.02 |
| RPCANet | 65.44 | 79.11 | 50.58 | 67.18 | 44.46 | 61.56 | 62.59 | 76.99 | 60.41 | 75.32 | 59.97 | 74.98 | 54.12 | 70.23 | 44.62 | 61.70 | 38.22 | 55.30 |
| MSHNet | 73.54 | 84.63 | 70.91 | 82.86 | <span style="color:blue">68.77</span> | <span style="color:blue">81.37</span> | 70.53 | 82.59 | 69.47 | 81.84 | 68.73 | 81.35 | 72.64 | 84.01 | 69.43 | 81.78 | 66.02 | 79.41 |
| PBT | 71.81 | 83.59 | 67.01 | 80.25 | 62.47 | 76.90 | 75.68 | 86.16 | 71.52 | 83.40 | 69.60 | 82.07 | 64.56 | 78.47 | 61.68 | 76.30 | 59.38 | 74.51 |
| SCTransNet | 73.61 | 84.68 | 70.81 | 82.78 | 68.51 | 81.18 | 77.06 | 86.93 | 74.80 | 85.45 | <span style="color:blue">72.91</span> | <span style="color:blue">84.22</span> | <span style="color:blue">74.22</span> | <span style="color:blue">85.08</span> | <span style="color:blue">73.74</span> | <span style="color:blue">84.75</span> | <span style="color:blue">73.10</span> | <span style="color:blue">84.33</span> |
| IDU-Net | 72.89 | 84.20 | 71.36 | 83.22 | 67.83 | 80.76 | <span style="color:blue">77.69</span> | <span style="color:blue">87.31</span> | <span style="color:red">77.67</span> | <span style="color:red">87.29</span> | 69.18 | 81.68 | <span style="color:blue">74.02</span> | <span style="color:blue">84.92</span> | <span style="color:blue">73.10</span> | <span style="color:blue">84.35</span> | 71.60 | 83.34 |
| MPCNet | <span style="color:red">78.13</span> | <span style="color:red">87.61</span> | <span style="color:red">75.62</span> | <span style="color:red">85.99</span> | <span style="color:red">69.14</span> | <span style="color:red">81.65</span> | <span style="color:red">78.66</span> | <span style="color:red">87.93</span> | <span style="color:red">78.23</span> | <span style="color:red">87.68</span> | <span style="color:red">76.00</span> | <span style="color:red">86.23</span> | <span style="color:red">78.19</span> | <span style="color:red">87.60</span> | 73.31 | 84.45 | <span style="color:red">73.13</span> | <span style="color:red">84.35</span> |

#### Quantitative Results on NUAA-SIRST, IRSTD-1K and NUDT-SIRST

| Dataset         | IoU (x10(-2)) | Pd(x10(-2))| Fa (x10(-6))|  F (x10(-2))|
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| NUAA-SIRST    | 77.47  |  96.20 | 13.72 | 87.30 |
| IRSTD-1K      | 67.24  |  92.26 | 11.41 | 80.40 |
| NUDT-SIRST    | 93.33  |  99.15 | 1.68  | 96.55 |


*This code is highly borrowed from [SCTransNet](https://github.com/xdFai/SCTransNet). Thanks to Shuai Yuan.

*The overall repository style is highly borrowed from [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks to Boyang Li.








