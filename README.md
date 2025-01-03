# MPCNet: Multi-scale Perception and Cross-attention Feature Fusion Network for Infrared Small Target Detection

**We have submitted the paper for review and will make the code available after publication.**

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

![outline](image/img2.jpg)

#### Quantitative Results on NUAA-SIRST, and IRSTD-1K

| Dataset         | IoU (x10(-2)) | Pd(x10(-2))| Fa (x10(-6))|  F (x10(-2))|
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| NUAA-SIRST    | 77.47  |  96.20 | 13.72 | 87.30 |
| IRSTD-1K      | 67.24  |  92.26 | 11.41 | 80.40 |
| NUDT-SIRST    | 93.33  |  99.15 | 1.68  | 96.55 |


*This code is highly borrowed from [SCTransNet](https://github.com/xdFai/SCTransNet). Thanks to shuai Yuan.

*The overall repository style is highly borrowed from [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks to Boyang Li.








