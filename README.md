# Semi-Supervised Learning With GANs: Revisiting Manifold Regularization

This is the code we used in [our](https://openreview.net/forum?id=Hy5QRt1wz) paper accepted at ICLR workshop 2018
>[Semi-Supervised Learning With GANs: Revisiting Manifold Regularization]

>Bruno Lecouat*, Chuan Sheng Foo*, Houssam Zenati, Vijay Ramaseshan Chandrasekhar

Please reach us via emails or via github issues for any enquiries!

Please cite our work if you find it useful for your research and work!

```
@misc{
  lecouat2018,
  title={Semi-Supervised Learning With GANs: Revisiting Manifold Regularization},
  author={Bruno Lecouat and Chuan Sheng Foo and Houssam Zenati and Vijay Ramaseshan Chandrasekhar},
  year={2018},
  url={https://openreview.net/forum?id=Hy5QRt1wz}
}

```


## Requirements

The repo supports python 3.5 + tensorflow 1.4


## Run the Code


To reproduce our results on SVHN
```
python train_svhn.py
```

To reproduce our results on CIFAR-10
```
python train_cifar.py
```

## Results

Here is a comparison of different models using standard architectures (1000 labels on SVHN, and 4000 labels on CIFAR):

Method | SVHN (% errors) | CIFAR (% errors)
-- | -- | --
CatGAN | - | 19.58 +/- 0.46
Ladder Network | - | 20.40 +/- 0.47
FM  | 8.11 +/- 1.3 | 18.63 +/- 2.32
ALI | 7.42 +/- 0.65 | 17.99 +/- 1.62
VAT small |  6.83 | 14.87
Bad GAN  | 4.25 +/- 0.03 | 14.41 +/- 0.30
Ours | **4.51 +/- 0.22**| **14.45 +/- 0.21**



