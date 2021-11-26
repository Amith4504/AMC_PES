# AMC_PES
Automatic modulation classification using Expert features and Convolutional Neural Networks.

# Network Architecture.
The network consists of 2 Convolutional Feature Detection layers . The first layer consists of 256 filters (kernels) each of size 3X1 , and the 2nd layer consists of 80 filters each of size 3X2 . We input 2X128 IQ (In phase & Quadrature) samples to the first layer. Convolution layers are followed by a Dense layer with 256 nodes in hidden layer and 11 nodes in the last layer. 

![Screenshot from 2021-11-26 23-51-49](https://user-images.githubusercontent.com/21309793/143619257-cec6b663-0912-4584-a295-6baa668e9dd0.png)


# Confusion Matrices for different Signal to Noise Ratios
## 16 SNR
![16snr](https://user-images.githubusercontent.com/21309793/143619307-6aa1864e-f605-4732-8859-ee08e797f0e8.png)

## 12 SNR

![12snr](https://user-images.githubusercontent.com/21309793/143619369-e46c5f2e-28ba-4049-84ab-5dd4c0e4fe8a.png)

## 8 SNR
![8snr](https://user-images.githubusercontent.com/21309793/143619410-c7655d85-e2a2-4df2-a700-6a626cfe1a0f.png)

## -10 SNR
![-10snr](https://user-images.githubusercontent.com/21309793/143619490-7c6826d9-c700-446d-a1fc-ee1f6c88e243.png)


## -18 SNR

![-18snr](https://user-images.githubusercontent.com/21309793/143619467-2d076e82-b2ae-4afc-9aaa-aff1ac3db96f.png)

