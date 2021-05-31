<h2>Deep Image Prior</h2>

> Deep convolutional networks have become a popular tool for image generation and restoration. 
> Generally, their excellent performance is imputed to their ability to learn realistic image priors from a large number of example images. In this paper, we show that, on the contrary, the structure of a generator network is sufficient to capture a great deal of low-level image statistics prior to any learning. In order to do so, we show that a randomly-initialized neural network can be used as a handcrafted prior with excellent results in standard inverse problems such as denoising, super-resolution, and inpainting.
> Furthermore, the same prior can be used to invert deep neural representations to diagnose them, and to restore images based on flash-no flash input pairs.

Reconstruction of Deep Image prior [1] in pytorch for denoising and inpainting. 
For additional please refer to source repository [2].  

<h3>Instructions</h3>

Check [requirments.txt](https://github.com/Sergo2020/Deep_Image_Prior_pytorch/blob/master/requirements.txt) and run main.py. You may specify task ('jpeg', 'denoise' or 'inpaint') and image location. 
Program will result two images - original and proccessed, and a plot of loss values during the training.

Examples of inpainting and denoisign below:

<p align="center">
  <strong> Denoising </strong>
</p>

<p align="center">
  <img src="https://github.com/Sergo2020/Deep_Image_Prior_pytorch/blob/master/results/doge_noise.png" width="40%"  alt="Noisy doge" />
  <img src="https://github.com/Sergo2020/Deep_Image_Prior_pytorch/blob/master/results/doge_denoised.png" width="40%" alt="Doge denoised" />
</p>
<p align="center">
  <hr size="0" width="100%"> 
</p>
<p align="center">
  <strong> Inpainting </strong>
</p>
<p align="center">
  <img src="https://github.com/Sergo2020/Deep_Image_Prior_pytorch/blob/master/results/car_paint.png" width="40%"  alt="Scratched car" />
  <img src="https://github.com/Sergo2020/Deep_Image_Prior_pytorch/blob/master/results/car_clean.png" width="40%"  alt="Car" />
</p>

<h3>References</h3>

[[1]](https://sites.skoltech.ru/app/data/uploads/sites/25/2018/04/deep_image_prior.pdf) Ulyanov, D. et al. “Deep Image Prior.” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (2018): 9446-9454.  
[[2]](https://github.com/DmitryUlyanov/deep-image-prior) Ulyanov, D. et al. (2018), 
DmitryUlyanov/deep-image-prior. Github
