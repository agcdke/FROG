# Vision
My doctoral research work focuses on harnessing MobileViT block in agricultural vision applications that could be deployed on Edge or Mobile devices. It has 3 basic parts:
<ol>
<li> Enhance CBAM block using MobileViT</li>
<li> Minimize error-metrics for Multi-spectral image regresion</li>
<li> Evaluate MobileViT block on texture and shape biasness</li>
<li></li>
</ol>
### Experiment workstation
### AgriBlazeNet
This work is presented at IEEE International Conference on Tools with Artificial Intelligence (ICTAI), 2025. BlazeFace, a lightweight and well-performing face detector tailored for mobile GPU inference is presented by Google and used in Google MediaPipe solutions. Four variants of BlazeFace-based image classification models are presented:
<ol>
<li>Blz: vanilla BlazeFace-based image classification</li>
<li>BlzCBAM: BlazeFace-based image classification with CBAM block.</li>
<li>T1H4B4: BlazeFace-based image classification with Type-1 CBwSSAM block (H4B4: heads of multi-head attention=4, no. of blocks=4 for Transformers block).</li>
<li>T2H4B4: BlazeFace-based image classification with Type-2 CBwSSAM block (H4B4: heads of multi-head attention=4, no. of blocks=4 for Transformers block).</li>
</ol>

### AgriBlazeU-Net
This work will be presented at GIL-Jahrestagung, 2026. Incorporating Type-1 and Type-2 CBwSSAM blocks within Single and Double BlazeBlocks. Four variants of BlazeFace-based U-Net semantic segmentation models are presented:
<ol>
<li>Blz: vanilla BlazeFace-based image classification</li>
<li>BlzCBAM: BlazeFace-based image classification with CBAM block.</li>
<li>T1H8B8: BlazeFace-based image classification with Type-1 CBwSSAM block (H8B8: heads of multi-head attention=8, no. of blocks=8 for Transformers block).</li>
<li>T2H8B8: BlazeFace-based image classification with Type-2 CBwSSAM block (H8B8: heads of multi-head attention=8, no. of blocks=8 for Transformers block).</li>
</ol>
The loss function is <em>Unet3p Hybrid</em> loss. 

### MSI-Brix-Anthocyanines
This ongoing work focuses on minimizing error metrics to predict Brix Index and Anthocyanines. Work-in-progress.