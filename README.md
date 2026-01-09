<img src="https://github.com/agcdke/frog/blob/master/assets/img/frog-logo.png" alt="Alt text" width="250">

# FROG
FROG (FROm Ground) is an initiative to offer several vision and language services for agricultural (business) applications. The objective of this initiative is to increase the sustainability of food supply from agricultural farms.


## Getting Started
### Vision Module
Vision module currently offers three services, i.e., image classification, U-Net based semantic segmentation, and MSI image regression for Brix Index and Anthocyanines prediction. The services are available at *vision* directory. 

**AgriBlazeNet:** It offers four variants of an image classification [model](assets/img/AgriBlazeNet.png), which incorporates BlazeFace feature extraction network as a baseline. It incorporates CBAM module(S. Woo et al.) in one variant. It also incorporates a custom CNN attention module known as Convolutional Block with Spatial Self-Attention Module(CBwSSAM), where the spatial attention of CBAM is replaced by the Vision Transformer (particularly MobileViT) to enhance the attention capability for several agricultural applications. The architecture of two variants of [CBwSSAM](assets/img/CBwSSAM.png) is presented at IEEE ICTAI, 2025 (Athens) conference. 

**AgriBlazeU-Net:** It offers an U-Net based semantic segmentation model, where CBwSSAM variants are incorporated. The annotated RGB images are inferred automatically during model training. For example, weed, crop and soil are annotated as red, green and black respectively. The annotated information is saved in *unet/data_dir/cwfid_class_dict.csv* file for CWFID dataset, and inferred automatically during model training. The information about train and testset is mentioned at *unet/data_dir/cwfid_train_test_split.yaml* file. This work will be presented at GIL, 2026 conference.

**MSI-Brix-Anthocyanines:** It offers a MSI image regression framework that takes multi-spectral images (here, MSI Grapes images from 3DeepM dataset, Navarro et al., Universidad Politécnica de Cartagena, Spain) for Brix Index and Anthocyanines prediction of grapes. The MSI images are saved in TIF/TIFF (Tagged Image File Format). The fraeḿework uses BlazeFace and MobileNetv2 models along with 3 types of MobileViT blocks, i.e., MobileViT (T0), Type-1 CBwSSAM (T1), and Type-2 CBwSSAM (T2).

### Language Module
Language module currently offers Question-Answering service. The services are available at *language* directory. 

**Question-Answering** It offers a preliminary Retrieval-Augmented Generation(RAG) based Question-Answering framework which uses LangChain, Chroma, and Ollama. It is available at *language/agriqa* directory.

## Acknowledgement:
Verify markdown texts at [Markdown Live Preview](https://markdownlivepreview.com)