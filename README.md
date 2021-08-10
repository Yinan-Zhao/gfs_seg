# PyTorch Generalized Few-Shot Semantic Segmentation

### Introduction

This repository is a PyTorch implementation for generalized few-shot semantic segmentation. It is developed based on [this repo](https://github.com/hszhao/semseg.git) for semantic segmentation / scene parsing.


### Usage

1. Requirement:

   - Hardware: better with >=11G GPU memory
   - Software: PyTorch>=1.1.0, Python3, [tensorboardX](https://github.com/lanpa/tensorboardX), 

   env.yml is an environment file that works on Dart machine. You can clone the environment by running:

   ```shell
   conda env create -f env.yml
   ```

2. Clone the repository:

   ```shell
   git clone git@github.com:Yinan-Zhao/gfs_seg.git
   ```

3. Dataset and Initialization:

   - Download PASCAL and COCO datasets from [here](https://drive.google.com/file/d/1I1INfF55axAqcmXHiJutLfzxjrpDfiU2/view?usp=sharing) and put them under folder `datasets`(you can alternatively modify the relevant paths specified in folder `config`). `COCO` and `pascal` should be found under `datasets`.

     ```
     cd semseg
     mkdir -p datasets
     ```


   - Download ImageNet pre-trained [models](https://drive.google.com/drive/folders/1RtPZRpyt4B3MCegzRG9cPTornpy2OFrZ?usp=sharing) and put them under folder `initmodel` for weight initialization. Remember to use the right dataset format detailed in [FAQ.md](./FAQ.md).


3. The First Training Stage: Base Training

   - Specify the gpu id used in config and the fold id, then do training:

     ```shell
     sh tool/train_normal.sh pascal pspnet50_normal_split0
     ```
     ```shell
     sh tool/train_normal.sh coco pspnet50_normal_split0
     ```
     If split id is specified as 100, it will train with all the training images.


4. The Second Training Stage: Few-Shot Fine-tuning

   - Specify `weight` as the trained model we obtain in the first base training stage and # of shots in config, then do fine-tuning:

     ```shell
     sh tool/train_finetune.sh pascal pspnet50_normal_finetune_split0_shot10_lr1e2
     ```
     ```shell
     sh tool/train_finetune.sh coco pspnet50_normal_finetune_split0_shot10_lr1e2
     ```

5. Test:

   - Download trained segmentation models and put them under folder specified in config or modify the specified paths.

   - For full testing (get listed performance):

     ```shell
     sh tool/test_normal.sh coco pspnet50_normal_finetune_split0_shot5_lr1e2
     ```


6. Visualization: [tensorboardX](https://github.com/lanpa/tensorboardX) incorporated for better visualization.

   ```shell
   tensorboard --logdir=exp/voc2012/pspnet50_normal_finetune_split0_shot5_lr1e2/
   ```

### Resources:
   - [Trained Models](https://drive.google.com/file/d/1pzoCj_en7mNZhai93dsw-q_4KbsBhi3J/view?usp=sharing)
   - [Method Figure](https://docs.google.com/presentation/d/1Jrlp9uXRHdii4Y8yWSnfLPo8lLGnDmnv/edit?usp=sharing&ouid=117678560817841457144&rtpof=true&sd=true)
   - Few-shot object detection ([paper](https://arxiv.org/pdf/2003.06957.pdf)), Few-shot image classification ([paper1](https://arxiv.org/pdf/2003.11539.pdf)), ([paper2](https://arxiv.org/pdf/1909.02729.pdf))
   - Data analysis tool: [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) 
   - Datasets: attributes (`names` and `colors`) are in folder `data` and some sample lists can be accessed.
   - Some FAQs from the original repo: [FAQ.md](./FAQ.md).

### Next Steps:

  - Similarity and difference between different few-shot tasks (semantic segmentation, object detection and image classification). Although fine-tuning based approaches have shown success in each task, the set of parameters that need to fine-tuned are different (for segmentation we fine-tune all the layers after the ResNet backbone while for object detection only the last conv layer is fine-tuned). The different set of parameters that need to be fine-tuned indicates the difference in terms of the generalization of the learned features. 

### Contact:
  - For any questions, please contact Yinan Zhao (yinanzhao@utexas.edu)

