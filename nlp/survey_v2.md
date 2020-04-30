`异想天开 1.`
**What role are you playing, my little PIXEL?**
(top-down -> bottom-up)
**Self-supervised learning**
no pre-defined category
reasons about an image from pixel/patch/global level.
co-operation relation, supplementary, substitute, complement ...

(Inpainting => eliminate the task of ..., then it needs to look at other pixels for co-operation)
**抱团游戏** (1,2,3 ... )
不同的下游任务对应不同的task
rich feature只是说懂得如何挑选他的合作伙伴
cam?

semantic segmentation (difference map)
scene graph generation (structure, how to extract knowledge)

Attention => via Kernel lens 
Attention => difference map 

`异想天开 2.`
**Color Pallette**
How to sample color from palette, and how to locate them given a boundary.
颜色块（symbol） + 轮廓（语序）

feature representation 
shape + color

完形填空，写作，是人类对语言掌握的技能体现
那么绘画，则是人们对视觉世界的重构。
**（素描/色彩）**
1. reconstruction
2. (common) object feature 如何去embed shape feature，为shape构建意义
3. ImageNet 哪些物体是由颜色决定的 哪些是由shape决定的

SIMCLR: **color distortion**(色彩) + **cropping** (猜测：可能没有shape信息好)


<br>


|  Paper | Venue | Remark
| ---------------- | ---- | ------------ | 
| [Image Transformer ✨](https://arxiv.org/pdf/1802.05751.pdf)| arxiv 2018 <br>(rejected by ICLR2018)| Cite [Conditional Image Generation with PixelCNN Decoders (NIPS'16)](https://arxiv.org/pdf/1606.05328.pdf) and [PixelSNAIL: An Improved Autoregressive Generative Model (ICML'18)](https://arxiv.org/pdf/1712.09763.pdf) <br>[[Code pytorch]](https://github.com/sahajgarg/image_transformer) 单向 => 双向 => 四向
|[Contrastive Multiview Coding](https://arxiv.org/pdf/1906.05849.pdf) | arxiv 2020 (v2)| maximize Mutual Information
|[Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/pdf/1606.05328.pdf)| NIPS 2016 |
| [Pixel Recurrent Neural Networks](https://arxiv.org/pdf/1601.06759.pdf)| 
|[Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/pdf/1603.09246.pdf) | ECCV 2016 | predict permutation order
| [Improving Generalization via Scalable Neighborhood Component Analysis](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhirong_Wu_Improving_Embedding_Generalization_ECCV_2018_paper.pdf) | ECCV 2018 | 
| [PatchVAE: Learning Local Latent Codes for Recognition](https://arxiv.org/pdf/2004.03623.pdf) | arxiv 2020 |  withdraw(reject) from ICLR2020  
| [Steering Self-Supervised Feature Learning Beyond Local Pixel Statistics](https://arxiv.org/pdf/2004.02331.pdf) | arxiv 2020 | 
| [💕【CPC】Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)|arxiv 2019 <br> (rejected by ICLR2020) | [`Talk`](https://slideslive.com/38922758/invited-talk-contrastive-predictive-coding) | 
| [The Right Tool for the Job: Matching Model and Instance Complexities](https://arxiv.org/pdf/2004.07453.pdf) | ACL 2020 [[code] (pytorch)](https://github.com/allenai/sledgehammer)| 
| [💕A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf) | arxiv 2020 [[code] (TPU+TF)](https://github.com/google-research/simclr) | We show that (1) **composition of data augmentations** plays a critical role in defining effective predictive tasks, (2) introducing **a learnable nonlinear transformation** between the representation and the contrastive loss substantially improves the quality of the learned representations, and (3) contrastive learning benefits from **larger batch sizes and more training steps** compared to supervised learning. <br> <img src="img/simclr.png" alt="drawing" width="300"/>
|[Learning Representations by Maximizing Mutual Information Across Views](https://papers.nips.cc/paper/9686-learning-representations-by-maximizing-mutual-information-across-views.pdf)| NIPS 2019 | 
| [Invariant Information Clustering for Unsupervised Image Classification and Segmentation](https://arxiv.org/pdf/1807.06653.pdf) | ICCV 2019 | 