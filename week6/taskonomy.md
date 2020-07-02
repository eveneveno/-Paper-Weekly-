### [TASKONOMY: Disentangling Task Transfer Learning](http://taskonomy.stanford.edu/)

Amir R. Zamir, Alexander Sax, William B. Shen, Leonidas Guibas, Jitendra Malik, Silvio Savarese


> * A fully computational method for quantifying task relationships
> * Extracting a structure out of them
> 
#### Task Relationships
<p align="center">
<img src="img/relation.png" alt="drawing" width="200"/>
</p>

Many Tasks -> Dense Relationships -> Redundancy -> Efficieny

**Task Bank**
* **26** Semantic, 2D, 3D tasks

**Dataset**
* **4 million** real Images
* Each image has the GT label for all tasks

**Task-Specific Networks**
* 26 networks

#### Experimental Results
* 26 Task-specific Networks
* 3000 Transfer Networks
* 47,829 GPU hours
* Transfers training data: 8x-120x less than task-specific

### Summary
* A striving step towards understanding the space of vision tasks.
* Treat tasks in concert, coming from a structured space, rather than isolated concepts.
* A fully computational framework.
* Transfer learning. Generalist perception model.

#### 24 Tasks
* DENOISING AUTOENCODER (Uncorrupted version of corrupted image)
* SURFACE NORMALS (Pixel-wise surface normal)
* Z-BUFFER DEPTH (Depth estimation)
* COLORIZATION (Colorizing input grayscale images)
* RESHADING (Reshading with new lighting placed at camera location)
* ROOM LAYOUT (Orientation and aspect ratio of cubic room layout)
* CAMERA POSE (FIXATED | Relative camera pose with matching optical centers)
* CAMERA POSE (NONFIX | Relative camera pose with distinct optical centers)
* VANISHING POINTS (Three Manhattan-world vanishing points)
* CURVATURES (Magnitude of 3D principal curvatures)
* UNSUPERVISED 2D SEGM (Segmentation, graph cut approximation on RGB)
* UNSUPERVISED 2.5D SEGM (Segmentation, graph cut approximation on RGB-D-Normals-Curvature image)
* 3D KEYPOINTS (3D Keypoint estimation from underlying scene 3D)
* 2D KEYPOINTS (Keypoint estimation from RGB-only, texture features)
* OCCLUSION EDGES (Edges which occlude parts of the scene)
* TEXTURE EDGES (Edges computed from RGB only, texture edges)
* INPAINTING (Filling in masked center of image)
* SEMANTIC SEGMENTATION (Pixel-wise semantic labeling via knowledge distillation from MS COCO)
* OBJECT CLASSIFICATION (1000-way object classification via knowledge distillation from ImageNet)
* SCENE CLASSIFICATION (Scene Classification (via knowledge distillation from MIT Places)
* JIGSAW PUZZLE (Putting scrambled image pieces back together)
* EGOMOTION (Odometry (camera poses) given three input images)
* AUTOENCODER (Image compression and decompression)
* POINT MATCHING (Classifying if centers of two images match or not)