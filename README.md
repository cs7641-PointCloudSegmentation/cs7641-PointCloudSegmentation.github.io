# CS7641 Project Proposal

## 3D Data Clustering and Segmentation for Indoor Environment

<h1 align="center">
<img src="https://user-images.githubusercontent.com/85851349/121944077-3c584f00-cd20-11eb-9a6d-91de2e1b2866.png" width="50%" /> 
</h1>
<p align = "center">
Figure 1. Example of point cloud processing result from [8]
</p>

## Team members
Kaylah Facey, Madison Manley, Seongyong Kim, Yosuke Yajima

## Introduction & Background
With the advancement of sensor technology, there is an explosion of 3D data, called point clouds, that is captured using 3D depth cameras and LiDAR sensors. Point clouds are useful in diverse applications including self-driving cars, surveys of infrastructure monitoring, and indoor mappings for buildings. Compared to research in 2D images, processing 3D data is a relatively new technology that uses machine learning to classify and detect objects. Using 3D data is advantageous as it is difficult, slow, and error prone to reconstruct 3D spaces using 2D images.

## Problem Statement
In civil engineering, point clouds are being used to create indoor mapping building models, as this method is significantly faster and more accurate compared to creating models by hand. However, once a point cloud is collected using LiDAR or 3D depth cameras, it takes a considerable amount of time and labor intensive work to process the point clouds into a usable model. Automating point cloud processing and reducing the processing time is very critical for applications that require point clouds to be processed immediately. In this project, we want to demonstrate how machine learning can be used to automate and improve point-cloud processing time.

## Methods

### 1. Dataset and Pre-processing
#### Standard 3D Indoor Space Dataset
We use the S3DIS dataset which contains 3D room scenes for semantic segmentation[3]. It contains point clouds of 276 rooms in 6 buildings, where each point is labeled in one of the 13 categories (wall, chair, table etc.).

<h1 align="center">
<img src="https://user-images.githubusercontent.com/27985242/124848554-17e04300-df6b-11eb-9668-74bce54c04ea.png" width="100%" /> 
</h1>
<p align = "center">
Figure 2. S3DIS dataset [3]
</p>

#### Pre-processing 
 To clean up this dataset, all points are rounded to first decimal place and downsampled to reduce the size of overall point cloud data. Additionally, all points are centered to align coordinates of point cloud objects. It's a very large dataset, so for the scope of this project, we are focusing on the first 2 buildings. Based on the ground truth provided, we prepare labeled point clouds for both semantic and instance segmentation (figure 3). 

 <h1 align="center">
<img src="https://user-images.githubusercontent.com/27985242/124848594-2fb7c700-df6b-11eb-9cb9-57546e23a1f5.png" width="100%" /> 
</h1>
<p align = "center">
Figure 3. A sample of (a) semantic label, (b) instance label. Note that the same objects like a wall and a chair in the semantic label is distinguished and separated in the instance label.
</p>

#### Feature Selections
Since a point cloud contains only 6 features of xyzrgb, we do not confine pre-processing methods as feature reduction like PCA. Additionally, we increase features reflecting geometry natures, such as surface normals and curvature.

### 2. Point Cloud Clustering via Unsupervised Learning

#### Semi-Supervised K-Means and GMM
We attempted K-Means and GMM clustering with our centered dataset preprocessed in several different ways: 1) the original data, 2) the normalized data, 3) the data with 3 PCA components, 4) the normalized data with 3 PCA components. We processed and clustered the data one room at a time to relate the points in each room to eachother and simplify the clustering task. For each room, we tried several different values of K and chose the one that resulted in the highest homogeneity with the ground-truth labels. The reason we clustered on homogeneity as opposed to Mutual Information or the Rand Index is because the points in the s3dis dataset are labeled by classes that can each correspond to multiple objects that would not necessarily be clustered together. For example, a chair on one end of a room would probably not be clustered with a chair on the other end, though they would both be labeled as a chair. Mutual Information and the Rand Index penalize clustering that divides classes, but we care more about how well clusters are divided into classes than about whether classes are divided into different clusters. We limited the number of clusters to between the number of ground-truth labels present in that room to double that number. We want to ensure that each label is theoretically represented at least once, and we limited the number to double the number of ground-truth labels to limit the time required to perform clustering and prevent super small, trivially homogenous clusters. For a single room, homogeneity scores can be visualized:

<h1 align="center">
<img src="https://user-images.githubusercontent.com/27985242/124848627-3c3c1f80-df6b-11eb-9ca2-4e5b72ce0ada.png" width="80%" /> 
</h1>
<p align = "center">
Figure 4. Homogeneity from 3 to 20 clusters for Area 1 Conference Room 1
</p>

Then we calculated the average homogeneity and found that for both K-Means and GMM, the most homogenous (43% for K-Means and 65% for GMM) clusters resulted from using the normalized data. To further analyze the normalized data clustering, we found the proportions of all ground-truth labels in each cluster in the dataset and graphed those percentages to show how strongly the ground-truth labels were clustered:

<h1 align="center">
<img src="https://user-images.githubusercontent.com/27985242/124848657-4a8a3b80-df6b-11eb-8b0e-53d2fea3905d.png" width="80%" /> 
</h1>
<p align = "center">
Figure 5. K-Means Homogeneity Analysis
</p>

<h1 align="center">
<img src="https://user-images.githubusercontent.com/27985242/124848676-5413a380-df6b-11eb-9007-0b9c9ed602bb.png" width="80%" /> 
</h1>
<p align = "center">
Figure 6. GMM Homogeneity Analysis
</p>

The most representative ground-truth label for each cluster forms roughly 40-100% of that cluster, with many forming over 80% of their cluster. Ground-truth labels are significantly correlated with clusters, supporting our 65% average homogeneity score. 
We also did a visual comparison of ground-truth labels to cluster labels for 3 rooms:

<h1 align="center">
<img src="https://user-images.githubusercontent.com/27985242/124848709-6261bf80-df6b-11eb-8d06-225ab2e44a20.png" width="80%" /> 
<img src="https://user-images.githubusercontent.com/27985242/124848737-6db4eb00-df6b-11eb-9a09-95f8335528fa.pngg" width="80%" /> 
</h1>
<p align = "center">
Figure 7. Area 1 Conference Room 1
</p>

The main features of the room are visible in all three images, showing the correlation between the clustering and the ground-truth labels. We can see that the KMeans clusters are more patchy; the GMM clusters more closely approximate the ground-truth labeling. In particular, the roof, yellow bars, orange table and chairs, and most of the doorframe are delineated by GMM where in K-Means they are broken up by many different clusters.

<h1 align="center">
<img src="https://user-images.githubusercontent.com/27985242/124848761-7b6a7080-df6b-11eb-9461-331412f9603c.png" width="80%" /> 
</h1>
<p align = "center">
Figure 8. Area 1 WC 1
</p>

Again, note that both K-Means and GMM have clearly delineated stall lines, though again, GMM's lines are cleaner.

<h1 align="center">
<img src="https://user-images.githubusercontent.com/27985242/124848785-858c6f00-df6b-11eb-91ac-ad55ef412e24.png" width="80%" /> 
</h1>
<p align = "center">
Figure 9. Area 2 Auditorium 1
</p>

Again, the lines under the ceiling are visible in all three renderings, and GMM provides better definition to the walls than K-Means.

### 3. Point Cloud Semantic Segmentation via Supervised Learning 
Supervised learning will be used to classify different types of objects. We plan to adapt the graph based deep learning method Dynamic Graph CNN (DGCNN) and PointNet to classify objects in a building dataset. PointNet will first be used to detect and remove large objects such as floors, ceilings, and walls from the building dataset. Following this, we will use DGCNN to detect small objects inside the building to learn more about point features.

As deep learning for 3D point cloud classification is an active research area, we want to compare DGCNN with conventional methods [8]. Therefore, as a point of comparison, we also plan to evaluate models using Random Forest.

#### Results for DGCNN and Random Forest
The DGCNN model is trained and tested with 100 room samples from S3DIS dataset. Each test set and training set contains different rooms, and all data is labeled (default semantic segmentation labels). The following plots show our training and validation results. By comparing the training and validation, training loss is converged to minimum point and the validation loss shows fluctuating results after the loss is converged to minimum point. Training and validation accuracy both improved as the number of training and validation epochs increased. Intersection of Union (IoU) is used to evaluate performance of the deep learning models.

Training Accuracy             |  Training Loss
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/27985242/124848816-9b019900-df6b-11eb-8481-78c12b8c41a6.png)  |  ![](https://user-images.githubusercontent.com/27985242/124848856-b1a7f000-df6b-11eb-87f5-24b892118f4a.png)

Validation Accuracy             |  Validation Loss
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/27985242/124848889-bec4df00-df6b-11eb-9118-8b9bc1eaa27d.png)  |  ![](https://user-images.githubusercontent.com/27985242/124848890-bec4df00-df6b-11eb-9a67-ae48dc22cfb6.png)

Training IoU             |  Validation IoU
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/27985242/124848951-d8febd00-df6b-11eb-8b27-baf3c51eb69b.png)  |  ![](https://user-images.githubusercontent.com/27985242/124848922-cdab9180-df6b-11eb-97ea-03646f8c28b5.png)

 Visual representations in point cloud data format are presented as follows. The left images are all ground truth labels and the right images are all predictions by our deep learning models.

Hallway (Ground Truth)   |  Hallway (Prediction)
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/27985242/124848975-e5831580-df6b-11eb-8a04-fa2e468a8b82.png)  |  ![](https://user-images.githubusercontent.com/27985242/124848976-e61bac00-df6b-11eb-9502-28ed1e27de6a.png)

Room (Ground Truth)   |  Room (Prediction)
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/27985242/124848977-e6b44280-df6b-11eb-8ba3-f9d1f30ec597.png)  |  ![](https://user-images.githubusercontent.com/27985242/124848978-e6b44280-df6b-11eb-8a98-7314bac2985a.png)

Office (Ground Truth)   |  Office (Prediction)
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/27985242/124848979-e6b44280-df6b-11eb-86de-b0698fa837b2.png)  |  ![](https://user-images.githubusercontent.com/27985242/124848980-e74cd900-df6b-11eb-9e9a-d8e52c3f3d86.png)

#### Random Forest
As a point of comparison to DGCNN, we have used random forest for 3D point cloud classification. To do this, features were extracted from the point cloud xyz coordinates and RGB values to train the random forest model. Below are the following features that were extracted and used:
1. eigenvalues
2. surface normals
3. curvature
4. anisotropy
5. eigensum
6. linearity
7. omnivariance
8. planarity
9. sphericity
10. RGB intensity

A parameter study was done on the number of k-nearest neighbors used to calculate the features and the number of trees used in the forest to find the best configuation. A study on feature importance was also done using mean decrease in impurity (MDI) and permutation.

<h1 align="center">
<img src="https://user-images.githubusercontent.com/27985242/124849084-1d8a5880-df6c-11eb-93e9-083ed5c4d8aa.png" width="50%" /> 
</h1>
<p align = "center">
Figure 10. Accuracy vs. KNNs
</p>

<h1 align="center">
<img src="https://user-images.githubusercontent.com/27985242/124849086-1d8a5880-df6c-11eb-9187-239005727c71.png" width="50%" /> 
</h1>
<p align = "center">
Figure 11. Accuracy vs. # Trees in Forest
</p>

<h1 align="center">
<img src="https://user-images.githubusercontent.com/27985242/124849082-1d8a5880-df6c-11eb-850e-3ccb2b306f95.png" width="50%" /> 
</h1>
<p align = "center">
Figure 12. Feature Importance using MDI
</p>

<h1 align="center">
<img src="https://user-images.githubusercontent.com/27985242/124849083-1d8a5880-df6c-11eb-84d1-7d0b074a90a1.png" width="50%" /> 
</h1>
<p align = "center">
Figure 13. Feature Importance using Permutation
</p>

## Next Steps
After finishing the instance labels and feature selection tasks, we plan to perform clustering using those labels. We also plan to include clustering labels as a feature for supervised learning to see if we can improve our accuracy. 

## References
[1] D. Maturana and S. Scherer, “Voxnet: A 3d convolutional neural network for real-timeobject recognition,” in2015 IEEE/RSJ International Conference on Intelligent Robotsand Systems (IROS), pp. 922–928, IEEE, 2015.  
[2] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning on point sets for 3dclassification and segmentation,” inProceedings of the IEEE conference on computervision and pattern recognition, pp. 652–660, 2017.  
[3] I. Armeni, O. Sener, A. R. Zamir, H. Jiang, I. Brilakis, M. Fischer, and S. Savarese, “3dsemantic parsing of large-scale indoor spaces,” inProceedings of the IEEE Conferenceon Computer Vision and Pattern Recognition, pp. 1534–1543, 2016.  
[4] S. Chen, C. Duan, Y. Yang, D. Li, C. Feng, and D. Tian, “Deep unsupervised learning of3d point clouds via graph topology inference and filtering,”IEEE Transactions on ImageProcessing, vol. 29, pp. 3183–3198, 2019.  
[5] Y. Yang, C. Feng, Y. Shen, and D. Tian, “Foldingnet: Interpretable unsupervised learningon 3d point clouds,”arXiv preprint arXiv:1712.07262, vol. 2, no. 3, p. 5, 2017.  
[6] J. M. Biosca and J. L. Lerma, “Unsupervised robust planar segmentation of terrestri-al laser scanner point clouds based on fuzzy clustering methods,”ISPRS Journal ofPhotogrammetry and Remote Sensing, vol. 63, no. 1, pp. 84–98, 2008.  
[7] L. Zhang and Z. Zhu, “Unsupervised feature learning for point cloud understandingby  contrasting  and  clustering  using  graph  convolutional  neural  networks,”  in2019International Conference on 3D Vision (3DV), pp. 395–404, IEEE, 2019.  
[8] S.-L. F. R.-P. J. Cabo C, Ordóñez C and C.-J. AJ, “Multiscale supervised classificationof point clouds with urban and forest applications,”Sensors(Basel), vol. 19, 2019.3. Chen, Jingdao, Zsolt Kira, and Yong K. Cho. "LRGNet: Learnable Region Growing for Class-Agnostic Point Cloud Segmentation." IEEE Robotics and Automation Letters 6.2 (2021): 2799-2806.  
[9] Chen, J., Kira, Z., & Cho, Y. K. (2021). LRGNet: Learnable Region Growing for Class-Agnostic Point Cloud Segmentation. IEEE Robotics and Automation Letters, 6(2), 2799-2806.
***

## Timeline
`June 14`
* project proposal due date.

`June 18`
* prepare, clean up, and preprocess the s3dis dataset. 
* work on implementing unsuperviesed learning and supervised learning algorithm.

`June 25`
* Complete implementing algorithm and validate results using small dataset. s
* start implementing evaluation metrics and visualization code.

`July 7 - Project midpoint report`  
* validate model using a large dataset and optimize the algorithm.
* start preparing base line models to compare the result against our proposed method.
* start writing a final report and prepare final presentation.
* complete implementing evaluation metrics.

`August2 - Final prohect due date`  
* record final presentation video.
* complete final report.
* clean up and organize github repository.  
  
## Team Members' Responsibility 
`Kaylah Facey`
* unsupervised learning method
* evaluation metrics
* helper function to visualize point cloud results

`Madison Manley`
* point cloud classification algorithm using Random Forest method
* evaluation metircs and optimizing model

`Seongyong Kim`
* raw data processing
* unsupervised learning method

`Yosuke Yajima`
* raw data processing
* DGCNN and evaluation metrics
