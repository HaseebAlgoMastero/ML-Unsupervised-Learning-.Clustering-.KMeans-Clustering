# ML=>Unsupervised=>Learning=>Clustering=>KMeans-Clustering
<h2>K-Means Clustering Algorithm</h2>
K-Means Clustering is an unsupervised learning algorithm that is used to solve the clustering problems in machine learning or data science. In this topic, we will learn what is K-means clustering algorithm, how the algorithm works, along with the Python implementation of k-means clustering.

<h2>What is K-Means Algorithm?</h2>
K-Means Clustering is an Unsupervised Learning algorithm, which groups the unlabeled dataset into different clusters. Here K defines the number of pre-defined clusters that need to be created in the process, as if K=2, there will be two clusters, and for K=3, there will be three clusters, and so on.

It is an iterative algorithm that divides the unlabeled dataset into k different clusters in such a way that each dataset belongs only one group that has similar properties.
It allows us to cluster the data into different groups and a convenient way to discover the categories of groups in the unlabeled dataset on its own without the need for any training.

It is a centroid-based algorithm, where each cluster is associated with a centroid. The main aim of this algorithm is to minimize the sum of distances between the data point and their corresponding clusters.


The algorithm takes the unlabeled dataset as input, divides the dataset into k-number of clusters, and repeats the process until it does not find the best clusters. The value of k should be predetermined in this algorithm.

<h2>The k-means clustering algorithm mainly performs two tasks:</h2>
<ul>
<li>Determines the best value for K center points or centroids by an iterative process.</li>
<li>Assigns each data point to its closest k-center. Those data points which are near to the particular k-center, create a cluster.
Hence each cluster has datapoints with some commonalities, and it is away from other clusters.</li>
</ul>
<h2>The below diagram explains the working of the K-means Clustering Algorithm:</h2>

<img src = "https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning.png">


<h2>How does the K-Means Algorithm Work?</h2>
The working of the K-Means algorithm is explained in the below steps:

<h2>Step-1:</h2> Select the number K to decide the number of clusters.

<h2>Step-2:</h2> Select random K points or centroids. (It can be other from the input dataset).

<h2>Step-3:</h2> Assign each data point to their closest centroid, which will form the predefined K clusters.


<h2>Step-4:</h2> Calculate the variance and place a new centroid of each cluster.

<h2>Step-5:</h2> Repeat the third steps, which means reassign each datapoint to the new closest centroid of each cluster.

<h2>Step-6:</h2> If any reassignment occurs, then go to step-4 else go to FINISH.

<h2>Step-7:</h2> The model is ready.

<h2>Let's understand the above steps by considering the visual plots:</h2>
Suppose we have two variables M1 and M2. The x-y axis scatter plot of these two variables is given below:
<img src = "https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning2.png">
Let's take number k of clusters, i.e., K=2, to identify the dataset and to put them into different clusters. It means here we will try to group these datasets into two different clusters.
We need to choose some random k points or centroid to form the cluster. These points can be either the points from the dataset or any other point. So, here we are selecting the below two points as k points, which are not the part of our dataset. Consider the below image:
<img src = "https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning3.png">
Now we will assign each data point of the scatter plot to its closest K-point or centroid. We will compute it by applying some mathematics that we have studied to calculate the distance between two points. So, we will draw a median between both the centroids. Consider the below image:
<img src = "https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning4.png">
From the above image, it is clear that points left side of the line is near to the K1 or blue centroid, and points to the right of the line are close to the yellow centroid. Let's color them as blue and yellow for clear visualization.
<img src = "https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning5.png">
As we need to find the closest cluster, so we will repeat the process by choosing a new centroid. To choose the new centroids, we will compute the center of gravity of these centroids, and will find new centroids as below:
<img src = "https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning6.png">
Next, we will reassign each datapoint to the new centroid. For this, we will repeat the same process of finding a median line. The median will be like below image:
<img src = "https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning7.png">
From the above image, we can see, one yellow point is on the left side of the line, and two blue points are right to the line. So, these three points will be assigned to new centroids.
<img src = "https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning8.png">

As reassignment has taken place, so we will again go to the step-4, which is finding new centroids or K-points.
<ul>
<li>We will repeat the process by finding the center of gravity of centroids</li>
<li>As we got the new centroids so again will draw the median line and reassign the data points</li>
<li>We can see in the above image; there are no dissimilar data points on either side of the line, which means our model is formed. </li>
<li>As our model is ready, so we can now remove the assumed centroids</li>
</ul>
<h2>How to choose the value of "K number of clusters" in K-means Clustering?</h2>
The performance of the K-means clustering algorithm depends upon highly efficient clusters that it forms. But choosing the optimal number of clusters is a big task. There are some different ways to find the optimal number of clusters, but here we are discussing the most appropriate method to find the number of clusters or value of K. The method is given below:

<h2>Elbow Method</h2>
The Elbow method is one of the most popular ways to find the optimal number of clusters. This method uses the concept of WCSS value. WCSS stands for Within Cluster Sum of Squares, which defines the total variations within a cluster. The formula to calculate the value of WCSS (for 3 clusters) is given below:

<h2>WCSS= ∑Pi in Cluster1 distance(Pi C1)2 +∑Pi in Cluster2distance(Pi C2)2+∑Pi in CLuster3 distance(Pi C3)2
In the above formula of WCSS,</h2>

<h2>∑Pi in Cluster1 distance(Pi C1)2: It is the sum of the square of the distances between each data point and its centroid within a cluster1 and the same for the other two terms.</h2>

<h2>To measure the distance between data points and centroid, we can use any method such as Euclidean distance or Manhattan distance.</h2>

<h2>To find the optimal value of clusters, the elbow method follows the below steps:</h2>
<ul>
<li>It executes the K-means clustering on a given dataset for different K values (ranges from 1-10).</li>
<li>For each value of K, calculates the WCSS value.</li>
<li>Plots a curve between calculated WCSS values and the number of clusters K.</li>
<li>The sharp point of bend or a point of the plot looks like an arm, then that point is considered as the best value of K.</li>
<li>Since the graph shows the sharp bend, which looks like an elbow, hence it is known as the elbow method. The graph for the elbow method looks like the below image:</li>
</ul>
<img src ="https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning13.png">

<h2>Note:</h2> We can choose the number of clusters equal to the given data points. If we choose the number of clusters equal to the data points, then the value of WCSS becomes zero, and that will be the endpoint of the plot.

<h2>Note:</h2> For Python Implementation See the repository
<h2>Best Of Luck</h2>
