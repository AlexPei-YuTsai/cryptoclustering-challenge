# Unsupervised Learning Challenge
> How can we use algorithms to bundle, group, and classify things easily?

## Folder Contents
- A `Resources` folder containing all the data we'll be using for this exercise.
- A `Visuals` folder containing most of the Bokeh charts that can't be previewed on Github (reproduced here in the ReadMe instead)
- A `.gitignore` file that ignores common things like PyCache, Jupyter Notebook checkpoints, and other common gitignorable Python entities. 
- A main `Crypto_Clustering` Jupyter Notebook file that uses the Scikit-Learn module to let us learn about any hidden patterns in our data.

### Installation/Prerequisites
- Make sure you can run Python. The development environment I used was set-up with:
```
conda create -n dev python=3.10 anaconda -y
```

#### Imported Modules
- Installing via the conda command given should give you access to most, if not all, of the script's modules locally. However, if you don't have them, be sure to grab yourself the following libraries:
  - [Pandas](https://pandas.pydata.org/docs/getting_started/install.html) for basic data management
  - [HVPlot](https://hvplot.holoviz.org/getting_started/installation.html) for basic visualizations
  - [Scikit-Learn](https://scikit-learn.org/stable/install.html) for the unsupervised learning algorithms we'll be using

## Code Breakdown
The premise of this challenge is simple: Given data about performance changes in different timeframes, how much can we learn about our cryptocurrencies as a whole? Overall, the machine learning code used will look like this:
```python
# Initialize a machine learning model
model = someAlgorithm(parameter1 = x1, parameter2 = x2, ...)

# Fit the model to your data
model.fit(yourDataHere)

# Predict unknown results based on your trained model
predictions = model.predict(yourDataHere)
```

### Some Findings
Depending on how your data looks, you might also need scaling. Take this data range chart for example: Our outliers was about 8,000 units away from the mean.

![Data range - Unscaled](https://cdn.discordapp.com/attachments/1107347677831778364/1157682177908146367/currency_unscaled.png?ex=65197f0b&is=65182d8b&hm=d385057c3d0e5807caec041cd4a9a751dc2297d6e292ba4abdd92cc6948b160a&)

After scaling it, though the outliers still stick out like a sore thumb, the difference is now a much more manageable number that won't immediately ruin an average-based algorithm like KMeans and PCA.

![Data range - Scaled](https://cdn.discordapp.com/attachments/1107347677831778364/1157682178210144327/currency_scaled.png?ex=65197f0b&is=65182d8b&hm=888bd8b9a8535eea7781151d2200b57742de08d30bd4dcd18bd814ff3837307c&)

To figure out how many clusters we should have, we used the "elbow plot" inertia-based heuristic on both our scaled data and our scaled data with PCA dimensionality reduction done. By brute-forcing through a number of clusters, this method helps us decide a good number of groups to use. In both cases, that number turned out to be `k=4`, but the inertia on the data with PCA done is generally lower than that of the regular unscaled data.

![Elbow plots](https://cdn.discordapp.com/attachments/1107347677831778364/1157682180051439636/both_elbow.png?ex=65197f0c&is=65182d8c&hm=2254ed929232c6418b1ec0e64ba39a88d61cb15a18f458d4f2488f069af68590&)

Finally, we did KMeans on our data to see how our data looks. Notably, cluster delineations are much more distinct on the PCA plot.

![PCA clusters](https://cdn.discordapp.com/attachments/1107347677831778364/1157682179736875008/pca_clusters.png?ex=65197f0c&is=65182d8c&hm=71dde32a17063141b9fd49e240c5ace214c856b5b44cbe9157a1e24f891f4288&)

To summarize what these clusters are, the yellow and red dots are one-dot clusters as they are the outliers of the dataset. Those dots correspond with the massive spikes in the data range charts previously shown. PC1 scales more with long term (more than 200 days) changes in value while PC2 scales more with monthly to bimonthly growth in value. The blue group denotes those that performed well in those monthly periods while the black group denotes those that did not. Through PCA, we can get a high-level view of any patterns data may have.

## Resources that helped a lot
We aren't coding any of the machine learning algorithms from scratch. There's no need to reinvent the wheel or rediscover calculus for the purposes of this exercise. However, it's still important to learn about how the algorithms work and when these can be applied. I found these theory videos to be very useful:
- Josh Starmer's [KMeans](https://www.youtube.com/watch?v=4b5d3muPQmA) and [PCA](https://www.youtube.com/watch?v=FgakZw6K1QQ) explanations very easily demystify what the computer does in order to bucket similar things into groups and how we can let the computer bucket things more appropriately.
- Cassie Kozyrkov's [Making Friends with Machine Learning](https://www.youtube.com/watch?v=1vkb7BCMQd0) 6-hour course is also great for giving people a look into the black boxes that now govern our data-centric world.

Most of the coding tricks can be found with Google and the official documentation for [SKLearn's Clustering functions](https://scikit-learn.org/stable/modules/clustering.html#clustering), so I didn't consult any tutorials. However, it's such a popular topic that Youtube will probably have at least a few decent resources to use.
- One example is [this video](https://www.youtube.com/watch?v=EItlUEPCIzM) by codebasics

## FINAL NOTES
> Project completed on August 31, 2023
