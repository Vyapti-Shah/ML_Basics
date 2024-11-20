#Rainbow CSV extension
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#'''download iris.csv from https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'''
#Load iris.csv into pandas dataFrame
iris = pd.read_csv("datas\iris.csv")

#How many data-points and features
print(iris.shape)
#what are the column names in the 
print(iris.columns)
#how many data-points for each class are present? or how many flowers for each species are present?
iris["species"].value_counts()
#balanced datasets vs imbalanced datasets
#Iris is a balanced datasets as the number of data-points for every class is equal

#2D scatter plot
#ALWAYS understand the axis: labels ans scale
iris.plot(kind="scatter",x="sepal_length",y="sepal_width")
plt.show()
#cannot make much sense out it
#What if we colour the points by their class_label/flower_type

#2-D scatter colour coding for each flower_type/class
#Here sns corresponds to seaborn
sns.set_style("whitegrid")
sns.FacetGrid(iris,hue="species").map(plt.scatter, "sepal_length","sepal_width").add_legend() #size=4
plt.show()
#Notice that the blue plot can be easily separated from the red and green by drawing line
#But red and green lines cannot be separated
#How many combinations exist ? 4C2 = 6
#Observation = Using sepal_length and sepal_width features, we can distinguish setosa flowers from others.
#              Separating Vernica and Vericolor is much harder as they have considerable overlap

#Pair Plot = pairwise scatter plot
#Disadvantages = Can be used when the number of features are high
#                Cannot visualize higher dimensional patter in 3D and 4D for which we learn principle component analysis(PCA) or t-SNE
#                Only possible to 2D view
plt.close()
sns.set_style("whitegrid")
sns.pairplot(iris,hue="species",size=3)
plt.show()
#NOTE: The diagonal elements are PDFs for each feature
#Observation = petal_length and petal_width are the most useful features to identify various flower types.
#              While setosa can be easily identified (linearly separable), Vernica and Vericolor have some overlap (almost linearly separable)
#              We can find "lines" and "if-else" conditions to build a simple model to classify the flower types.

#Univarient Analysis usinf PDF EDA #HISTOGRAM
sns.FacetGrid(iris,hue="species").map(sns.displot, "petal_length").add_legend() #size=5
plt.show()
sns.FacetGrid(iris,hue="species").map(sns.displot, "sepal_length").add_legend() #size=5
plt.show()
sns.FacetGrid(iris,hue="species").map(sns.displot, "petal_width").add_legend() #size=5
plt.show()
sns.FacetGrid(iris,hue="species").map(sns.displot, "sepal_width").add_legend() #size=5
plt.show()

# Filter species
iris_setosa = iris[iris['species'] == 'setosa']
iris_virginica = iris[iris['species'] == 'virginica']
iris_versicolor = iris[iris['species'] == 'versicolor']

#1-D scatter plot using just one feature
#1-D scatter plotof petal length
#Very hard to make sense as points are overlapping a lot
plt.plot(iris_setosa["petal_length"],np.zeros_like(iris_setosa['petal_length']))
plt.plot(iris_virginica["petal_length"],np.zeros_like(iris_virginica['petal_length']))
plt.plot(iris_versicolor["petal_length"],np.zeros_like(iris_versicolor['petal_length']))
plt.show()

#Mean, Variance and Std-deviation
print ("Means:")
print ("Setosa Mean Petal Length:", np.mean(iris_setosa["petal_length"]))
#Mean with an outlier.
print("Setosa Mean with Outlier:", np.mean(np.append(iris_setosa["petal_length"],50))) #here an outlier comes as the n is 50 but due 
                                #to error someone enters 51 then to avoid error we put the 50 elemnts and state the 51st element as 50
print("Virginica Mean Petal Length:", np.mean(iris_virginica["petal_length"]))
print ("Versicolor Mean Petal Length:", np.mean (iris_versicolor["petal_length"]))
print ("\nStd-dev:")
print("Setosa Std-dev Petal Length:", np.std(iris_setosa["petal_length"]))
print("Virginica Std-dev Petal Length:", np.std(iris_virginica["petal_length"]))
print ("Versicolor Std-dev Petal Length:", np.std(iris_versicolor["petal_length"]))

#Median, Quantiles, Percentiles, IQR
print ("Median:")
print ("Setosa Median Petal Length:", np.median(iris_setosa["petal_length"]))
#Mean with an outlier.
print("Setosa Median with Outlier:", np.median(np.append(iris_setosa["petal_length"],50)))
print("Virginica Median Petal Length:", np.median(iris_virginica["petal_length"]))
print ("Versicolor Median Petal Length:", np.median(iris_versicolor["petal_length"]))

print ("\nQuantiles:")
print ("Setosa Percentile Petal Length:", np.percentile((iris_setosa["petal_length"]),np.arange(0,100,25)))
print("Virginica Percentile Petal Length:", np.percentile(((iris_virginica["petal_length"])),np.arange(0,100,25)))
print("Versicolor Percentile Petal Length:", np.percentile((iris_versicolor["petal_length"]),np.arange(0,100,25)))

print("\n90th Percentiles:")
print ("Setosa Percentile Petal Length:", np.percentile((iris_setosa["petal_length"]),90))
print("Virginica Percentile Petal Length:", np.percentile((iris_virginica["petal_length"]),90))
print("Versicolor Percentile Petal Length:", np.percentile((iris_versicolor["petal_length"]),90))

#Box-plot with whiskers: another method of visualizing the 1-D scatter plot
#The Concept of median, percentile, quantile.
#How to draw the box in the box-plot?
#How to draw whiskers: (no standard way) Could use min and max or use othe
#IOR Like idea.
#NOTE: IN the plot below, a technique call inter-quartile range is used in
#Whiskers in the plot below donot correpoand to the min and max values.
#Box-plot can be visualized as a PDF on the side-ways. 
sns.boxplot(x='species', y='petal_length', data=iris) 
plt.show()

#Violin Plot = A violin plot combines the benefits of the previous two plots and simplifies them
#Denser regions of the data are fatter, and sparser ones thinner in a violin plot
sns.violinplot(x="species",y="petal_length",data=iris) #size=8
plt.show()

#2-D Density plot, contors-plot
sns.jointplot(x="petal_length",y="petal_width",data=iris_setosa,kind="kde")
plt.show()

from statsmodels import robust
print("\nMedian Absolute Deviation:")
print(robust.mad(iris_setosa['petal_length']))
print(robust.mad(iris_setosa['petal_length']))
print(robust.mad(iris_setosa['petal_length']))

#Need for Cumulative Distribution Functioin (CDF)
#We can visually see what percentage of versicolor flowers have a petal_length of less than 1.6?
#Plot CDF of petal_length
counts, bin_edges = np.histogram(iris_setosa['petal_length'],bins=10,density=True)
pdf = counts/(sum(counts)) #probability density function
print(pdf)
print(bin_edges)
#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()

counts, bin_edges = np.histogram(iris_setosa['petal_length'],bins=10,density=True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
counts, bin_edges = np.histogram(iris_setosa['petal_length'],bins=20,density=True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf)
plt.show()
