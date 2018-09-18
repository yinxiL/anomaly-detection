# Anomaly Detection
This review refers to these two papers: [CHANDOLA](http://cucis.ece.northwestern.edu/projects/DMS/publications/AnomalyDetection.pdf) and [陈斌](http://gxbwk.njournal.sdu.edu.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=1957) 

## Research Status
Since the abnormal data is usually very small, the current anomaly detection method is mostly from the construction of the hypothetical model, even though we're not doing noise removal. 

Besides, as this area has been fully explored, researchers are mostly exploring the nature of abnormal production, adding priori information based on data characteristics to develop special algorithms in specific application areas, or optimizing in the context of big data and real-time networking.

### Challenges
- Encompass every normal behavior
- Malicions may appear normal
- "Normal" keeps envolving
- Differ from domains
- The availability of data
- Data contains noise

### Type of Anomaly
- Point 
- Contextual: Each data instance has contextual and behavior attributes. eg. time series or spatial data 
- Collective: Occurrence together as a collection is anomalous. eg. sequence, graph or spatial data

## Methods

### Classification
pros: 
* can distinguish between classes
* testing phase is easy

cons:
* rely on labels
* no probabilistic score
#### Neural Networks-Based
Replicator Neural Networks (Hawkins et al. 2002; Williams et al. 2002)
#### Bayesian Networks-Based
- smoothed zero probabilities using Laplace smoothing
- capture the conditional dependencies between the different attributes(Siaterlis and Maglaris 2004; Janakiram et al. 2006; Das and Schneider 2007)

#### Support Domain
- SVM <=> slab SVM <=> SV data description
- minimum enclosing ellipsoid
#### Rule-Based
- RIPPER
- Decision Trees

### Set Density Threshold
pros: 
* unsupervised
* rarely miss anomaly
* easy to migrate

cons:
* has requirements for data structures and volume
* testing phase is challange
* perform rely on diatance measures

Passing parameters or non-parametric methods to estimate the density model of training samples and set the density threshold. If a test data is smaller than the threshold, it is considered abnormal.

The easiest parameter method is Gaussian One-dimensional distribution, from which developed Gaussian Multi-element model and [Gaussian Mixed Model](https://pdfs.semanticscholar.org/eef1/bb217a8235643318e38122605a8ca5d1d07a.pdf), the problem with this method is that γ is hard to chose and more samples are needed to overcome dimensional disasters.There are also other regression modes like ARIMA and ARMA.

KNN is a typical example of a nonparametric method, and [Kanamori](https://papers.nips.cc/paper/3387-efficient-direct-density-ratio-estimation-for-non-stationarity-adaptation-and-outlier-detection.pdf) improve its performace under high dimensional finite samples by estimating the importance of the sample using its probability density function on the testset.

### Refactoring
#### Information Theoretic Anomaly
> analyze the information content of a data set using different information theoretic measures such as Kolomogorov Complexity, entropy,
relative entropy, Local Search Algorithm，and so on.
#### Spectral techniques
>  try to find an approximation of the data using a combination of attributes that capture the bulk of the variability in the data.
- Compact Matrix Decomposition
- two main method based on PCA:   
[Transform nonlinear problems into linear problems](http://www-isl.stanford.edu/~cover/papers/paper2.pdf) and compute the principle components in high dimensional space like [KPCA](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis) 

[Principle curves](https://web.stanford.edu/~hastie/Papers/Principal_Curves.pdf)

## Difficulty
### Dealing with Data Imbalance
#### Artificially generated sample
- The origin of coordinate [KFDA](http://read.pudn.com/downloads190/ebook/893343/Kernel_Methods_for_Pattern_Analysis/0521813972.pdf)
- Evenly distributed [Steinwart](http://www.jmlr.org/papers/volume6/steinwart05a/steinwart05a.pdf)
- Boundary [Banhalmi](https://link.springer.com/content/pdf/10.1007%2F978-3-540-74958-5_51.pdf)
#### Existing abnormal sample
- NSVDD
- MEMEM

### Handling Contextual Ananomies
 require that the data has a set of contextual attributes (to define a context), and a set of behavioral attribute
 - Spatial: Lu et al. 2003; Shekhar et al. 2001; Kou et al. 2006; Sun and Chawla 2004
 - Graphs: Sun et al. 2005 
 - Sequential: 
 * Time-series data: Abraham and Chuang 1989; Abraham and Box 1979; Rousseeuw and Leroy 1987; Bianco et al. 2001; Fox 1972; Salvador and
Chan 2003; Tsay et al. 2000; Galeano et al. 2004; Zeevi et al. 1997
 * Web data: Ilgunetal.1995; VilaltaandMa2002; WeissandHirsh 1998; Smyth 1994
 - Profile:  
 * cell-phone fraud detection : Fawcett and Provost 1999; Teng et al. 1990
 * CRM databases: He et al. 2004b
 * credit-card fraud detection: Bolton and Hand 1999
 #### Reduction to Point
  eg. Song et al. 2007
  - identify a context for each test instance using the contextual attributes.
  - compute an anomaly score for the test instance within the context using a known point anomaly detection
technique.

 #### Utillizing the Structure
 modeling the regression as well as correlation between the sequences
 - ARMA and ARIMA
 - FSA
 - HMM
 
 
## Features of Different Application Scenarios
### Intrusion (malicious)
  - Has huge volume data --> Focus on computing efficiency
  - Hot-based (co-occurrence of events)
  - Network (intrustion)
### Frand (criminal in commerxial organizations)
  - credit card: owners' usage history and location
  - mobile phone: calling behavior
  - insurance: manually investigated cases and active monitoring
  - insider tracking (stock markets): as early as possible, temporal association
### Medical
  detect disease --> require high accuracy
### Industrial Damage
  - fault detection
  - structual defect
### Image Processing
  - motion
  - regions
  - large-size input
### Text Data
  - large variations in documents
  - topics/events/stories
### Sensor
  - itself
  - event


