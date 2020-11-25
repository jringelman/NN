README FOR NNApp1.java

This application runs either MNIST numeric image data or SP500 historical MARKET data through a NeuralNetwork.
Both data sets are included with this project. 

MNIST DATA
The 4 data files for MNIST are stored here:  ./data/mnist/
	t10k-images-idx3-ubyte.gz
	t10k-labels-idx1-ubyte.gz
	train-images-idx3-ubyte.gz
	train-labels-idx1-ubyte.gz
(THE MNIST DATABASE of handwritten digits : http://yann.lecun.com/exdb/mnist/)

MARKET DATA
Historical data for SP500 for MARKET is stored here: ./data/market/
	- sp500 1927-12 to 2020-11-23.csv

TO COMPILE APPLICATION
- download and unzip project
- open terminal and navigate to : NN-master/NN/
- Run java compiler:
      javac -d ./bin/ -cp ./bin/ ./src/jmr/util/*.java ./src/jmr/nn/*.java 

TO RUN APPLICATION
Pass MNIST or MARKET as application parameter to run either data set. (default is MNIST)
Examples:
	java -cp ./bin/ jmr.nn.NNApp1 MNIST
	java -cp ./bin/ jmr.nn.NNApp1 MARKET
      
     
      
CLASSES OVERVIEW

APPLICATION CLASS
	- jmr.nn.NNApp1 (has main function)

CORE NEURAL NETWORK CLASSES
	- jmr.nn.NeuralNetwork (contains 1 or more NNLayers)
	- jmr.nn.NNLayer (contains 1 or more Neurons)
	- jmr.nn.Neuron
	
DATA LOADING CLASSES
	- jmr.nn.MnistReader (reads mnist data and loads into NeuralNetwork for training and testing)
	- jmr.nn.MarketData  (reads historical SP500 price data and loads into NeuralNetwork for training and testing)
	
UTILITY CLASSES
	- jmr.util.ArrayUtil (helper class for double arrays)
	- jmr.util.StdOut (used for formatted printing)
	- jmr.util.AppProperties (class to load from properties file) (NOT USED IN THIS VERSION)
	- jmr.util.Log (log file class) (NOT USED IN THIS VERSION)
		 	
      