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
      