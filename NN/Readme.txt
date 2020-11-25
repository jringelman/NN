README FOR NNApp1.java

This application runs either MNIST image data or SP500 MARKET data through a NeuralNetwork 

Pass MNIST or MARKET as application parameter to run either data set. (default is MNIST)

Examples:
   java NNApp1.java MNIST
   java NNApp1.java MARKET


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

