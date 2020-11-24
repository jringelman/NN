package jmr.nn;

import java.nio.file.Paths;
import java.util.List;
import jmr.util.AppProperties;
import jmr.util.ArrayUtil;
import jmr.util.StdOut;

/* NeuralNetwork is composed of 1 or more NNLayers which are composed of Neurons.
 * There is no input layer. Inputs are simply passed to the first layer
 * The last layer is the output layer. Prior layers are hidden layers.
 * By default, the initial weights for each neuron are randomly generated. For debugging, the weights can be overidden.
 * The neurons use a sigmoid activation function.
 */

public class NeuralNetwork {
	
	NNLayer [] m_aLayer;
	double m_dLearningRate;
	int m_iNbrNNInputs;
	
	//THERE ARE 2 WAYS TO CREATE NeuralNetwork
	
	//METHOD 1 IS TO CREATE THE NNLAYERS THEN PASS THEM IN
	public NeuralNetwork(NNLayer [] aLayers, double dLearningRate){
		m_aLayer =  aLayers;
		m_dLearningRate = dLearningRate;
	}
	
	//METHOD 2 IS TO PASS IN AN ARRAY WITH THE NBR OF LAYERS/NEURONS AND AN ARRAY W/ THE BIASES AND LET NeuralNetwork CREATE THE LAYERS
	//THE aiNbrNeuronsByLayer ARRAY WOULD LOOK LIKE THIS FOR NN  A HIDDEN LAYER W/ 4 NEURONS AND AN OUTPUT LAYER W/ 3 NEURONS
	//	 aiNbrNeuronsByLayer = {4,3}  ==> THE NUMBER OF ELEMENTS IS EQUAL TO NUMBER OF LAYERS, EACH VALUE IS THE NBR NEURONS IN THAT LAYER
	// THE 	adBiasByLayer MUST HAVE SAME NBR OF ELEMENTS e.g. adBiasByLayer	{0.35, 0.60} FOR 2 LAYERS
	//FOR 3 LAYERS, aiNbrNeuronsByLayer & adBiasByLayer WOULD HAVE 3 ELEMENTS  
	//		aiNbrNeuronsByLayer = {5,10,3}  adBiasByLayer = {0.25, 0.3, 0.15}
	public NeuralNetwork (int iNbrNNInputs, int [] aiNbrNeuronsByLayer, double [] adBiasByLayer,  double dLearningRate)	{
		if (aiNbrNeuronsByLayer.length != adBiasByLayer.length) throw new RuntimeException("Mismatch params in NeuralNetwork constructor");

		this.m_aLayer = new NNLayer[aiNbrNeuronsByLayer.length];
		m_dLearningRate = dLearningRate;
		
		int iNbrLayerInputs = iNbrNNInputs;
		for (int iLayer=0; iLayer < aiNbrNeuronsByLayer.length; iLayer++)
		{
			//	public Layer(int nLayerNbr,  int nNbrInputs, int nNbrNeurons, double dBias)
			m_aLayer[iLayer] = new NNLayer(iLayer, iNbrLayerInputs, aiNbrNeuronsByLayer[iLayer], adBiasByLayer[iLayer]);
			iNbrLayerInputs = aiNbrNeuronsByLayer[iLayer];
		}
	}
	
	public void setWeights(int iLayer, double [][] aadWeights)
	{
		m_aLayer[iLayer].setWeights(aadWeights);
	}
	
	public double [] activateNN(double [] adInput, double [] adTarget) {
		double [] adACT = adInput;
		
		//FEEDFORWARD - ACTIVATE EACH LAYER GOING FORWARD
		for (int iLayer=0; iLayer<m_aLayer.length; iLayer++) {
			adACT = m_aLayer[iLayer].activate(adInput);
			adInput = adACT; //INPUT FOR NEXT LAYER IS ACTIVATION OUTPUT FROM PRIOR LAYER.
		}
		return adACT;
	}

	public double [][] trainNetwork(double [] adInput, double [] adTarget){
		double [] adACT = adInput;
		double [][] aadACT = new double [2][];
		
		
		//FEEDFORWARD - ACTIVATE EACH LAYER GOING FORWARD
		for (int iLayer=0; iLayer<m_aLayer.length; iLayer++) {
			adACT = m_aLayer[iLayer].activate(adInput);
			adInput = adACT; //INPUT FOR NEXT LAYER IS ACTIVATION OUTPUT FROM PRIOR LAYER.
			aadACT[iLayer] = adACT.clone();
			//ArrayUtil.show(adACT, "adACT for Layer " + iLayer);
		}
		
		//CALC TOTAL ERROR FOR OUTPUT LAYER
		double dErrorTotal = m_aLayer[m_aLayer.length-1].computeError(adTarget);

		//BACKPROPOGATE OUTPUT LAYER
		//  WITH dET/dACT AS INPUT, EACH LAYER CAN COMPUTE NEW WEIGHTS
		//  AND, EACH LAYER WILL RETURN dET/dACT as input for upstream layer
		//TO START, FOR OUTPUT LAYER, dET/dACT = (ACT-TARGET)
		double [] adETdACT = ArrayUtil.minus(adACT, adTarget);
		
		for (int iLayer=m_aLayer.length-1;iLayer >=0; iLayer--) {
			adETdACT = m_aLayer[iLayer].computeNewWeights(adETdACT, m_dLearningRate);
		}
		return aadACT;
	}

//**************************************************
//************ STATIC TEST METHODS *****************
//**************************************************

	
	public static void testMinstData()
	{
		double dBias = 0.1; 
		final double dLEARNING_RATE = 0.1;
		final int iNBR_INPUTS = 784; //28 x 28 pixels = 784 pixels
		final int iNBR_NEURONS_LAYER0 = 20; // THIS IN THE ONLY HIDDEN LAYER
		final int iNBR_NEURONS_LAYER1 = 10; // THIS IS OUPUT LAYER WITH EACH NODE FOR DIGITS 0 THRU 9;
		final int iNBR_EPOCHS = 1;
	      	
		NNLayer [] aLayer = new NNLayer[2];
		//	public Layer(int nLayerNbr,  int nNbrInputs, int nNbrNeurons, double dBias)
	    aLayer[0] = new NNLayer(0, iNBR_INPUTS, iNBR_NEURONS_LAYER0, dBias);
	    aLayer[1] = new NNLayer(1, iNBR_NEURONS_LAYER0, iNBR_NEURONS_LAYER1, dBias);
	    NeuralNetwork nn = new NeuralNetwork(aLayer, dLEARNING_RATE);

	    //TRAIN THE NN WITH MINST DATA
	    String sMinstDataPath = "";
		try{
			sMinstDataPath = AppProperties.getProperty("dir.mnistdata");
		}catch (Exception e) {System.out.println(e);}

    	String sFileTrainLabels = sMinstDataPath + "train-labels-idx1-ubyte.gz";
    	String sFileTrainImages = sMinstDataPath + "train-images-idx3-ubyte.gz";

    	try{
    		 
    	int[] aiTrainLabels = MnistReader.getLabels(Paths.get(sFileTrainLabels));
    	List<int[][]> listTrainImages = MnistReader.getImages(Paths.get(sFileTrainImages));
    	
		System.out.println(aiTrainLabels.length + " Train Labels");
		System.out.println(listTrainImages.size() + " Train Images");
		System.out.println("Beginning " + iNBR_EPOCHS + " epochs of training");

		for (int iEpoch=0; iEpoch<iNBR_EPOCHS; iEpoch++)
		{
			//System.out.println("\nEpoch: " + iEpoch);
			for (int i=0; i<aiTrainLabels.length; i++) {
			
				double[] adTarget = MnistReader.createTarget(aiTrainLabels[i]);
			
				int[][] aaiImage = listTrainImages.get(i);
			//	String sImage = MnistReader.renderImage(aaiImage);
			//	System.out.println(sImage);

				int [] aiImageFlat = MnistReader.flat(aaiImage);
				double[] adInput = MnistReader.scaleImagePixels(aiImageFlat);
				//System.out.println("adInput.length= " + adInput.length);
				//System.out.println("adTarget.length= " + adTarget.length);
			
				double [][] aadACT = nn.trainNetwork(adInput, adTarget);
				if(((i+1) % 10000) == 0)
				{
					System.out.println((i+1) + " Images trained for Epoch " + iEpoch);
				//	ArrayUtil.show(aadACT[0], "  Layer 0 Activations");
					//ArrayUtil.show(aadACT[1], "  Layer 1 Activations");
				}
			}	
		} 
	  	}catch(Exception e){System.out.println(e);}   

    	//TEST THE NN
    	
       	String sFileTestLabels = sMinstDataPath + "t10k-labels-idx1-ubyte.gz";
    	String sFileTestImages = sMinstDataPath + "t10k-images-idx3-ubyte.gz";

    	try{
    	
    	int[] aiTestLabels = MnistReader.getLabels(Paths.get(sFileTestLabels));
    	List<int[][]> listTestImages = MnistReader.getImages(Paths.get(sFileTestImages));
    	
		System.out.println(aiTestLabels.length + " Test Labels");
		System.out.println(listTestImages.size() + " Test Images");
    	
		int iCorrect =0;
		int iWrong = 0;
		
		for (int i=0; i<aiTestLabels.length; i++) {
			

			double[] adTarget = MnistReader.createTarget(aiTestLabels[i]);
			
			int[][] aaiImage = listTestImages.get(i);
			int [] aiImageFlat = MnistReader.flat(aaiImage);
			double[] adInput = MnistReader.scaleImagePixels(aiImageFlat);

			double [] adOutput = nn.activateNN(adInput, adTarget);
			int iNNGuess = ArrayUtil.maxValueIndex(adOutput);
			if(iNNGuess ==  aiTestLabels[i])
				iCorrect++;
			else
				iWrong++;
		//	StdOut.printf("Test Label Target: %d  NN Guess: %d\n", aiTestLabels[i], iNNGuess);
		//	ArrayUtil.show(adOutput, "NN Output");
		//	String sImage = MnistReader.renderImage(aaiImage);
		//	System.out.println(sImage);
		} 
		StdOut.printf("Total Images Trained: %d Correct: %d  Wrong: %d Accuracy %5.1f%%\n",(iCorrect + iWrong), iCorrect, iWrong, (double)iCorrect/(double)(iCorrect + iWrong)*100.0);
//		StdOut.printf("Correct: %d  Wrong: %d  Total:%d\n", iCorrect, iWrong, (iCorrect + iWrong) );

    	}catch(Exception e){System.out.println(e);}   
	
	}


	public static void test1()
	{	//BUILDING THIS NEURAL NETWORK TO VERIFY SAME RESULTS:
		//	https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
		
		
		final int iNBR_NEURONS_LAYER_0 = 2; // THIS IN THE ONLY HIDDEN LAYER
		final int iNBR_NEURONS_LAYER_1 = 2; // THIS IS OUPUT LAYER 
		final double dLAYER_0_BIAS = 0.35;
        final double dLAYER_1_BIAS = 0.60;
        
      	double[] adInput = {0.05,0.1};  
      	double[] adTarget = {0.01,0.99};  
        final double dLEARNING_RATE = 0.5;
      	
		NNLayer [] aLayer = new NNLayer[2];

		//	public Layer(int nLayerNbr,  int nNbrInputs, int nNbrNeurons, double dBias)
        aLayer[0] = new NNLayer(0, adInput.length, iNBR_NEURONS_LAYER_0, dLAYER_0_BIAS);
        aLayer[1] = new NNLayer(1, iNBR_NEURONS_LAYER_0, iNBR_NEURONS_LAYER_1, dLAYER_1_BIAS);

        NeuralNetwork nn = new NeuralNetwork(aLayer, dLEARNING_RATE);
        
		double[][] aadWeights0 = {{0.15,0.2},{0.25,0.3}} ;
		double[][] aadWeights1 = {{0.4,0.45},{0.5,0.55}} ;
		nn.setWeights(0, aadWeights0);
		nn.setWeights(1, aadWeights1);
				
		double [][] aadReturn = nn.trainNetwork(adInput, adTarget);
		//ArrayUtil.show(aadReturn, "aadReturn", "%9.5f");
  	}

	
	public static void test2()
	{
		//BUILDING THIS NEURAL NETWORK TO VERIFY SAME RESULTS:
		//	https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
				
		final int iNBR_INPUTS = 2;
		final int iNBR_NEURONS_LAYER_0 = 2; // THIS IN THE ONLY HIDDEN LAYER
		final int iNBR_NEURONS_LAYER_1 = 2; // THIS IS OUPUT LAYER 
		final double dBIAS_LAYER_0 = 0.35;
        final double dBIAS_LAYER_1 = 0.60;
        final double dLEARNING_RATE = 0.5;

        //CREATE ARRAYS TO PASS IN NUMBER NEURONS/LAYER AND BIAS FOR EACH LAYER
        //THIS NN HAS 1 HIDDEN & 1 OUTPUT LAYER SO SIZE OF EACH ARRAY IS 2
        int [] aiNbrNeuronsByLayer = {iNBR_NEURONS_LAYER_0 , iNBR_NEURONS_LAYER_1 };
        double [] adBiasByLayer = { dBIAS_LAYER_0 , dBIAS_LAYER_1 };
        
        NeuralNetwork nn = new NeuralNetwork(iNBR_INPUTS, aiNbrNeuronsByLayer, adBiasByLayer, dLEARNING_RATE); 
		
        double[][] aadWeights0 = {{0.15,0.2},{0.25,0.3}} ;
		double[][] aadWeights1 = {{0.4,0.45},{0.5,0.55}} ;
		nn.setWeights(0, aadWeights0);
		nn.setWeights(1, aadWeights1);
		
      	double[] adInput = {0.05,0.1};  
      	double[] adTarget = {0.01,0.99};  

		double [][] aadReturn = nn.trainNetwork(adInput, adTarget);

	}

	
}
