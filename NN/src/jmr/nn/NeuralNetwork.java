package jmr.nn;

import java.nio.file.Paths;
import java.util.List;
import jmr.util.AppProperties;
import jmr.util.ArrayUtil;
import jmr.util.StdOut;

/* NeuralNetwork is composed of 1 or more NNLayers which are composed of Neurons.
 * There is no input layer. Inputs are simply passed to the first layer
 * The last layer is the output layer. Prior layers are hidden layers.
 * By default, the initial weights for each neuron are randomly generated. For debugging, the weights can be overridden.
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
	
	public void setWeights(int iLayer, double [][] aadWeights) {
		m_aLayer[iLayer].setWeights(aadWeights);
	}
	
	public double [] query(double [] adInput) {
		double [] adACT = adInput;
		
		//FEEDFORWARD - ACTIVATE EACH LAYER GOING FORWARD
		for (int iLayer=0; iLayer<m_aLayer.length; iLayer++) {
			adACT = m_aLayer[iLayer].activate(adInput);
			adInput = adACT; //INPUT FOR NEXT LAYER IS ACTIVATION OUTPUT FROM PRIOR LAYER.
		}
		return adACT; //RETURNS FINAL OUTPUT
	}

	public double trainNetwork(double [] adInput, double [] adTarget){
		double [] adACT = adInput;
		//double [][] aadACT = new double [m_aLayer.length][];
		
		//FEEDFORWARD - ACTIVATE EACH LAYER GOING FORWARD
		for (int iLayer=0; iLayer<m_aLayer.length; iLayer++) {
			adACT = m_aLayer[iLayer].activate(adInput);
			adInput = adACT; //INPUT FOR NEXT LAYER IS ACTIVATION OUTPUT FROM PRIOR LAYER.
			//aadACT[iLayer] = adACT.clone();
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
		return dErrorTotal;
	}

	
	
	
//*********************************************************
//************ STATIC METHODS FOR TESTING *****************
//*********************************************************

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
				
//		double [][] aadReturn = nn.trainNetwork(adInput, adTarget);
		//ArrayUtil.show(aadReturn, "aadReturn", "%9.5f");
  
		double dTotalError = nn.trainNetwork(adInput, adTarget);
		StdOut.printf("Total Error=%d\n", dTotalError);
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

		double dTotalError = nn.trainNetwork(adInput, adTarget);
		StdOut.printf("Total Error=%d\n", dTotalError);

	//	double [][] aadReturn = nn.trainNetwork(adInput, adTarget);
	}
}
