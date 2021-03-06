package jmr.nn;

import java.nio.file.Paths;
import java.util.List;
import jmr.util.AppProperties;
import jmr.util.ArrayUtil;
import jmr.util.StdOut;


/* NeuralNetwork is composed of 1 or more Layers which are composed of Neurons.
 * There is no input layer. Inputs are simply passed to the first layer
 * The last layer is the output layer. Prior layers are hidden layers.
 * By default, the initial weights for each neuron are randomly generated. For debugging, the weights can be overridden.
 * The neurons use a sigmoid activation function.
 */

public class NeuralNetwork {
	
	Layer [] m_aLayer;
	double m_dLearningRate;
	int m_iNbrInputs;
	
	//THERE ARE 2 CONSTRUCTORS FOR CREATING NeuralNetwork
	
	//METHOD 1 IS TO CREATE THE NNLAYERS THEN PASS THEM IN
	public NeuralNetwork(Layer [] aLayers, double dLearningRate){
		m_aLayer =  aLayers;
		m_dLearningRate = dLearningRate;
		m_iNbrInputs = aLayers[0].getNumberInputs();
	}
	
	//METHOD 2 IS TO PASS IN AN ARRAY WITH THE NBR OF LAYERS/NEURONS AND AN ARRAY W/ THE BIASES AND LET NeuralNetwork CREATE THE LAYERS
	//THE aiNbrNeuronsByLayer ARRAY WOULD LOOK LIKE THIS FOR NN  A HIDDEN LAYER W/ 4 NEURONS AND AN OUTPUT LAYER W/ 3 NEURONS
	//	 aiNbrNeuronsByLayer = {4,3}  ==> THE NUMBER OF ELEMENTS IS EQUAL TO NUMBER OF LAYERS, EACH VALUE IS THE NBR NEURONS IN THAT LAYER
	// THE 	adBiasByLayer MUST HAVE SAME NBR OF ELEMENTS e.g. adBiasByLayer	{0.35, 0.60} FOR 2 LAYERS
	//FOR 3 LAYERS, aiNbrNeuronsByLayer & adBiasByLayer WOULD HAVE 3 ELEMENTS  
	//		aiNbrNeuronsByLayer = {5,10,3}  adBiasByLayer = {0.25, 0.3, 0.15}
	public NeuralNetwork (int iNbrNNInputs, int [] aiNbrNeuronsByLayer, double [] adBiasByLayer,  double dLearningRate)	{
		if (aiNbrNeuronsByLayer.length != adBiasByLayer.length) throw new RuntimeException("Mismatch params in NeuralNetwork constructor");

		m_iNbrInputs = iNbrNNInputs;
		this.m_aLayer = new Layer[aiNbrNeuronsByLayer.length];
		m_dLearningRate = dLearningRate;
		
		int iNbrLayerInputs = iNbrNNInputs;
		for (int iLayer=0; iLayer < aiNbrNeuronsByLayer.length; iLayer++)
		{
			//	public Layer(int nLayerNbr,  int nNbrInputs, int nNbrNeurons, double dBias)
			m_aLayer[iLayer] = new Layer(iLayer, iNbrLayerInputs, aiNbrNeuronsByLayer[iLayer], adBiasByLayer[iLayer]);
			iNbrLayerInputs = aiNbrNeuronsByLayer[iLayer];
		}
	}
	
// THIS LOADS PARAMS FROM PROPERTIES FILE AND USES METHOD 2 CONSTRUCTOR TO CREATE NN
	// INPUTS FOR PROPERTIES FILE FOR 3 LAYER NN
	//		nn.mnist.inputs=784
	//		nn.mnist.neurons=10,20,30
	//		nn.mnist.bias=0.1,0.2,0.3
	//		nn.mnist.learningrate=0.5
	//		nn.mnist.epochs=7
	public static NeuralNetwork loadNNFromPropertiesFile(String sKey)
	{
		int iNbrLayers = 2;
		int iNbrInputs = 10;
		double dLearningRate = 0.1;
		int [] aiNbrNeurons = new int[iNbrLayers];
		double [] adBiases = new  double[iNbrLayers];
			
		String sKeyPrefix = "nn." + sKey + ".";
		try {
			iNbrInputs = Integer.parseInt(AppProperties.getProperty(sKeyPrefix + "inputs"));
			
			String sNeurons = AppProperties.getProperty(sKeyPrefix + "neurons");
            String[] asNeuronByLayer = sNeurons.split(",");
            
			String sBiases = AppProperties.getProperty(sKeyPrefix + "bias");
            String[] sBiasByLayer = sBiases.split(",");
		
			iNbrLayers = asNeuronByLayer.length;
			
			aiNbrNeurons = new int[iNbrLayers];
			adBiases = new double[iNbrLayers];
			for(int i=0; i<iNbrLayers; i++)
			{
				aiNbrNeurons[i] = Integer.parseInt(asNeuronByLayer[i]);
				adBiases[i] = Double.parseDouble(sBiasByLayer[i]);
			}
			dLearningRate = Double.parseDouble(AppProperties.getProperty(sKeyPrefix + "learningrate"));
		}
		catch (Exception e) {
			System.out.println(e);
		}
		//CREATE THE NN AND RETURN
		NeuralNetwork nn = new NeuralNetwork(iNbrInputs, aiNbrNeurons, adBiases, dLearningRate);
		return nn;
	}
	
	public String getDescription(){
		String sDesc = "Inputs=" + this.m_iNbrInputs;
		sDesc += " Layers=" + this.m_aLayer.length;
		sDesc += " Neurons={";
		for(int i=0; i<m_aLayer.length; i++) {
			sDesc += m_aLayer[i].getNbrNeurons();
			if(i < m_aLayer.length-1)
				sDesc += ",";
		}
		sDesc += "} Biases={";
		for(int i=0; i<m_aLayer.length; i++) {
			sDesc += m_aLayer[i].getBias();
			if(i < m_aLayer.length-1)
				sDesc += ",";
		}
		sDesc += "} LearningRate=" + this.m_dLearningRate;
		return sDesc;				
	}

	public void setWeights(int iLayer, double [][] aadWeights) {
		m_aLayer[iLayer].setWeights(aadWeights);
	}
	public int getNbrLayers()	{
		return this.m_aLayer.length;
	}
	public int getNbrNeuronsInLayer(int iLayer)	{
		
		if (iLayer >=  m_aLayer.length) throw new RuntimeException("Mismatch params in NeuralNetwork.getNbrNeuronsInLayer");
		return m_aLayer[iLayer].getNbrNeurons();
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

	public double [] trainNetwork(double [] adInput, double [] adTarget){
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
	//	double dErrorTotal = m_aLayer[m_aLayer.length-1].computeError(adTarget);

		//BACKPROPOGATE OUTPUT LAYER
		//  WITH dET/dACT AS INPUT, EACH LAYER CAN COMPUTE NEW WEIGHTS
		//  AND, EACH LAYER WILL RETURN dET/dACT as input for upstream layer
		//TO START, FOR OUTPUT LAYER, dET/dACT = (ACT-TARGET)
		double [] adETdACT = ArrayUtil.minus(adACT, adTarget);
		
		for (int iLayer=m_aLayer.length-1;iLayer >=0; iLayer--) {
			adETdACT = m_aLayer[iLayer].computeNewWeights(adETdACT, m_dLearningRate);
		}
		return adACT;
	}

	public int getNbrInputs() {
		return m_iNbrInputs;
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
      	
		Layer [] aLayer = new Layer[2];

		//	public Layer(int nLayerNbr,  int nNbrInputs, int nNbrNeurons, double dBias)
        aLayer[0] = new Layer(0, adInput.length, iNBR_NEURONS_LAYER_0, dLAYER_0_BIAS);
        aLayer[1] = new Layer(1, iNBR_NEURONS_LAYER_0, iNBR_NEURONS_LAYER_1, dLAYER_1_BIAS);

        NeuralNetwork nn = new NeuralNetwork(aLayer, dLEARNING_RATE);
        
		double[][] aadWeights0 = {{0.15,0.2},{0.25,0.3}} ;
		double[][] aadWeights1 = {{0.4,0.45},{0.5,0.55}} ;
		nn.setWeights(0, aadWeights0);
		nn.setWeights(1, aadWeights1);
				
//		double [][] aadReturn = nn.trainNetwork(adInput, adTarget);
		//ArrayUtil.show(aadReturn, "aadReturn", "%9.5f");
  
		//double dTotalError = nn.trainNetwork(adInput, adTarget);
		//StdOut.printf("Total Error=%d\n", dTotalError);
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

	//	double dTotalError = nn.trainNetwork(adInput, adTarget);
		//StdOut.printf("Total Error=%d\n", dTotalError);

	//	double [][] aadReturn = nn.trainNetwork(adInput, adTarget);
	}
}
