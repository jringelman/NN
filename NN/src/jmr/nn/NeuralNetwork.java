package jmr.nn;

import java.nio.file.Paths;
import java.util.List;

import jmr.util.AppProperties;

import jmr.util.ArrayUtil;
import jmr.util.StdOut;

/* NeuralNetwork uses these other classes: Layer, Neuron, Matrix, StdOut
 */

public class NeuralNetwork {
	
	Layer [] m_aLayer;
	double m_dLearningRate;
	
	public NeuralNetwork(Layer [] aLayers, double dLearningRate){
		m_aLayer =  aLayers;
		m_dLearningRate = dLearningRate;
	}

	
	public void randomizeWeights()
	{
		for(int i=0; i<m_aLayer.length; i++) 
			m_aLayer[i].randomizeWeights();
	}
	
	public void setWeights(int iLayer, double [][] aadWeights)
	{
		m_aLayer[iLayer].setWeights(aadWeights);
	}
	
	public double [] useNN(double [] adInput, double [] adTarget) {
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
		

		//CALC TOTAL ERROR
		double dErrorTotal = m_aLayer[m_aLayer.length-1].computeError(adTarget);

		//BACKPROPOGATE OUTPUT LAYER
		
		//WITH dET/dACT AS INPUT, EACH LAYER CAN COMPUTE NEW WEIGHTS
		//AND, EACH LAYER WILLL RETURN dET/dACT as input for upstream layer

		//TO START, FOR OUTPUT LAYER, dET/dACT = (ACT-TARGET)
		double [] adETdACT = ArrayUtil.minus(adACT, adTarget);
		
		for (int iLayer=m_aLayer.length-1;iLayer >=0; iLayer--) {
			adETdACT = m_aLayer[iLayer].computeNewWeights(adETdACT, m_dLearningRate);
		}
		return aadACT;
	}
	
	public static void useMinstData()
	{
		double dBias = 0.1; 
		final double dLEARNING_RATE = 0.1;
	      	
		Layer [] aLayer = new Layer[2];
		//Layer(int nLayerNbr, int nNbrNeurons,  int nNbrInputs, double dBias)
	    aLayer[0] = new Layer(0, 20, 784, dBias);
	    aLayer[1] = new Layer(1, 10, 20, dBias);
	    NeuralNetwork nn = new NeuralNetwork(aLayer, dLEARNING_RATE);
	    nn.randomizeWeights();

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

		for (int iEpoch=0; iEpoch<10; iEpoch++)
		{
			System.out.println("\nEpoch: " + iEpoch);
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
					System.out.println("Trained: " + (i+1));
					ArrayUtil.show(aadACT[0], "Layer 0");
					ArrayUtil.show(aadACT[1], "Layer 1");
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

			double [] adOutput = nn.useNN(adInput, adTarget);
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
		StdOut.printf("Correct: %d  Wrong: %d  Total:%d\n", iCorrect, iWrong, (iCorrect + iWrong) );

    	}catch(Exception e){System.out.println(e);}   
	
	}


	public static void test2()
	{
        double dBias; 
      	double[] adInput = {0.05,0.1};  
      	double[] adTarget = {0.01,0.99};  
        final double dLEARNING_RATE = 0.5;
      	
		Layer [] aLayer = new Layer[2];
		
      	dBias = 0.35;
        aLayer[0] = new Layer(0, 2, adInput.length, dBias);
        
        dBias = 0.60;
        aLayer[1] = new Layer(1, 2, aLayer[0].getNbrNeurons(), dBias);

        NeuralNetwork nn = new NeuralNetwork(aLayer, dLEARNING_RATE);
        
		//double[] adWeights = new double[2]; 
      //  nn.randomizeWeights();
		double[][] aadWeights0 = {{0.15,0.2},{0.25,0.3}} ;
		double[][] aadWeights1 = {{0.4,0.45},{0.5,0.55}} ;
		aLayer[0].setWeights(aadWeights0);
		aLayer[1].setWeights(aadWeights1);
				
        nn.trainNetwork(adInput, adTarget);
  	}
	
	public static void test1()
	{
		Neuron [] aNeurons = new Neuron [2];
		double[] adWeights = new double[2]; 
        double dBias; 
        Layer [] aLayer = new Layer[2];
        final double dLEARNING_RATE = 0.5;
      //  final int iNBR_INPUTS_TO_HIDDEN = 2;;
        
        //CREATE INPUT
      	double[] adInput = {0.05,0.1};  //[2][1]

      	//CREATE TARGET
      	double[] adTarget = {0.01,0.99};  //[2][1]

        
        //LAYER 1
      	dBias = 0.35;
        adWeights[0] = 0.15;
        adWeights[1] = 0.20;
        aNeurons[0] = new Neuron(adWeights, 0,0); 
        adWeights[0] = 0.25;
        adWeights[1] = 0.30;
        aNeurons[1] = new Neuron(adWeights, 0,1); 
        aLayer[0] = new Layer(aNeurons, 0,adInput.length, dBias);

       
        //LAYER 2
        dBias = 0.60;
        adWeights[0] = 0.40;
        adWeights[1] = 0.45;
        aNeurons[0] = new Neuron(adWeights, 1,0); 
        adWeights[0] = 0.50;
        adWeights[1] = 0.55;
        aNeurons[1] = new Neuron(adWeights, 1,1); 
        aLayer[1] = new Layer(aNeurons, 1, aLayer[0].getNbrNeurons(), dBias);

        NeuralNetwork nn = new NeuralNetwork(aLayer, dLEARNING_RATE);
         		
		nn.trainNetwork(adInput, adTarget);
		}
}
