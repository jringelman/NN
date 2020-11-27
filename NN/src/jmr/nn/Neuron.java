package jmr.nn;

import jmr.util.ArrayUtil;
import jmr.util.StdOut;
import java.util.Random;


public class Neuron {
	
	int m_iLayerNbr;
	int	m_iNeuronNbr;
	String m_sLabel;
	double m_adWeights[];
	double m_adInputs[];
	double m_dNet; //THE NET SUM OF INPUTS * WEIGHTS PLUS BIAS
	double m_dActivation; //NEURON ACTIVATION AFTER SIGMOID FUNCTION PERFORMED ON NET VALUE
	
	private final static Random random = new Random();
	
	public Neuron (int iLayerNbr, int iNeuronNbr, int nNbrInputs) {
		m_iLayerNbr = iLayerNbr;
		m_iNeuronNbr = iNeuronNbr;
		m_adWeights = new double [nNbrInputs];
		m_adInputs = new double [nNbrInputs]; // DO THIS JUST SO m_adInputs REFERENCE IS NOT NULL
		m_sLabel = "Neuron " + m_iLayerNbr + ":" + m_iNeuronNbr;
		randomizeWeights();
	}
	
	//IF YOU JUST USE RANDOM (0.0 to 1.0) VALUES FOR INTIAL WEIGHTS THEN, WITH LOTS OF INPUTS, THE ACTIVATION
	//   FUNCTION CAN GET PEGGED AT 1.0 BECAUSE THE SUM OF NET VALUES GETS LARGE AND SIMOID BECOMES ~1.0
	//TO COUNTER THIS, AJUST RANDOM WEIGHT BASED ON NUMBER OF INPUTS.
	//    RANDOM WEIGHT = RANDOM-NBR x nbrInputNodes^-0.5 = RANDOM-NBR x 1/(nbrInputNodes^0.5)
	//    random.nextGaussian() * desiredStandardDeviation = random.nextGaussian() * Math.pow(nbrInputNodes, -0.5)
	//https://golb.hplar.ch/2018/12/simple-neural-network.html
	//https://github.com/ralscha/blog/tree/master/mnist/java/src/main/java/ch/rasc/mnist
	private void randomizeWeights() {
		for (int i=0; i < m_adWeights.length; i++) {
			double dDesiredStandardDeviation = Math.pow(m_adWeights.length, -0.5);
			m_adWeights [i] = random.nextGaussian() * dDesiredStandardDeviation; 
			if (m_adWeights [i] == 0.0) //DON'T WANT WEIGHT TO BE ZERO SO INITIALIZE TO SMALL POSITIVE VALUE IF ZERO
				m_adWeights [i] += 0.0001;
		}		
	}
	
	//SETTING WEIGHTS IS USED FOR TESTING; IT OVERIDES THE randomizeWeights SET IN THE CONSTRUCTOR
	protected void setWeights(double [] adWeights){
		m_adWeights = adWeights.clone();	
	}
	
	protected double activate(double [] adInputs, double dBias){
		if (adInputs.length != m_adWeights.length) throw new RuntimeException("Mismatch params in NNNode.activate");
		
		m_adInputs = adInputs.clone();
		m_dNet = 0;
		for (int i=0; i < adInputs.length; i++){
			m_dNet += adInputs[i] * m_adWeights[i];
		}
		m_dNet += dBias;
		m_dActivation = 1/(1+Math.exp(-m_dNet));
       // StdOut.printf("%s Activated  NET=%8.5f  ACT=%8.5f\n", m_sLabel , m_dNet, m_dActivation);   
        return m_dActivation;
	}
	
	//ONLY USED FOR OUTPUT LAYER NEURONS
	protected double computeError(double dTarget) {
		double dError = 0.5 * (dTarget - m_dActivation)*(dTarget - m_dActivation);
		//StdOut.printf("%s Error=%8.5f\n", m_sLabel, dError);
		return dError;
	}
	
	/* computeNewWeights DOES BACKPROPAGATION....
	   FIRST, COMPUTE NEW WEIGHTS FOR THIS NEURON
		   NEW WEIGHT = Current Weight - LearningRate * dET/dW
	
	  	   dET/dW = dNET/dW x        dET/dNET 
		   dET/dW = dNET/dW x (dACT/dNET x dET/dACT) 
			 -   dNET/dW : INPUT which is ACT from upstream neurons or NN INPUT if 1st hidden layer
			 -   dACT/dNET: ACT x (1-ACT)
			 -   dET/dACT: passed in; either (ACT-TARG) for output layer or calculated by downstream layer 
	
	 THEN, COMPUTE dET/dACT TO BE USED BY upstream layers (returns one value for each weight) 
			dET/dACT = dNET(this layer)/dACT(upstream layer) x dE/dNET(this layer)
			        =   Weight					x    dACT/dNET x dET/dACT (or just dE/dNET) 
	*/
	protected double [] computeNewWeights(double dETdACT, double dLearningRate) {

		//COMPUTE NEW WEIGHTS FOR THIS NEURON
		double dACTdNET = m_dActivation*(1-m_dActivation);
		double dETdNET = dACTdNET * dETdACT;
		double [] adNETdW =  m_adInputs.clone(); // dNETdW is the INPUTS
		double [] adETdW = ArrayUtil.times(adNETdW, dETdNET);
		double [] adNewWeights = ArrayUtil.minus(m_adWeights, ArrayUtil.times(adETdW, dLearningRate));

		//DEBUGGING TRACES
	//	StdOut.printf("%s dETdACT=%8.5f dACTdNET=%8.5f dETdNET=%8.5f\n",m_sLabel,dETdACT,dACTdNET,dETdNET);
	//	ArrayUtil.showFlat(adNETdW, m_sLabel + " adNETdW", "%8.5f");
	//	ArrayUtil.showFlat(adETdW, m_sLabel + " adETdW", "%8.5f");
	//	ArrayUtil.showFlat(adNewWeights, m_sLabel + " New Weights", "%8.5f");
				
		//COMPUTE dET/dACT TO BE USED BY upstream layers
		double [] adNETdACTupstrm =  m_adWeights.clone(); // dNETdACTup is the weights.
		double [] adEdACTupstrm  = ArrayUtil.times(adNETdACTupstrm, dETdNET);

		//MORE DEBUGGING TRACES
	 //	ArrayUtil.showFlat(adNETdACTupstrm,m_sLabel + " adNETdACTupstrm", "%8.5f");
	//	ArrayUtil.showFlat(adEdACTupstrm,m_sLabel + " adEdACTupstrm", "%8.5f");
		
		m_adWeights = adNewWeights.clone();
		return adEdACTupstrm;  //THIS VALUE WILL BE USED BY UPSTREAM LAYERS FOR COMPUTING NEW WEIGHTS
	}
	
	
//*********************************************************
//************ STATIC METHODS FOR TESTING *****************
//*********************************************************

	public static void test1()
	{
        final int iLAYER_NBR = 0;
        final int iNEURON_NBR = 0;
        final int iNBR_OF_INPUTS = 2;

		Neuron neuron = new Neuron(iLAYER_NBR, iNEURON_NBR, iNBR_OF_INPUTS);

		double[] adWeights = {0.15,0.2};
		neuron.setWeights(adWeights);
				
		double adBias = 0.35; 
        double[] adInputs = {0.05,0.1};
		
		double dActivation = neuron.activate(adInputs, adBias);
		double dExpectedNet = adInputs[0] * adWeights[0] + adInputs[1] * adWeights[1]  + adBias;
		double dExpectedAct = 1/(1+Math.exp(-dExpectedNet));
		System.out.println("Activation = " + dActivation + "; Expected = " + dExpectedAct);
		
	}
	
	public static void test2() {
	    final int iLAYER_NBR = 0;
	    final int iNEURON_NBR = 0;
	    final int iNBR_OF_INPUTS = 3;
	
		Neuron neuron = new Neuron(iLAYER_NBR, iNEURON_NBR, iNBR_OF_INPUTS);
	
		double[] adWeights = {0.1, 0.3, 0.5};
		neuron.setWeights(adWeights);
				
	    double adBias = 0.5; 
		double[] adInputs = {1.0, 4.0, 5.0};
		
		double dActivation = neuron.activate(adInputs, adBias);
		double dExpectedNet = adInputs[0] * adWeights[0] + adInputs[1] * adWeights[1] + adInputs[2] * adWeights[2]  + adBias;
		double dExpectedAct = 1/(1+Math.exp(-dExpectedNet));
		System.out.println("Activation = " + dActivation + "; Expected = " + dExpectedAct);
	}
}

