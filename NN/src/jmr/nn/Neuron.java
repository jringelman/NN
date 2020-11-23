package jmr.nn;

import jmr.util.ArrayUtil;
//import jmr.util.StdOut;
import java.util.Random;


public class Neuron {
	
	int m_iLayerNbr;
	int	m_iNeuronNbr;
//	String m_sId;
	double m_adWeights[];
	double m_adInputs[];
	double m_dNet;
	double m_dAct;
	
	private final static Random random = new Random();
	
	public Neuron (double [] adWeights, int iLayerNbr, int iNodeNbr) {
		m_adWeights = adWeights.clone(); 
		m_iLayerNbr = iLayerNbr;
		m_iNeuronNbr = iNodeNbr;
	}
	
	public Neuron (int iLayerNbr, int iNeuronNbr, int nNbrInputs) {
		m_iLayerNbr = iLayerNbr;
		m_iNeuronNbr = iNeuronNbr;
		m_adWeights = new double [nNbrInputs];
	}
	
	//WITH LOTS OF INPUTS, IF YOU JUST USE RANDOM (0.0 to 1.0) VALUES FOR INTIAL WEIGHTS
	//  THE ACTIVATION FUNCTION CAN GET PEGGED AT 1.0 BECAUSE THE SUM OF NET VALUES GETS LARGE AND SIMOID BECOMES 1.0
	//THIS EXAMPLE SETS RANDOM WEIGHT = RANDOM-NBR 	x nbrInputNodes^-0.5 = RANDOM-NBR 	x 1/(nbrInputNodes^0.5)
	//https://golb.hplar.ch/2018/12/simple-neural-network.html
	//https://github.com/ralscha/blog/tree/master/mnist/java/src/main/java/ch/rasc/mnist
	//random.nextGaussian() * desiredStandardDeviation = random.nextGaussian() * Math.pow(nbrInputNodes, -0.5)
	
	public void randomizeWeights() {
		for (int i=0; i < m_adWeights.length; i++) {
			double dDesiredStandardDeviation = Math.pow(m_adWeights.length, -0.5);
			m_adWeights [i] = random.nextGaussian() * dDesiredStandardDeviation; 
			if (m_adWeights [i] == 0.0)
				m_adWeights [i] += 0.0001;
		//SPECIAL FOR THIS NETWORK  *********
			//if(m_iLayerNbr == 0)
			//	m_adWeights [i] = m_adWeights [i] / 10.0;
	}		
	}
	
	public void setWeights(double [] adWeights){
		m_adWeights = adWeights.clone();	
	}
	
	public double activate(double [] adInputs, double dBias){
		if (adInputs.length != m_adWeights.length) throw new RuntimeException("Mismatch params in NNNode.activate");
		
		m_adInputs = adInputs.clone();
		m_dNet = 0;
		for (int i=0; i < adInputs.length; i++){
			m_dNet += adInputs[i] * m_adWeights[i];
		}
		m_dNet += dBias;
		m_dAct = 1/(1+Math.exp(-m_dNet));
     //   StdOut.printf("Neuron %d:%d Activated NET= %9.5f, ACT= %9.5f\n", m_iLayerNbr, m_iNeuronNbr, m_dNet, m_dAct);   
        return m_dAct;
	}
	
	//ONLY USED FOR OUTPUT LAYER NEURONS
	public double computeError(double dTarget) {
		double dError = 0.5 * (dTarget - m_dAct)*(dTarget - m_dAct);
	//	StdOut.printf("Neuron %d:%d Error = %9.5f\n", m_iLayerNbr, m_iNeuronNbr, dError);
		return dError;
	}
	
	public double [] computeNewWeights(double dETdACT, double dLearningRate) {

		/*  dET/dW = dNET/dW x dACT/dNET x dET/dACT
			   dNET/dW : INPUT which is ACT from upstream or INPUT
			   dACT/dNET: ACT x (1-ACT)
			   dET/dACT: passed in; either (ACT-TARG) for output layer or calculated by downstream layer 

	 		dET/dNET = dACT/dNET x dET/dACT      - JUST CONSOLIDATES THIS TERM FOR USE
				so
			dET/dW = dNET/dW x dET/dNET
		
			NEW WEIGHTS = W - LearningRate * dET/dW

		COMPUTE dET/dACT for upstream layers (returns 1 value for each weight) 
		     dET/dACT = dNET(this layer)/dACT(upstream layer) x dE/dNET(this layer)
				      =   Weight							x    dACT/dNET x dET/dACT (or just dE/dNET)
		*/
		
		double dACTdNET = m_dAct*(1-m_dAct);
		double dETdNET = dACTdNET * dETdACT;
		double [] adNETdW =  m_adInputs.clone(); // dNETdW is the INPUTS
		double [] adETdW = ArrayUtil.times(adNETdW, dETdNET);
		double [] adNewWeights = ArrayUtil.minus(m_adWeights, ArrayUtil.times(adETdW, dLearningRate));
		
		//CALCULATIONS FOR UPSTREAM
		double [] adNETdACTupstrm =  m_adWeights.clone(); // dNETdACTup is the weights.
		double [] adEdACTupstrm  = ArrayUtil.times(adNETdACTupstrm, dETdNET);

		//StdOut.printf("Neuron %d:%d\n", m_iLayerNbr, m_iNeuronNbr);
		/*StdOut.printf("   dETdACT= %9.5f\n", dETdACT);
		StdOut.printf("   dACTdNET= %9.5f\n", dACTdNET);
		StdOut.printf("   dETdNET= %9.5f\n", dETdNET);
		ArrayUtil.show(adNETdW,"adNETdW");
		ArrayUtil.show(adETdW,"adETdW");
		*/
		//String sLabel = "Neuron " + m_iLayerNbr + ":" + m_iNeuronNbr + " New Weights";
//ArrayUtil.show(adNewWeights,"Neuron " + m_iLayerNbr + ":" + m_iNeuronNbr + " New Weights");
		
	//	ArrayUtil.show(adNETdACTupstrm,"adNETdACTupstrm");
	//	ArrayUtil.show(adEdACTupstrm,"adEdACTupstrm");

		m_adWeights = adNewWeights.clone();
		return adEdACTupstrm;
	}
/******** STATIC TEST METHODS ********/
	public static void test()
	{
		double[] adWeights = {0.15,0.2};
        double adBias = 0.35; 
        				
		Neuron node = new Neuron(adWeights, 1,1);
		
		double[] adInputs = {0.05,0.1};
		node.activate(adInputs, adBias);
	}
	
	public static void test2()
	{
		double[] adWeights = {0.1, 0.3, 0.5};
        double adBias = 0.5; 
        				
		Neuron node = new Neuron(adWeights, 1,1 );
		
		double[] adInputs = {1.0, 4.0, 5.0};
		node.activate(adInputs, adBias);
	}
}

