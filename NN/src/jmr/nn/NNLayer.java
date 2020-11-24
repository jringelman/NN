package jmr.nn;

import java.util.Arrays;

import jmr.util.ArrayUtil;
import jmr.util.StdOut;

public class NNLayer {
	
	Neuron [] m_aNeurons;
	int m_nLayerNbr;
	int m_nNbrInputs;
	double m_dBias;
	
	/*
	public Layer(Neuron [] aNeurons, int nLayerNbr,  int nNbrInputs, double dBias)
	{
		m_aNeurons = aNeurons.clone();
		m_nLayerNbr = nLayerNbr;
		m_nNbrInputs = nNbrInputs;
		m_dBias = dBias;
	}*/
	
	public NNLayer(int nLayerNbr,  int nNbrInputs, int nNbrNeurons, double dBias)
	{
		m_nLayerNbr = nLayerNbr;
		m_aNeurons = new Neuron[nNbrNeurons];
		m_nNbrInputs = nNbrInputs;
		m_dBias = dBias;
		//public Neuron (int iLayerNbr, int iNeuronNbr, int nNbrInputs)
		for(int iNeuronNbr=0; iNeuronNbr< nNbrNeurons; iNeuronNbr++)
			m_aNeurons[iNeuronNbr] = new Neuron(nLayerNbr, iNeuronNbr, nNbrInputs);
	}
	
	public int getNbrNeurons() {
		return m_aNeurons.length;
	}
	
	/*public void randomizeWeights()
	{
		for(int i=0; i<m_aNeurons.length; i++) 
			m_aNeurons[i].randomizeWeights();
	}*/
	
	protected void setWeights(double [][] aadWeights)
	{
		for(int i=0; i<m_aNeurons.length; i++)
			m_aNeurons[i].setWeights(aadWeights[i]);
	}
	
	public double[] activate(double [] adInputs)
	{
		if (adInputs.length != m_nNbrInputs) throw new RuntimeException("Mismatch params in Layer.activate");

		double [] adActivation = new double [m_aNeurons.length];

		for(int i=0; i<m_aNeurons.length; i++) {
			adActivation[i] = m_aNeurons[i].activate(adInputs, m_dBias);
		}
		return adActivation;	
	}
	
	public double computeError(double [] adTarget) //ONLY USED FOR OUTPUT LAYER NEURONS
	{
		if (adTarget.length != m_aNeurons.length) throw new RuntimeException("Mismatch params in Layer.computeError");

		double [] adError = new double [m_aNeurons.length];
		double dErrorTotal = 0;
	
		for(int i=0; i<m_aNeurons.length; i++) {
			adError[i] = m_aNeurons[i].computeError(adTarget[i]);
			dErrorTotal += adError[i];
		}
	//	StdOut.printf("Layer %d Total Error = %9.5f\n",m_nLayerNbr, dErrorTotal);
		return dErrorTotal;
	}
	
	public double [] computeNewWeights(double [] adETdACT, double dLearningRate)
	{
		if (adETdACT.length != m_aNeurons.length) throw new RuntimeException("Mismatch params in Layer.computeNewWeights");

		double [][] aadETdACTNext = new double [m_aNeurons.length][m_nNbrInputs];

		for(int iNeuron=0; iNeuron<m_aNeurons.length; iNeuron++) {
			aadETdACTNext[iNeuron] = m_aNeurons[iNeuron].computeNewWeights(adETdACT[iNeuron], dLearningRate);
		}
				
		//HAVE TO BE CAREFULL HOW TO ADD VALUES IN aadETdACTNext TO PASS TO UPSTREAM LAYER
		double [] adETdACTNext = new double [m_nNbrInputs];
		Arrays.fill(adETdACTNext, 0.0);
		for (int iInput=0; iInput< m_nNbrInputs; iInput++)
			for (int iNeuron=0; iNeuron< m_aNeurons.length; iNeuron++)
				adETdACTNext[iInput] += aadETdACTNext[iNeuron][iInput];
		
		return adETdACTNext;
	}

	//**************************************************
	//************ STATIC TEST METHODS *****************
	//**************************************************


	public static void test1()
	{
	    final int iLAYER_NBR = 0;
	    final int iNBR_NEURONS = 2;
        final int iNBR_OF_INPUTS = 2;
		double dBias = 0.35; 

		NNLayer layer = new NNLayer(iLAYER_NBR, iNBR_OF_INPUTS, iNBR_NEURONS, dBias);
	    
		double[][] aadWeights = {{0.15,0.2},{0.25,0.3}} ;
		layer.setWeights(aadWeights);
				
	    double[] adInputs = {0.05,0.1};  
	    double adActivation[] = layer.activate(adInputs);
	    ArrayUtil.show(adActivation, "adActivation");
	    
	}
}
