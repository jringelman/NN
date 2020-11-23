package jmr.nn;

import java.util.Arrays;

import jmr.util.StdOut;

public class Layer {
	
	Neuron [] m_aNeurons;
	int m_nLayerNbr;
	int m_nNbrInputs;
	double m_dBias;
	
	public Layer(Neuron [] aNeurons, int nLayerNbr,  int nNbrInputs, double dBias)
	{
		m_aNeurons = aNeurons.clone();
		m_nLayerNbr = nLayerNbr;
		m_nNbrInputs = nNbrInputs;
		m_dBias = dBias;
	}
	
	public Layer(int nLayerNbr, int nNbrNeurons,  int nNbrInputs, double dBias)
	{
		m_nLayerNbr = nLayerNbr;
		m_aNeurons = new Neuron[nNbrNeurons];
		m_nNbrInputs = nNbrInputs;
		m_dBias = dBias;
		//public Neuron (int iLayerNbr, int iNeuronNbr, int nNbrInputs)
		for(int i=0; i< nNbrNeurons; i++)
			m_aNeurons[i] = new Neuron(nLayerNbr, i, nNbrInputs);
	}

	
	public int getNbrNeurons() {
		return m_aNeurons.length;
	}
	
	public void randomizeWeights()
	{
		for(int i=0; i<m_aNeurons.length; i++) 
			m_aNeurons[i].randomizeWeights();
	}
	
	public void setWeights(double [][] aadWeights)
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
	
	public static void test()
	{
		Neuron [] aNeurons = new Neuron [2];
		double[] adWeights = new double[2]; 
        double adBias; 

        //LAYER 1
        adBias = 0.35;
        adWeights[0] = 0.15;
        adWeights[1] = 0.20;
        aNeurons[0] = new Neuron(adWeights, 1,1); 
        adWeights[0] = 0.25;
        adWeights[1] = 0.30;
        aNeurons[1] = new Neuron(adWeights, 1,2); 

        Layer layer = new Layer(aNeurons, 1, 2, adBias);
		double[] adInputs = {0.05,0.1};
		adInputs = layer.activate(adInputs);
		
	}

}
