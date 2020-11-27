package jmr.nn;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

import jmr.util.AppProperties;
//import jmr.nn.MarketData.Price;
//import jmr.util.ArrayUtil;
import jmr.util.StdOut;


public class MarketData {
	
	static final String m_sDATA_FILE = "./data/market/sp500 1927-12 to 2020-11-23.csv";
	static final int m_nDATA_SET_SIZE = 100;
	static final double m_dTRAIN_DATA_PCT = 0.99;
	static final int m_nPREDICT_DAYS_IN_FUTURE = 1;
	
	public class Price{
		public String m_sDate;
		public double m_dPrice;

		public Price(String sDate, double dPrice){
			m_sDate = sDate;
			m_dPrice = dPrice;
		}
		public String getDate() { return m_sDate; }
		public double getPrice() { return m_dPrice; }

	}

	//runNNusingPrices
	// First, the list of Prices (date, price) is loaded
	// THE LIST OF PRICE DATA WILL BE COPIED NUMEROUS TIME TO CREATE nTotalDataSets DATA SETS
	// Each data set will be m_nDATA_SET_SIZE in size
	// THE 1st data set will start with the 1st price
	// A 2nd data set will start with the 2nd price
	// A 3rd data set will start with the 3rd price (and so on)
	// Each price will be used in m_nDATA_SET_SIZE data sets, but in a different position in each data set.
	// The nTotalDataSets will be split between nTrainDataSets and nTestDataSets
	// Single Value targets are associated with each data set. The target is the price m_nPREDICT_DAYS_IN_FUTURE in the future.


	public void runNNusingPrices()
	{
		System.out.println("BEGINNING NEURAL NETWORK ON SP500 DATA");
		//LOAD PRICES INTO AN ARRAY
		double [] aPrice = loadPriceArray();
		System.out.println("Loaded " + aPrice.length + " SP500 historical prices");
		
		//SCALE THE PRICES TO FIT BETWEEN 0 AND 1
//		double [] aScaledPrice = scaleDataSet(aPrice);
		double [] aScaledPrice = pctChgDataSet(aPrice);
		
	  //COMPUTE SIZES OF TRAIN AND TEST DATA SETS
		int nTotalDataSets = aPrice.length - m_nDATA_SET_SIZE;
		int nTrainDataSets = (int)  (nTotalDataSets * m_dTRAIN_DATA_PCT);
		int nTestDataSets = nTotalDataSets - nTrainDataSets;
		StdOut.printf("Computed DataSets: TotalDataSets= %d  TrainDataSets= %d TestDataSets= %d\n", nTotalDataSets, nTrainDataSets, nTestDataSets);
		
		//CREATE TRAIN DATA SETS AND ASSOCIATED TARGETS
		int nCntTrain = 0;
		double [][] aadTrainDataSets = new double [nTrainDataSets][m_nDATA_SET_SIZE];
		double [][] aadTrainTargets = new double  [nTrainDataSets][1];

		for (int j=0; j<nTrainDataSets; j++)		{
			aadTrainDataSets[j] = Arrays.copyOfRange(aScaledPrice, j, (j + m_nDATA_SET_SIZE));
			
			int nTargetFutureIndex = j + m_nDATA_SET_SIZE + m_nPREDICT_DAYS_IN_FUTURE -1;
			//aadTrainTargets[j][0] = aScaledPrice[nTargetFutureIndex];
			if (aScaledPrice[nTargetFutureIndex] == 0.0)
				aadTrainTargets[j][0] = 0.5;
			else if (aScaledPrice[nTargetFutureIndex] < 0.0)
				aadTrainTargets[j][0] = 0.001;
			else
				aadTrainTargets[j][0] = 0.999;
			nCntTrain++;
		}
	
		//CREATE TEST DATA SETS AND ASSOCIATED TARGETS
		int nCntTest = 0;
		double [][] aadTestDataSets = new double [nTestDataSets][m_nDATA_SET_SIZE];
		double [][] aadTestDataSetsScaled = new double [nTestDataSets][m_nDATA_SET_SIZE];
		double [][] aadTestTargets = new double  [nTestDataSets][1];
		double [][] aadTestTargetsScaled = new double  [nTestDataSets][1];

		//****** WARNING: NEED TO FIX THIS IF m_nPREDICT_DAYS_IN_FUTURE > 1 *********
		for (int j=0; j<(nTestDataSets); j++)		{
			
			aadTestDataSetsScaled[j] = Arrays.copyOfRange(aScaledPrice, (j+nTrainDataSets), (j + nTrainDataSets + m_nDATA_SET_SIZE));
			aadTestDataSets[j] = Arrays.copyOfRange(aPrice, (j+nTrainDataSets), (j + nTrainDataSets + m_nDATA_SET_SIZE));

			int nTargetFutureIndex = j + nTrainDataSets + m_nDATA_SET_SIZE + m_nPREDICT_DAYS_IN_FUTURE -1;
			//aadTestTargetsScaled[j][0] = aScaledPrice[nTargetFutureIndex];
			aadTestTargets[j][0] = aPrice[nTargetFutureIndex];
			if (aScaledPrice[nTargetFutureIndex] == 0.0)
				aadTestTargetsScaled[j][0] = 0.5;
			else if (aScaledPrice[nTargetFutureIndex] < 0.0)
				aadTestTargetsScaled[j][0] = 0.001;
			else
				aadTestTargetsScaled[j][0] = 0.999;
			
			nCntTest++;
		}
		StdOut.printf("Created Data Sets: TrainDataSets= %d TestDataSets= %d\n", nCntTrain, nCntTest);
	
		NeuralNetwork nn =  NeuralNetwork.loadNNFromPropertiesFile("sp500");
		System.out.println("NN Created: " + nn.getDescription());
	   // NeuralNetwork nn = this.createNN();
	    
	   //GET NBR EPOCHS FROM PROPERTIES FILE
		int iNbrEpochs=1;
		try {
			iNbrEpochs = Integer.parseInt(AppProperties.getProperty("nn.sp500.epochs"));
			System.out.println("\nBeginning " + iNbrEpochs + " epochs of training");
		}catch(Exception e)	{
			System.out.println(e);
		};

	    //TRAIN NETWORK
		for (int iEpoch=0; iEpoch<iNbrEpochs; iEpoch++)
		{
			for (int i=0; i<aadTrainDataSets.length; i++) {
				double [] adOutput = nn.trainNetwork(aadTrainDataSets[i], aadTrainTargets[i]);
				if(((i+1) % 5000) == 0) {
					StdOut.printf("%d data sets trained for Epoch %d\n",(i+1),iEpoch);
				}
			}
			StdOut.printf("Training Epoch %d completed\n\n", iEpoch);
		} 
     
	
		//TEST NETWORK
		StdOut.printf("Beginning Testing Phase\n");
		for (int i=0; i<aadTestDataSets.length; i++) {			
			double [] adOutputScaled = nn.query(aadTestDataSetsScaled[i]);
			//double dNNGuess = Math.pow(10.0, adOutputScaled[0]*10);
			//StdOut.printf("DataSet %d  Predicted=%9.2f  Actual=%9.2f  Delta=%9.2f\n",i, dNNGuess,aadTestTargets[i][0],(dNNGuess- aadTestTargets[i][0]) );
			double dNNGuess = adOutputScaled[0];
			
			StdOut.printf("DataSet %d  Predicted=%9.6f  Actual=%9.6f  Delta=%9.6f\n",i, dNNGuess,aadTestTargetsScaled[i][0],(dNNGuess- aadTestTargetsScaled[i][0]) );
	
		} 
		System.out.println("COMPLETED NEURAL NETWORK ON MARKET DATA\n");
	}


	protected double [] pctChgDataSet(double [] adDataSet)
	{
		double [] adTemp = new double [adDataSet.length];
		adTemp[0] = 0.0; //seed 1st pctChg w/ 0.0;
		for (int i=1; i<(adDataSet.length); i++ ) {
			adTemp[i] = (adDataSet[i]-adDataSet[i-1])/adDataSet[i-1];
			//StdOut.printf("%d %8.5f %8.5f %8.5f\n", i, adDataSet[i], adDataSet[i-1], adTemp[i]);
		}
		return adTemp;
	}
	
	
	protected double [] scaleDataSet(double [] adDataSet)
	{
		double [] adTemp = new double [adDataSet.length];
		for (int i=0; i<adDataSet.length; i++ )
			adTemp[i] = Math.log10(adDataSet[i])/10;
		return adTemp;
	}
	
	
	//This calls "loadPriceList" to load price data from CSV file to List
	// then converts to double[]
	public double [] loadPriceArray()
	{
		//LOAD THE MARKET DATA
		List<Price> listPrice = loadPriceList();
		int nNbrPrices = listPrice.size();
				
  		//LOAD PRICES INTO AN ARRAY
		double [] aPrice = new double [nNbrPrices];
		for (int j=0; j<listPrice.size(); j++)
			aPrice[j] = listPrice.get(j).getPrice();
	//	System.out.println("aPrice.length= " + aPrice.length);

		return aPrice;
	}
	
	//LOADS COMMA DELIMITED FILE OF PRICES WITH 1ST COLUMN DATE AND 2ND COLUMN PRICE
	//FIRST ROW IS ASSUMED TO BE HEADERS AND IS IGNORED
	public List<Price> loadPriceList()
	{
		//Delimiters used in the CSV file
	    final String COMMA_DELIMITER = ",";
	    
        BufferedReader br = null;
        List<Price> listPrice = null;
        try {
            //Reading the csv file
            br = new BufferedReader(new FileReader(m_sDATA_FILE));
            
            //Create List for holding Employee objects
            listPrice = new ArrayList<Price>();
            
            String line = "";
            //Read to skip the header
            br.readLine();
            //Reading from the second line
            while ((line = br.readLine()) != null){
                String[] asDataLine = line.split(COMMA_DELIMITER);
                if(asDataLine.length > 1){
                	Price price = new Price (asDataLine[0], Double.parseDouble(asDataLine[1]));
                	listPrice.add(price);
                }
            }
        }
        catch(Exception ee)
        {
            ee.printStackTrace();
        }
        finally {
            try {
                br.close();
            }
            catch(IOException ie)    {
                System.out.println("Error occured while closing the BufferedReader");
                ie.printStackTrace();
            }
        }
        return listPrice;
    }
}


