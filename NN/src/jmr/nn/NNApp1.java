package jmr.nn;


import java.io.File;
import java.util.Date;

import jmr.util.AppProperties;

import java.text.SimpleDateFormat;
import jmr.util.Log;

/* NNApp1 runs MNIST image data or MARKET data through NeuralNetwork 
 * pass into app MNIST or MARKET
 * Data for MNIST is stored here:  ./data/mnist/
 * Data for MARKET is stored here: ./data/market/
 */

public class NNApp1 {
	static final String m_sPROPERTIES_FILE = "./NN.properties";
	
	public static void main(String[] args) {
		
		//OPEN PROPERTIES FILE
		System.out.println("Loading Properties File: " +  m_sPROPERTIES_FILE);
		
		//GET DATASET TO RUN
		String sDataset = "";
		try	{
			AppProperties.loadProperties(m_sPROPERTIES_FILE);
			sDataset = AppProperties.getProperty("dataset");
			
		}catch (Exception e){
			System.out.println(e);
		}
					
		switch (sDataset) {
			case "mnist":
				MnistReader.runMinstDataSet();
				break;
			case "sp500":
				MarketData md = new MarketData();
				md.runNNusingPrices();
				break;
			default:
				MnistReader.runMinstDataSet();
		}
	}

	
	//LOG FILE NOT BEING USED CURRENTLY
	protected static void openLogFile(String sLogPath)
	{
		try
		{
			//OPEN THE LOG FILE
			//String sLogPath = AppProperties.getProperty("dir.log");
			SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMdd'_'HHmmss");
			String sDateTime = dateFormat.format(new Date());
			String sFileName = "Prod_" + sDateTime + ".log";
			File fileLog = new File(sLogPath);
			fileLog.mkdirs();
			Log.open(sLogPath + sFileName);
			System.out.println("Log File Created: " + sLogPath + sFileName);
		}catch (Exception e) {
			System.out.println(e);
		}
	
	}
}
