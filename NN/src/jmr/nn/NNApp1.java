package jmr.nn;


import java.io.File;
import java.util.Date;

import jmr.util.AppProperties;

import java.text.SimpleDateFormat;
import jmr.util.Log;
import jmr.util.StdOut;

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
		
		//GET DATASETS TO RUN
		String sRunMnist = "1";
		String sRunSP500 = "0";
		
		try	{
			AppProperties.loadProperties(m_sPROPERTIES_FILE);
			sRunMnist = AppProperties.getProperty("nn.mnist.run");
			sRunSP500 = AppProperties.getProperty("nn.sp500.run");
			StdOut.printf("Properties file has nn.mnist.run=%s & nn.sp500.run=%s\n", sRunMnist, sRunSP500);
		}catch (Exception e){
			System.out.println(e);
		}
		
		if(sRunMnist.equals("1"))
			MnistReader.runMinstDataSet();

		if(sRunSP500.equals("1"))
		{
			MarketData md = new MarketData();
			md.runNNusingPrices();

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
