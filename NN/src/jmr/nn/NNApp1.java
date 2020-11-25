package jmr.nn;


import java.io.File;
import java.util.Date;
import java.text.SimpleDateFormat;
import jmr.util.Log;

/* NNApp1 runs MNIST image data or MARKET data through NeuralNetwork 
 * pass into app MNIST or MARKET
 * Data for MNIST is stored here:  ./data/mnist/
 * Data for MARKET is stored here: ./data/market/
 */

public class NNApp1 {
	
	public static void main(String[] args) {
		
		if(args.length > 0) {

			System.out.println("args[0]= " + args[0]);

			String sDataToRun = args[0];
			
			switch (sDataToRun) {
			case "MNIST":
				MnistReader.runMinstDataSet();
				break;
			case "MARKET":
				MarketData md = new MarketData();
				md.runNNusingPrices();
				break;
			default:
				MnistReader.runMinstDataSet();
			}
		}
		else
			MnistReader.runMinstDataSet();
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
