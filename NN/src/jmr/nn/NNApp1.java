package jmr.nn;


import java.io.File;
import java.util.Date;
import java.text.SimpleDateFormat;
import jmr.util.Log;
import jmr.util.AppProperties;

//MUST PASS IN FULL PATH TO PROPERTIES FILE
//    e.g ==>  /Users/JMR/Dropbox/projects/git/repository/AppTestJava/AppTestMain.properties

//THIS IMPLEMENTATION JUST USING STATIC METHOD SO NO INSTANCE CREATED.
// CAN CREATE INSTANCE IF DESIRED.

public class NNApp1 {

	
	public static void main(String[] args) {
		if(args.length == 0) {
			System.out.println("Cannot start application : Properties path must be passed as command line parameter. Exiting Application...");
			System.exit(0);
		}
		
		//OPEN PROPERTIES FILE
		String sPropertiesFile = args[0];
		System.out.println("Loading Properties File: " +  sPropertiesFile);
		try	{AppProperties.loadProperties(sPropertiesFile);}catch (Exception e) {System.out.println(e);}
		
		//OPEN LOG FILE
		//openLogFile();
		
		//NeuralNetwork.useMinstData();
		
	//	TestClass tc = new TestClass();
		//tc.test();

		test1();
		//CAN USE CONSTRUCTOR OR JUST USE STATIC METHODS
		//	new AppTestMain();
	}
	
	public NNApp1() {
	}

	protected static void openLogFile()
	{
		try
		{
			//OPEN THE LOG FILE
			String sLogPath = AppProperties.getProperty("dir.log");
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
	
	public static void test1()
	{
		System.out.println("NNApp1.test1 method");
		//Neuron.test2();
		//Layer.test1();
		//NeuralNetwork.test2();
		NeuralNetwork.testMinstData();
	//	TestClass tc = new TestClass();
	//	tc.test();
	}
	
}
