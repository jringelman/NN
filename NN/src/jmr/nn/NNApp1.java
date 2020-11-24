package jmr.nn;


import java.io.File;
import java.nio.file.Paths;
import java.util.Date;
import java.util.List;
import java.text.SimpleDateFormat;
import jmr.util.Log;
import jmr.util.StdOut;
import jmr.util.AppProperties;
import jmr.util.ArrayUtil;

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
		
		//openLogFile();
		
		testMinstData();

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
	
	//**************************************************
	//************ STATIC TEST METHODS *****************
	//**************************************************
	
	public static void testMinstData()
	{
		double dBias = 0.1; 
		final double dLEARNING_RATE = 0.1;
		final int iNBR_INPUTS = 784; //28 x 28 pixels = 784 pixels
		final int iNBR_NEURONS_LAYER0 = 20; // THIS IN THE ONLY HIDDEN LAYER
		final int iNBR_NEURONS_LAYER1 = 10; // THIS IS OUPUT LAYER WITH EACH NODE FOR DIGITS 0 THRU 9;
		final int iNBR_EPOCHS = 5;
		
		NNLayer [] aLayer = new NNLayer[2];
		//	public Layer(int nLayerNbr,  int nNbrInputs, int nNbrNeurons, double dBias)
	    aLayer[0] = new NNLayer(0, iNBR_INPUTS, iNBR_NEURONS_LAYER0, dBias);
	    aLayer[1] = new NNLayer(1, iNBR_NEURONS_LAYER0, iNBR_NEURONS_LAYER1, dBias);
	    NeuralNetwork nn = new NeuralNetwork(aLayer, dLEARNING_RATE);

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
		System.out.println("Beginning " + iNBR_EPOCHS + " epochs of training");

		double dErrorForEpoch = 0.0;
		for (int iEpoch=0; iEpoch<iNBR_EPOCHS; iEpoch++)
		{
			//System.out.println("\nEpoch: " + iEpoch);
			for (int i=0; i<aiTrainLabels.length; i++) {
			
				double[] adTarget = MnistReader.createTarget(aiTrainLabels[i]);
			
				int[][] aaiImage = listTrainImages.get(i);
			//	String sImage = MnistReader.renderImage(aaiImage);
			//	System.out.println(sImage);

				int [] aiImageFlat = MnistReader.flat(aaiImage);
				double[] adInput = MnistReader.scaleImagePixels(aiImageFlat);
			
				double dErrorThisTrain = nn.trainNetwork(adInput, adTarget);
				dErrorForEpoch += dErrorThisTrain;
				if(((i+1) % 10000) == 0)
				{
					StdOut.printf("%d Images trained for Epoch %d  TotalError=%7.4f\n",(i+1),iEpoch, dErrorThisTrain);
				}
			}
			StdOut.printf("Epoch %d completed with average Error= %7.4f\n\n", iEpoch, dErrorForEpoch/(double)aiTrainLabels.length);
		} 
	  	}catch(Exception e){System.out.println(e);}   

    	//RUN THE MINST TEST DATA
    	String sFileTestLabels = sMinstDataPath + "t10k-labels-idx1-ubyte.gz";
    	String sFileTestImages = sMinstDataPath + "t10k-images-idx3-ubyte.gz";

    	try{
    	
    	int[] aiTestLabels = MnistReader.getLabels(Paths.get(sFileTestLabels));
    	List<int[][]> listTestImages = MnistReader.getImages(Paths.get(sFileTestImages));
    	
		System.out.println(aiTestLabels.length + " Test Labels");
		System.out.println(listTestImages.size() + " Test Images");
    	
		int iCorrect =0;
		
		for (int i=0; i<aiTestLabels.length; i++) {			
			int[][] aaiImage = listTestImages.get(i);
			int [] aiImageFlat = MnistReader.flat(aaiImage);
			double[] adInput = MnistReader.scaleImagePixels(aiImageFlat);

			double [] adOutput = nn.query(adInput);
			int iNNGuess = ArrayUtil.maxValueIndex(adOutput);
			if(iNNGuess ==  aiTestLabels[i])
				iCorrect++;
		//	StdOut.printf("Test Label Target: %d  NN Guess: %d\n", aiTestLabels[i], iNNGuess);
		//	ArrayUtil.show(adOutput, "NN Output");
		//	String sImage = MnistReader.renderImage(aaiImage);
		//	System.out.println(sImage);
		} 
		StdOut.printf("Total Images Trained: %d Correct: %d  Wrong: %d Accuracy %5.1f%%\n",aiTestLabels.length, iCorrect, aiTestLabels.length - iCorrect, (double)iCorrect/(double) aiTestLabels.length * 100.0);

    	}catch(Exception e){System.out.println(e);}   
	
	}
	
}
