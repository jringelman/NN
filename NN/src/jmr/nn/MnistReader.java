package jmr.nn;

/* LEVERAGE THIS ARTICLE AND CODE FOR ACCESSING MINST DATA
 * Building a simple neural network with Java and JavaScript
 * https://golb.hplar.ch/2018/12/simple-neural-network.html
 * https://github.com/ralscha/blog/tree/master/mnist
 */

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

import jmr.util.AppProperties;
import jmr.util.ArrayUtil;
import jmr.util.StdOut;

import java.util.Arrays;

public class MnistReader {
	
	static final String m_sDATA_PATH = "./data/mnist/";
	
	static final String sFILE_TRAIN_LABELS = "train-labels-idx1-ubyte.gz";
	static final String sFILE_TRAIN_IMAGES = "train-images-idx3-ubyte.gz";
	static final String sFILE_TEST_LABELS = "t10k-labels-idx1-ubyte.gz";
	static final String sFILE_TEST_IMAGES = "t10k-images-idx3-ubyte.gz";
	static final int iREQUIRED_INPUTS = 784;
	static final int iREQUIRED_OUTPUTS_NEURONS = 10;


	public static void runMinstDataSet()
	{
		System.out.println("\nBEGINNING NEURAL NETWORK ON MNIST DATA");

		//CREATE NN FROM PROPS FILE
		NeuralNetwork nn =  NeuralNetwork.loadNNFromPropertiesFile("mnist");
		System.out.println("NN Created: " + nn.getDescription());

		//VERIFY THE NUMBER OF INPUTS AND THE NBR OF OUTPUT NEURONS
		if(nn.getNbrInputs() != iREQUIRED_INPUTS){
			StdOut.printf("ERROR! MNIST requires %d inputs; Properties file has %d inputs. Exiting NN run",iREQUIRED_INPUTS,nn.getNbrInputs());
			return;
		}
		int iNbrNeuronsOutputLayer = nn.getNbrNeuronsInLayer(nn.getNbrLayers()-1);
		if(iNbrNeuronsOutputLayer != iREQUIRED_OUTPUTS_NEURONS){
			StdOut.printf("ERROR! MNIST requires %d output neurons; Properties file has %d neurons in output layer. Exiting NN run",iREQUIRED_OUTPUTS_NEURONS,iNbrNeuronsOutputLayer);
			return;
		}
	

    	try{  		
    //LOAD TRAINING IMAGES AND LABELS FROM MNIST DATA FILES	
    	String sFileTrainLabels = m_sDATA_PATH + sFILE_TRAIN_LABELS;
        String sFileTrainImages = m_sDATA_PATH + sFILE_TRAIN_IMAGES;
 
        StdOut.printf("Loading Training Labels: " + sFileTrainLabels + "\n");
        int[] aiTrainLabels = MnistReader.getLabels(Paths.get(sFileTrainLabels));
        StdOut.printf("Loading Training Images: " + sFileTrainImages + "\n");
        List<int[][]> listTrainImages = MnistReader.getImages(Paths.get(sFileTrainImages));
    
        StdOut.printf("Loaded Mnist Training Data: %d Images and %d Labels\n", listTrainImages.size(), aiTrainLabels.length);
		
    //GET NBR EPOCHS FROM PROPERTIES FILE
		int iNbrEpochs = Integer.parseInt(AppProperties.getProperty("nn.mnist.epochs"));
		System.out.println("\nBeginning " + iNbrEpochs + " epochs of training");
		
	//TRAIN THE NETWORK EP0CHS TIMES
		int iErrorsEpoch = 0;

		for (int iEpoch=0; iEpoch<iNbrEpochs; iEpoch++)
		{
			iErrorsEpoch =0;
			for (int i=0; i<aiTrainLabels.length; i++) {
				//iCorrectSet = 0;
				//CREATE TARGET ARRAY FOR THIS EPOCH; CONVERTS DIGITS TO 10 DIGIT ARRAY OF DOUBLES 
				double[] adTarget = MnistReader.createTarget(aiTrainLabels[i]);
			
				//GET 2 DIMENSION IMAGE, FLATTEN TO 1 DIMENSION, THEN SCALE VALUES TO 0.0 to 1.0 RANGE
				int[][] aaiImage = listTrainImages.get(i);
				int [] aiImageFlat = MnistReader.flat(aaiImage);
				double[] adInput = MnistReader.scaleImagePixels(aiImageFlat);
//				System.out.println(MnistReader.renderImage(aaiImage));
		
				//TRAIN NN WITH INPUTS AND TARGETS
				double [] adOutput = nn.trainNetwork(adInput, adTarget);
				
				int iNNGuess = ArrayUtil.maxValueIndex(adOutput);
				if(iNNGuess !=  aiTrainLabels[i]) {
					iErrorsEpoch++;
				}
				
				if(((i+1) % 10000) == 0)				{
					StdOut.printf("%d Images trained for Epoch %d;  Error Rate=%5.1f%%\n",(i+1),iEpoch+1, (100.0 * iErrorsEpoch)/(double) i);
				}
			}
			StdOut.printf("Epoch %d completed with Error Rate= %5.1f%%\n\n", iEpoch+1, (100.0 * iErrorsEpoch)/(double)aiTrainLabels.length);
		} 
	  	}catch(Exception e){System.out.println(e);}   

    //RUN THE MINST TEST DATA

    	try{
         StdOut.printf("Beginning Testing Phase\n");

    //LOAD TESTING IMAGES AND LABELS FROM MNIST DATA FILES	
       	String sFileTestLabels = m_sDATA_PATH + sFILE_TEST_LABELS;
       	String sFileTestImages = m_sDATA_PATH + sFILE_TEST_IMAGES;

        StdOut.printf("Loading Testing Labels: " + sFileTestLabels + "\n");
        int[] aiTestLabels = MnistReader.getLabels(Paths.get(sFileTestLabels));
        StdOut.printf("Loading Testing Images: " + sFileTestImages + "\n");
    	List<int[][]> listTestImages = MnistReader.getImages(Paths.get(sFileTestImages));
        StdOut.printf("Loaded Mnist Test Data: %d Images and %d Labels\n", listTestImages.size(), aiTestLabels.length);

		int iTestErrors =0;
		
		for (int i=0; i<aiTestLabels.length; i++) {	
			//GET 2 DIMENSION IMAGE, FLATTEN TO 1 DIMENSION, THEN SCALE VALUES TO 0.0 to 1.0 RANGE
			int[][] aaiImage = listTestImages.get(i);
			int [] aiImageFlat = MnistReader.flat(aaiImage);
			double[] adInput = MnistReader.scaleImagePixels(aiImageFlat);

			//QUERY THE NN WITH THE INPUT IMAGE DATA
			double [] adOutput = nn.query(adInput);
			
			//SEE IF THE NN GUESS OF OUTPUT IS CORRECT
			int iNNGuess = ArrayUtil.maxValueIndex(adOutput);
			if(iNNGuess !=  aiTestLabels[i])
				iTestErrors++;
		
		//	System.out.println(MnistReader.renderImage(aaiImage));
		//	StdOut.printf("Test Label Target: %d  NN Guess: %d\n", aiTestLabels[i], iNNGuess);
		//	ArrayUtil.show(adOutput, "NN Output", "%8.5f");
		} 
		StdOut.printf("Tested %d Images with Error Rate of %5.1f%%\n",aiTestLabels.length, (double)iTestErrors/(double) aiTestLabels.length * 100.0);

    	}catch(Exception e){System.out.println(e);}   
		System.out.println("\nCOMPLETED NEURAL NETWORK ON MNIST DATA\n");
	}
	
	
	public static int[] getLabels(Path labelsFile) throws IOException {
		ByteBuffer bb = ByteBuffer.wrap(decompress(Files.readAllBytes(labelsFile)));
		if (bb.getInt() != 2049) {
			throw new IOException("not a labels file");
		}

		int numLabels = bb.getInt();
		int[] labels = new int[numLabels];

		for (int i = 0; i < numLabels; i++) {
			labels[i] = bb.get() & 0xFF;
		}
		return labels;
	}

	public static List<int[][]> getImages(Path imagesFile) throws IOException {
		ByteBuffer bb = ByteBuffer.wrap(decompress(Files.readAllBytes(imagesFile)));
		if (bb.getInt() != 2051) {
			throw new IOException("not an images file");
		}

		int numImages = bb.getInt();
		int numRows = bb.getInt();
		int numColumns = bb.getInt();
		List<int[][]> images = new ArrayList<>();

		for (int i = 0; i < numImages; i++) {
			int[][] image = new int[numRows][];
			for (int row = 0; row < numRows; row++) {
				image[row] = new int[numColumns];
				for (int col = 0; col < numColumns; ++col) {
					image[row][col] = bb.get() & 0xFF;
				}
			}
			images.add(image);
		}

		return images;
	}

	private static byte[] decompress(final byte[] input) throws IOException {
		try (ByteArrayInputStream bais = new ByteArrayInputStream(input);
				GZIPInputStream gis = new GZIPInputStream(bais);
				ByteArrayOutputStream out = new ByteArrayOutputStream()) {
			byte[] buf = new byte[8192];
			int n;
			while ((n = gis.read(buf)) > 0) {
				out.write(buf, 0, n);

			}
			return out.toByteArray();
		}
	}

	public static String renderImage(int[][] image) {
		StringBuilder sb = new StringBuilder();
		int threshold1 = 256 / 3;
		int threshold2 = 2 * threshold1;

		for (int[] element : image) {
			sb.append("|");
			for (int pixelVal : element) {
				if (pixelVal == 0) {
					sb.append(" ");
				}
				else {
					if (pixelVal < threshold1) {
						sb.append(".");
					}
					else {
						if (pixelVal < threshold2) {
							sb.append("x");
						}
						else {
							sb.append("X");
						}
					}
				}
			}
			sb.append("|\n");
		}

		return sb.toString();
	}
	
	//SCALE IMAGE PIXEL DATA FROM 0 to 255 int TO 0.001 to 1.0 double
	public static double[] scaleImagePixels(int[] aiImagePixels) {
		double[] adImgPixels = new double[aiImagePixels.length];
		for (int i = 0; i < aiImagePixels.length; i++) {
			adImgPixels[i] = aiImagePixels[i] / 255.0 * 0.999 + 0.001;
		}
		return adImgPixels;
	}
	
	public static int[] flat(int[][] i) {
		int[] result = new int[i.length * i[0].length];
		for (int r = 0; r < i.length; r++) {
			int[] row = i[r];
			System.arraycopy(row, 0, result, r * row.length, row.length);
		}
		return result;
	}
	
	public static double[] createTarget(int iLabel) {
		double [] adTarget = new double [10];
		Arrays.fill(adTarget,0.001);
		adTarget[iLabel] = 0.999;
		return adTarget;
	}
	
	


	
//**************************************************
//************ STATIC TEST METHODS *****************
//**************************************************
	
	public static void test()
	{
		
		int [] aiTest = {0, 100, 255};
		double [] adResult = MnistReader.scaleImagePixels(aiTest);
	
		double [] adTarget = MnistReader.createTarget(0);
		adTarget = MnistReader.createTarget(1);
		adTarget = MnistReader.createTarget(2);
		adTarget = MnistReader.createTarget(3);
		adTarget = MnistReader.createTarget(4);
		adTarget = MnistReader.createTarget(5);
		adTarget = MnistReader.createTarget(6);
		adTarget = MnistReader.createTarget(7);
		adTarget = MnistReader.createTarget(8);
		adTarget = MnistReader.createTarget(9);

		
	
	    	String sFileTestLabels = "/Users/JMR/Dropbox/projects/data/mnist/t10k-labels-idx1-ubyte.gz";
	    	String sFileTestImages = "/Users/JMR/Dropbox/projects/data/mnist/t10k-images-idx3-ubyte.gz";
	    	String sFileTrainLabels = "/Users/JMR/Dropbox/projects/data/mnist/train-labels-idx1-ubyte.gz";
	    	String sFileTrainImages = "/Users/JMR/Dropbox/projects/data/mnist/train-images-idx3-ubyte.gz";

	    	 try{
	    		 
	    	int[] aiTrainLabels = MnistReader.getLabels(Paths.get(sFileTrainLabels));
	    	int[] aiTestLabels = MnistReader.getLabels(Paths.get(sFileTestLabels));

	    	List<int[][]> listTrainImages = MnistReader.getImages(Paths.get(sFileTrainImages));
	    	List<int[][]> listTestImages = MnistReader.getImages(Paths.get(sFileTestImages));
	    	
			System.out.println(aiTrainLabels.length + " Train Labels");
			System.out.println(listTrainImages.size() + " Train Images");
			System.out.println(aiTestLabels.length + " Test Labels");
			System.out.println(listTestImages.size() + " Test Images");
	    	
			System.out.println("Train Labels");
			for (int i=0; i<50; i++)
				System.out.print(aiTrainLabels[i]);
			System.out.println("");
			System.out.println("");

			System.out.println("Train Data");
			int[][] image  = listTrainImages.get(0);
			for (int row = 0; row < 28; row++)
			{
				for (int col = 0; col < 28; ++col) 
						System.out.print(image[row][col]); 
				System.out.println();
			}
			System.out.println();

			//System.out.print(listTrainImages.);
			//System.out.println("");
		    	}catch(Exception e){System.out.println(e);}   
	}
}