package jmr.nn;

/* LEVERAGE THIS ARTICLE AND CODE FOR ACCESSING MINST DATA
 * Building a simple neural network with Java and JavaScript
 * https://golb.hplar.ch/2018/12/simple-neural-network.html
 * https://github.com/ralscha/blog/tree/master/mnist
 */

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
//import java.io.FileInputStream;
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

	public static void runMinstDataSet()
	{
		System.out.println("BEGINNING NEURAL NETWORK ON MNIST DATA");

		double dBias = 0.1; 
		final double dLEARNING_RATE = 0.1;
		final int iNBR_INPUTS = 784; //28 x 28 pixels = 784 pixels
		final int iNBR_NEURONS_LAYER0 = 20; // THIS IN THE ONLY HIDDEN LAYER
		final int iNBR_NEURONS_LAYER1 = 10; // THIS IS OUPUT LAYER WITH EACH NODE FOR DIGITS 0 THRU 9;
		final int iNBR_EPOCHS = 1;
		
		NNLayer [] aLayer = new NNLayer[2];
		//	public Layer(int nLayerNbr,  int nNbrInputs, int nNbrNeurons, double dBias)
	    aLayer[0] = new NNLayer(0, iNBR_INPUTS, iNBR_NEURONS_LAYER0, dBias);
	    aLayer[1] = new NNLayer(1, iNBR_NEURONS_LAYER0, iNBR_NEURONS_LAYER1, dBias);
	    NeuralNetwork nn = new NeuralNetwork(aLayer, dLEARNING_RATE);

    	String sFileTrainLabels = m_sDATA_PATH + "train-labels-idx1-ubyte.gz";
    	String sFileTrainImages = m_sDATA_PATH + "train-images-idx3-ubyte.gz";

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
    	String sFileTestLabels = m_sDATA_PATH + "t10k-labels-idx1-ubyte.gz";
    	String sFileTestImages = m_sDATA_PATH + "t10k-images-idx3-ubyte.gz";

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
		//	ArrayUtil.show(adOutput, "NN Output", "%8.5f");
		//	String sImage = MnistReader.renderImage(aaiImage);
		//	System.out.println(sImage);
		} 
		StdOut.printf("Total Images Tested: %d Correct: %d  Wrong: %d Accuracy %5.1f%%\n",aiTestLabels.length, iCorrect, aiTestLabels.length - iCorrect, (double)iCorrect/(double) aiTestLabels.length * 100.0);

    	}catch(Exception e){System.out.println(e);}   
		System.out.println("COMPLETED NEURAL NETWORK ON MNIST DATA\n");
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