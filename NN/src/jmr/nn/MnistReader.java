package jmr.nn;

/* https://golb.hplar.ch/2018/12/simple-neural-network.html
 * Building a simple neural network with Java and JavaScript
 *
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

import java.util.Arrays;


public class MnistReader {
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