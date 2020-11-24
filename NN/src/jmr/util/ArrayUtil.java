package jmr.util;

import java.util.Arrays;

public class ArrayUtil {
	
	public ArrayUtil(){
	}
	
	//public static void show (double [] ad, String sName)	{
	//	show(ad,sName, "%9.5f");
	//}
	
	public static void show (double [] ad, String sName, String sFormat)
	{
		//sFormat += "\n";
    	StdOut.println(sName);
		for (int i=0; i<ad.length; i++){
			StdOut.printf(sFormat, ad[i]);
		}
		StdOut.println();
	}
	
	public static void show (double [][] aad, String sName, String sFormat)
	{
    	StdOut.println(sName);
        for (int i = 0; i < aad.length; i++) {
            for (int j = 0; j < aad[0].length; j++) 
                StdOut.printf(sFormat, aad[i][j]);
            StdOut.println();
        }
		StdOut.println();
	} 
	
	public static void showFlat (double [] ad, String sName, String sFormat)
	{
    	StdOut.printf(sName +" {");
		for (int i=0; i<ad.length; i++){
			StdOut.printf(sFormat, ad[i]);
            if (i < ad.length-1)
                StdOut.print(",");
		}
		StdOut.println("}");
	}

	public static void showFlat (double [][] aad, String sName, String sFormat)
	{
    	StdOut.print(sName + " {");
    	for (int i = 0; i < aad.length; i++) {
    		StdOut.print("{"); 
    		for (int j = 0; j < aad[0].length; j++) { 
                StdOut.printf(sFormat, aad[i][j]);
                if (j < aad[0].length-1)
                    StdOut.print(",");
            }
            StdOut.print("}");
            if (i < aad.length-1)
                StdOut.print(",");
	    }
		StdOut.println("}");
	} 

	
	//MULTIPLIES EACH ELEMENT OF DOUBLE [] BY DOUBLE VALUE; RETURNS NEW ARRAY; DOES NOT MODIFY ORIGINAL ARRAY.
	public static double [] times(double[] ad, double d)
	{
		double [] adRet = new double[ad.length];
		for (int i=0; i<ad.length; i++) {
			adRet[i] = ad[i] * d;
		}
		return adRet;
	}
	
	public static double [] plus(double[] ad, double d)
	{
		double [] adRet = new double[ad.length];
		for (int i=0; i<ad.length; i++) {
			adRet[i] = ad[i] + d;
		}
		return adRet;
	}
	public static double [] plus(double[] ad1, double [] ad2)
	{
		if (ad1.length != ad2.length) throw new RuntimeException("Mismatch params in ArrayUtil.simpleTimes");

		double [] adRet = new double[ad1.length];
		for (int i=0; i<ad1.length; i++) 
			adRet[i] = ad1[i] + ad2[i];
		return adRet;
	}
	
	public static double [] minus(double[] ad, double d)
	{
		double [] adRet = new double[ad.length];
		for (int i=0; i<ad.length; i++) 
			adRet[i] = ad[i] - d;
		return adRet;
	}
	
	public static double [] minus(double[] ad1, double [] ad2)
	{
		if (ad1.length != ad2.length) throw new RuntimeException("Mismatch params in ArrayUtil.simpleTimes");

		double [] adRet = new double[ad1.length];
		for (int i=0; i<ad1.length; i++) 
			adRet[i] = ad1[i] - ad2[i];
		return adRet;
	}

	
	public static double [][] simpleTimes(double[][] aad, double[] ad)
	{
		if (aad.length != ad.length) throw new RuntimeException("Mismatch params in ArrayUtil.simpleTimes");

		double [][] aadRet = new double[aad.length][aad[0].length];
		for (int i = 0; i < aad.length; i++) 
            for (int j = 0; j < aad[0].length; j++) 
            	aadRet[i][j] = aad[i][j] * ad[i];
		return aadRet;
	}
	
	public static double [][] plus(double[][] aad1, double[][] aad2)
	{
		if (aad1.length != aad2.length) throw new RuntimeException("Mismatch params in ArrayUtil.simpleTimes");
		if (aad1[0].length != aad2[0].length) throw new RuntimeException("Mismatch params in ArrayUtil.simpleTimes");

		double [][] aadRet = new double[aad1.length][aad1[0].length];
		for (int i = 0; i < aad1.length; i++) 
            for (int j = 0; j < aad1[0].length; j++) 
            	aadRet[i][j] = aad2[i][j] + aad2[i][j];
		return aadRet;
	}
	
	public static double maxValue(double [] ad)
	{
		double dMaxValue = -Double.MAX_VALUE;
		for (int i=0; i<ad.length; i++)
			dMaxValue = (dMaxValue > ad[i]) ? dMaxValue : ad[i];
		return dMaxValue;
	}
	
	public static int maxValueIndex(double [] ad)
	{
		int nMaxValueIndex = -1;
		double dMaxValue = -Double.MAX_VALUE;
		for (int i=0; i<ad.length; i++)
			if (dMaxValue < ad[i]) {
				dMaxValue = ad[i];
				nMaxValueIndex = i;
			}
		return nMaxValueIndex;
	}


/***********  STATIC TEST METHOD ********************/
	public static void test()
	{
		double [][] aad6 = {{1.1,2.2},{3.3, 4.4},{5.5,6.6}};
		ArrayUtil.show(aad6, "aad6 NORMAL", "%8.5f");
		StdOut.println();
		ArrayUtil.showFlat(aad6, "FLAT DOUBLE ARRAY", "%9.5f");
		StdOut.println();

		double [] ad3 = {1.0, 2.22222, 3.33, 4.444444444444};
	//	ArrayUtil.show(ad, "Test Single Array", "%9.5f ");
	//	ArrayUtil.show(ad, "Test Single Array");
		ArrayUtil.show(ad3, "Test Single Array", "%8.5f");
		StdOut.println();
		ArrayUtil.showFlat(ad3, "Test Single Array - flat", "%8.5f");
		StdOut.println();

		//double [][] aad = {{1.11,222.22},{3.3333, 444444444.4},{555555.55,66.6666}};
		//ArrayUtil.show(aad, "double array", "%9.5f ");
	
		//double [] ad2 = {1.0, 2.2, 3.3, 4.4};
		//ArrayUtil.show(ad2, "ad2", "%9.5f ");	
		//double [] ad3 = ArrayUtil.times(ad2, 2.0);
		//ArrayUtil.show(ad3, "ad3", "%9.5f ");	
		//ArrayUtil.show(ad2, "ad2", "%9.5f ");	
		
		double [][] aad = {{1.1,2.2},{3.3, 4.4},{5.5,6.6}};
		double [] ad = {1.0, 2.0, 3.0};
		double [][] aaRet = ArrayUtil.simpleTimes(aad, ad);
		ArrayUtil.show(aad, "aad", "%9.5f ");
		ArrayUtil.show(ad, "ad", "%9.5f ");
		ArrayUtil.show(aaRet, "aaRet", "%9.5f ");

		double [] ad5 = {1.0, 2.22222, 3.33, 4.444444444444, -1.5, 2.3};
		double dMaxValue = ArrayUtil.maxValue(ad5);
		System.out.println(Arrays.toString(ad5));
		System.out.println(dMaxValue);
		int nMaxValueIndex = ArrayUtil.maxValueIndex(ad5);
		System.out.println(nMaxValueIndex);
		
	}
	
}
