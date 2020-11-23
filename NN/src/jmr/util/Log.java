package jmr.util;

import java.io.*;
import java.text.*;
import java.util.*;

public class Log
{
   protected static final String sTYPE_TRACE = "trace";
	protected static final String sTYPE_ERROR = "ERROR";

	synchronized public static String getThreadTag()
	{
//		StringBuffer sb = new StringBuffer("               ");
  //		sb.replace(0,14,Thread.currentThread().getName());
	 //	return sb.toString();
	 	return Thread.currentThread().getName();

	}

	synchronized public static void writePlain(String sMsg)
	{
		System.out.println(sMsg);
		m_printWriter.println(sMsg);
	}

	synchronized public static void error(String sMsg)
	{
		String sOutput =sTYPE_ERROR + " " + getThreadTag() + " " + sMsg;
      System.out.println(sOutput);
      m_printWriter.println(getTimeStamp() + " " + sOutput);
   }

   synchronized public static void error(Exception exception)
	{
		Log.error(exception, true);
	}
	synchronized public static void error(Exception exception, boolean bPrintStackTrace)
	{
		String sOutput =sTYPE_ERROR + " " + getThreadTag() + " " + "Exception Occured.";
		System.out.println(sOutput);
		m_printWriter.println(getTimeStamp() + " " + sOutput);

		String sExceptionMsg = "Exception Msg: " + exception.getMessage();
		System.out.println(sExceptionMsg);
		m_printWriter.println(sExceptionMsg);
		if(bPrintStackTrace)
		{
			exception.printStackTrace(m_printWriter);
			exception.printStackTrace();
		}
	}

	synchronized public static void error(Exception exception, String sMsg)
	{
		Log.error(exception, true, sMsg);
	}

	synchronized public static void error(Exception exception, boolean bPrintStackTrace, String sMsg)
	{
		String sOutput =sTYPE_ERROR + " " + getThreadTag() + " " + sMsg;
		System.out.println(sOutput);
		m_printWriter.println(getTimeStamp() + " " + sOutput);

		String sExceptionMsg = "Exception Msg: " + exception.getMessage();
		System.out.println(sExceptionMsg);
		m_printWriter.println(sExceptionMsg);

		if(bPrintStackTrace)
		{
			exception.printStackTrace(m_printWriter);
			exception.printStackTrace();
		}
   }

   synchronized public static void trace(String sMsg)
   {
      String sOutput =sTYPE_TRACE + " " + getThreadTag() + " " + sMsg;
      System.out.println(sOutput);
      m_printWriter.println(getTimeStamp() + " " + sOutput);
   }

   synchronized public static void open(String sFileName) throws IOException
   {
      File file = new File(sFileName);
      file.getParentFile().mkdirs();
      m_printWriter = new PrintWriter(new FileWriter(sFileName), true);
   }

   synchronized public static void close() throws IOException
   {
      m_printWriter.close();
      m_printWriter = null;
   }

   synchronized public static boolean isOpen(){return m_printWriter != null;}

   protected static String getTimeStamp()
   {
      return m_dateFormat.format(new Date());
   }

   //PRIVATE SO THAT CANNOT CREATE INSTANCE
   private Log(){}

   private static PrintWriter m_printWriter = null;
 //  protected final static DateFormat m_dateFormat = DateFormat.getTimeInstance(DateFormat.MEDIUM);
   protected final static SimpleDateFormat m_dateFormat = new SimpleDateFormat("[yyyy/MM/dd kk:mm:ss]");
}


