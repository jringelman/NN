package jmr.util;


import java.util.*;
import java.io.*;
import java.io.FileNotFoundException;
import java.io.IOException;


public class AppProperties {
    private static Properties m_properties = new Properties();

    public static void loadProperties(String sPropertiesFile) throws Exception// FileNotFoundException,  IOException 
    {
      FileInputStream fis = null;
      try{
			fis = new FileInputStream(sPropertiesFile);
		}
		catch(FileNotFoundException fnfex)
		{
			throw new Exception("Properties File not found:" + sPropertiesFile);
		}
		try
		{
			m_properties.load(fis);
		}
		catch(IOException ioe)
		{
			throw new Exception("unable to load prop file=" + sPropertiesFile);
		}
    }
    
    public static String getProperty(String sKey) throws Exception 
    {
      String sValue = m_properties.getProperty(sKey);
      if (sValue == null)
			//throw new CodedException(CodedException.shTYPE_PROP_MISSING, "property=" + sKey);
			throw new Exception("missing property, key=" + sKey);

		int iIdxStart = 0;
		while ((iIdxStart = sValue.indexOf('[')) >= 0)
		{
			int iIdxStop = sValue.indexOf(']');
			if (iIdxStop == -1)
//            throw new CodedException(CodedException.shTYPE_PROP_INVALID, "terminating brace missing - property=" + sKey);
				throw new Exception("invalid property, terminating brace missing, key=" + sKey);
			else
         {
            String sNestedProp = m_properties.getProperty(sValue.substring(iIdxStart+1,iIdxStop));
            sValue = sValue.substring(0,iIdxStart)+ sNestedProp +  sValue.substring(iIdxStop+1);
         }
      }
      return sValue.trim();
    }
    
    private AppProperties() {
    }
    
    public static void test()
    {
    	
    }
}
