package edu.nyu.unik.expriments;

import java.io.IOException;

import au.edu.rmit.trajectory.clustering.kmeans.kmeansAlgorithm;
import edu.wlu.cs.levy.cg.KeyDuplicateException;
import edu.wlu.cs.levy.cg.KeySizeException;


/*
 * to do
 * 1. 
 */
public class fmeans {

	public fmeans() {
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) throws IOException, KeySizeException, KeyDuplicateException {
		// TODO Auto-generated method stub
		int[] kvalue = new int[]{10, 50, 100, 200, 400, 600, 800, 1000};//10, 100, 1000
		int[] scales = new int[] {100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000}; //1000, 5000, 10000, 50000, 100000, 500000, 1000000
		int[] capacities = new int[] {10, 20, 30, 40, 50, 60};//capacity
		int[] dimensions = new int[] {2, 10, 20, 30, 40, 50};
		int[] groups = new int[] {10, 20, 30, 40, 50};
		
	//	dimensions = new int[] {7};
		kvalue = new int[]{10,100,1000};//10, 100, 1000
	//	kvalue = new int[]{10,20,30,40,50};//10, 100, 1000
	//	kvalue = new int[]{1000};//10, 100, 1000
	//	kvalue = new int[]{10, 100, 400, 700, 1000};//10, 100, 1000
		dimensions = new int[] {2, 10, 100, 200, 300, 400, 500}; //for 
		dimensions = new int[] {2};
		scales = new int[] {10000};
	//	scales = new int[] {1000, 5000, 10000, 50000, 100000, 500000, 1000000};
		capacities = new int[] {30};
	//	dimensions = new int[] {2};// test the parameters
		int testTime = 1;//test one time
		
	//	runkmeans.experiments(kvalue, testTime);// run the configuration that specified
		
		for(int kv: kvalue) {
			int []ka = new int[1];
			ka[0] = kv;
			kmeansAlgorithm<?> runkmeans = new kmeansAlgorithm<>(args);
			runkmeans.setGroupNumber(10);
			runkmeans.experiments(ka, testTime);// run the configuration that specified
		}
	
	//	scales = new int[]{scaleData};
		kvalue = new int[]{100};
		System.out.print(args[4]);
	//	for(int dimension: dimensions)
	//	for(int capacity: capacities)
	//	for(int scale: scales) 
		for(int group: groups)
		{
		//	if(dimension>dim || scale>scaleData)
		//		continue;
		//	runkmeans.setDimension(dimension);
		//	runkmeans.setCapacity(capacity);
		//	runkmeans.setScale(scale);
			kmeansAlgorithm<?> runkmeans1 = new kmeansAlgorithm<>(args);
			runkmeans1.setGroupNumber(group);
			runkmeans1.experiments(kvalue, testTime);
		}
		
		
		
	//	runkmeans.staticKmeans(false, true, false);//index sign, bound, scan whole tree again
		
	//	runkmeans.staticKmeans(true, false, true);//test the functionality		
		//run script to draw the figures using gnuplot
	}
}