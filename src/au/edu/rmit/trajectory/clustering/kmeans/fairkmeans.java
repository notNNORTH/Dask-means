package au.edu.rmit.trajectory.clustering.kmeans;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import com.google.common.collect.Range;
import com.google.j2objc.annotations.Weak;

import au.edu.rmit.trajectory.clustering.kpaths.Util;

public class fairkmeans {
	
	static double maxDistance = 30000; // for normalization
	static int userID[]; // label each point's belonging group
	static Map<Integer, Integer> userNumber = new HashMap<Integer, Integer>(); // number of points in each group for 
	
	static int[] getUserID() {
		return userID;
	}
	
	static Map<Integer, Integer> getuserNumber() {
		return userNumber;
	}
	
	static void readDataset(String rawdata, String poi, String newfile) throws IOException {
		File file = new File(poi);
		Map<String, String> poiMap = new HashMap<String, String>();
		try (BufferedReader br = new BufferedReader(new FileReader(file))) {
			String strLine;
			while ((strLine = br.readLine()) != null) {
				String[] splitString = strLine.split("\t");
				String countryString = splitString[4];
				if(countryString.equals("US"))// just present the results in US, we can also shrink the range
					poiMap.put(splitString[0], splitString[1]+","+splitString[2]);
			}
		}
		file = new File(rawdata);
		try (BufferedReader br = new BufferedReader(new FileReader(file))) {
			String strLine;
			
			while ((strLine = br.readLine()) != null) {
				String[] splitString = strLine.split("\t");
				if(poiMap.containsKey(splitString[1])) {
					Util.write(newfile, splitString[0]+","+poiMap.get(splitString[1])+"\n");
				}
			}
		}
	}
	
	/*
	 * preprocess the data, generate the normal k-means, group all the people's point,
	 */
	static void FairProcessData(double [][]originData, int userID[], Map<Integer, Double> userNumber, int dim, String file) {
		Map<Integer, double[]> summapMap = new HashMap<Integer, double[]>();
		for(int i=0; i<originData.length; i++) {
			double []point = originData[i];
			int id = userID[i];
			double sum[] = new double[dim];
			if(summapMap.containsKey(id)) {
				sum = summapMap.get(id);
			}
			for(int j=0; j<dim; j++) {
				sum[j] += point[j];
			}
			summapMap.put(id, sum);
		}
		
		for(int user: summapMap.keySet()) {
			double sum[] = summapMap.get(user);
			String content = "";
			for(int j=0; j<dim; j++) {
				content += Double.toString(sum[j]/userNumber.get(user))+",";
			}
			Util.write(file, content+"\n");
		}
	}
	
	static void ProcessNYCData(String nycdata, String newnycdata, String race) throws FileNotFoundException, IOException {
		HashMap<String, Integer> raceIDHashMap = new HashMap<>();
		int id = 1;
		int counter = 0;
		try (BufferedReader br = new BufferedReader(new FileReader(nycdata))) {
			String strLine;
			while ((strLine = br.readLine()) != null) {
				if(counter++==0)
					continue;
				String[] splitString = strLine.split(",");
				String group = splitString[0];
				int groupid = 0;
				if(!raceIDHashMap.containsKey(group)) {
					groupid = id;
					raceIDHashMap.put(group, id++);
				}else {
					groupid = raceIDHashMap.get(group);
				}
				double x = Double.valueOf(splitString[1]);
				double y = Double.valueOf(splitString[2]);
				Util.write(newnycdata, groupid+","+splitString[1]+","+splitString[2]+"\n");
			}
		}
		System.out.println(id); // we have 8 races in total
		for(String i:raceIDHashMap.keySet())
			Util.write(race, i+","+raceIDHashMap.get(i)+"\n");
	}
	/*
	 * we study the quality of generated clusters, distance vs frequency, show whether they are related in the normal k-means.
	 * 
	 * we design several metrics to prove it is good.
	 * 
	 * whether points in each cluster are euqality accessible to the centroids.
	 */
	void effectivenessStudy(ArrayList<cluster> CENTERSEuc, double[][] dataOriginal) {
		// we study the number of people in each group
		
		
		// we need to give multiple metric to reflect we give more weight to the under-represented person.
		
		// over-representation
		
		// the top-n closers whether are the frequent people.
		
		// whether every one can find the site within a given threshold.
		
		// we compute the KL divergence by selecting
		
		// we compute the divergence through python, we output the distance for each 
		
	}
	
	/*
	 * group all the points based on distance, 
	 * 
	 * and further assign the weight based on the number of points in each group
	 * 
	 * detect how many groups we need, maybe we can run k-means, or just fixed the gap as a parameter, there is clear gap.
	 * 
	 * we will 
	 */
	static void distanceGrouping(ArrayList<Double> distanceArrayList, int numberGroup, ArrayList<Double> distanceArrayListUnsorted) {
		double gap = distanceArrayList.get(0)/numberGroup;
		ArrayList<Integer> groupNumberArrayList = new ArrayList<Integer>();
		userID = new int[distanceArrayList.size()];
		int length = distanceArrayList.size();
		int counter = 0;
		int counter1 = numberGroup-1;
		for(double value: distanceArrayList) {// the distance is sorted 
			if(value>(counter1*gap))
				counter++;
			else {
				counter1--;
				userNumber.put(counter1, counter);
				System.out.println(counter);
				counter = 1;
			}
		}
		
		int pointid = 0;
		for(double value: distanceArrayListUnsorted) {
			int group = (int)(value/gap);
			userID[pointid++] = group; // label each point with its belonging group
		}
		// we assign the weight based on the groupNumberArrayList
	}
	// fair clustering evaluation
	
	static void GroupDistanceCDF(String distanceFile, String percentileFile) throws FileNotFoundException, IOException{
		HashMap<Integer, ArrayList<Double>> groupDistance = new HashMap<>();
		ArrayList<Double> distanceArrayList = new ArrayList<>();
		ArrayList<Double> distanceArrayListUnSorted = new ArrayList<>();
		// create an array list to store all the distance,
		// divide them into groups
		try (BufferedReader br = new BufferedReader(new FileReader(distanceFile))) {
			String strLine;
			while ((strLine = br.readLine()) != null) {
				String[] splitString = strLine.split("\t");
				int group = Integer.valueOf(splitString[0]);
				double distance = Double.valueOf(splitString[1]);
				distanceArrayList.add(distance);
				distanceArrayListUnSorted.add(distance);
				ArrayList<Double> arrayList = new ArrayList<>(); 
				if(groupDistance.containsKey(group)) {
					arrayList = groupDistance.get(group);
				}
				arrayList.add(distance);
				groupDistance.put(group, arrayList);
			}
		}
		Collections.sort(distanceArrayList, Collections.reverseOrder());
		distanceGrouping(distanceArrayList, 10, distanceArrayListUnSorted);
	//	System.out.println(distanceArrayList.get(0));
		
		
		// get the percentile, 100, and count how many of them are less than the 
		double unit = maxDistance/100.0;
	//	System.out.println(maxDistance);
		
		double percentile[][] = new double[groupDistance.size()][100];
		int counter = 0;
		for(int group: groupDistance.keySet()) {
			double maxx = 0;
			double minn = Double.MAX_VALUE;
			double average = 0;
			ArrayList<Double> aDoubles = groupDistance.get(group);
			for(double dis: aDoubles) {
				if(maxx<dis)
					maxx = dis;
				if(minn>dis)
					minn = dis;
				average+=dis;
				int order = (int)(dis/unit);// maybe we can use the exponential to check.
				for(int i=order; i<100; i++)
					percentile[counter][i]++;
			}
			average/=aDoubles.size();
			for(int i=0; i<100; i++) {
				percentile[counter][i]/=aDoubles.size();
			//	System.out.println(percentile[counter][i]);
			}
			System.out.println("size: "+aDoubles.size()+",max: "+maxx+","+"min: "+minn+",average: "+average);
			counter++;
		}
		for(int i=0; i<100; i++) {
			String content = Integer.toString(i);
			for(int j=0; j<counter; j++) {
				content+=","+percentile[j][i];
			}
			Util.write(percentileFile, content+"\n");
		}
		
		// we can prune those small groups and outliers, 
		// calculate the group divergence, 
		double maxgap = 0;
		for(int i=0; i<100; i++) {
			double maxvalue = 0, minvalue = Double.MAX_VALUE;
			for(int j=0; j<counter; j++) {
				if(maxvalue < percentile[j][i])
					maxvalue = percentile[j][i];
				if(minvalue > percentile[j][i])
					minvalue = percentile[j][i];
			}
			double gap = maxvalue - minvalue;
			if(gap > maxgap)
				maxgap = gap;
		}
		System.out.println("CDF gap: "+maxgap);
		
		
		//calculate more features, each cluster, the ratio, the cluster, conduct a full evaluation
	}
				
	// We can foresee that normal method will lead to unfair planning, all go to dense areas, check-in, taxi trips dataset all have this bias.

	
	public static void main(String[] args) throws IOException {
	//	readDataset("/Users/sw160/Documents/datasets/dataset_WWW2019/raw_Checkins_anonymized.txt", "/Users/sw160/Documents/datasets/dataset_WWW2019/raw_POIs.txt", 
	//			"/Users/sw160/Documents/datasets/dataset_WWW2019/clustering.txt");
	//	GroupDistanceCDF("/Users/sw160/Downloads/torch-clus/logs/fairkmeans/1000Distance.txt");
	//	GroupDistanceCDF("/Users/sw160/Downloads/torch-clus/logs/fairkmeans/1000fairDistance.txt");
	//	ProcessNYCData("/Users/sw160/Downloads/torch-clus/dataset/FairPseudoSynthDemogData.csv", "/Users/sw160/Downloads/torch-clus/dataset/NYC8millions.csv",
	//			"/Users/sw160/Downloads/torch-clus/dataset/NYCRace.csv");
		GroupDistanceCDF("/Users/sw160/Downloads/torch-clus/logs/fairkmeansNYC/1000Distance99.txt", "/Users/sw160/Downloads/torch-clus/logs/fairkmeansNYC/1000Distance99Percentile.csv");
		GroupDistanceCDF("/Users/sw160/Downloads/torch-clus/logs/fairkmeansNYC/1000fairDistance99.txt", "/Users/sw160/Downloads/torch-clus/logs/fairkmeansNYC/1000fairDistance99Percentile.csv");
	}
	
	// comparisons with k-means
	public void runExperiments() {
		
		// run normal k-means
		
		// run f-means with various configurations
		
		// configure datasets, 
		
		// configure comparisons
		
		// configure results, log the running time, means, and variance for the plotting.
		
		// 
	}
}
