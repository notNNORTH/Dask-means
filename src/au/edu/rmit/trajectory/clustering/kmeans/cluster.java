package au.edu.rmit.trajectory.clustering.kmeans;

import java.util.*;

import au.edu.rmit.trajectory.clustering.kpaths.Util;

public class cluster{
	protected Set<Integer> coveredPoints;
	protected Set<indexNode> covertedNodes;
	int dimension = 0;		//the dimension of the Euclidean dataset
	protected double []finalCentroid;	// centroid from previous iteration
	protected double []sumTotal;	//this records the sum
	protected double sumdistance=0;
	protected boolean centerChanged;
	PriorityQueue<kmeansHeap> heap;
	
	public double []getcentroid(){
		return finalCentroid;
	}
	
	public void setHeap(PriorityQueue<kmeansHeap> heap) {
		this.heap = heap;
	}
	
	public boolean emptyheap() {
		if(heap!=null)
			return heap.isEmpty();
		else
			return true;
	}
	
	public void clearSet() {
		coveredPoints = new HashSet<Integer>();
		covertedNodes = new HashSet<indexNode>();
		finalCentroid = new double[dimension];
		sumTotal = new double[dimension];
	}
	
	public void clearSetPICK() {
		coveredPoints = new HashSet<Integer>();
		covertedNodes = new HashSet<indexNode>();
		sumTotal = new double[dimension];
	}
	
	public boolean push(kmeansHeap aHeap) {
		return heap.add(aHeap);
	}
	
	public kmeansHeap poll() {
		return heap.poll();
	}
	
	public Set<Integer> getcoveredPoints(){
		return coveredPoints;
	}
	
	public void setcoveredPoints(ArrayList<Integer> arrayList){
		coveredPoints = new HashSet<>();
		if(arrayList!=null)
			coveredPoints.addAll(arrayList);
	}
	
	public Set<indexNode> getcoveredNodes(){
		return covertedNodes;
	}
		
	public void initializeHeap(){
		heap = new PriorityQueue<kmeansHeap>();//this is for storing the bounds, Harmly
	}
	
	
	/*
	 * add the trajectory into the clusters
	 */
	void mergePointToCluster(ArrayList<Integer> index){
		coveredPoints.addAll(index);
	}
	
	/*
	 * add the trajectory into the clusters
	 */
	void mergeNodesToCluster(ArrayList<indexNode> nodes){
		for(indexNode ad: nodes) {
			addNode(ad);
		//	addSum(ad.getSum());
		}
	}
	
	/*
	 * add the trajectory into the clusters
	 */
	void addPointToCluster(int index, double[] tra){
		if(!coveredPoints.contains(index)) {
			coveredPoints.add(index);
			addSum(tra);
		}
	}
	
	/*
	 * add the trajectory into the clusters
	 */
	void removePointToCluster(int index, double[] tra){
		if(!coveredPoints.contains(index)) {
			coveredPoints.remove(index);
			minusSum(tra);
		}
	}
	
	/*
	 * add the trajectory into the clusters
	 */
	void removePointToCluster(int index){
		if(coveredPoints.contains(index)) {
			coveredPoints.remove(index);
		}
	}
	
	/*
	 * add the trajectory into the clusters
	 */
	void addPointToCluster(int index){
		if(!coveredPoints.contains(index)) {
			coveredPoints.add(index);
		}
	}
	
	/*
	 * remove the trajectory into the clusters
	 */
	void removePointToCluster(ArrayList<Integer> index){
		coveredPoints.removeAll(index);
	}
	
	/*
	 * remove the trajectory into the clusters
	 */
	void deleteSinglePointToCluster(int index){
		coveredPoints.remove(index);
	}
	
	public cluster(double[] cluster, ArrayList<Integer> points, int dimen, double[][] datamapEuc) {
		dimension = dimen;
		finalCentroid = new double[dimension];
		sumTotal =  new double[dimension];
		for(int i=0; i<dimension; i++) {
			finalCentroid[i] = cluster[i];
		}
		coveredPoints = new HashSet<>();
		for(int i: points) {
			coveredPoints.add(i);		// assign the points to the cluster randomly
			addSum(datamapEuc[i-1]);
		}
		covertedNodes = new HashSet<>();
		heap = new PriorityQueue<kmeansHeap>();//this is for storing the bounds, Harmly
	}
	
	
	public cluster(double[] cluster, int n, int start, int end, int dimen, double[][] datamapEuc) {
		dimension = dimen;
		finalCentroid = new double[dimension];
		sumTotal =  new double[dimension];
	//	finalCentroid = cluster;
		for(int i=0; i<dimension; i++) {
			finalCentroid[i] = cluster[i];
		}
		coveredPoints = new HashSet<>();
		for(int i=start; i<end; i++) {
			coveredPoints.add(i+1);		// assign the points to the cluster randomly
			addSum(datamapEuc[i]);
		}
		covertedNodes = new HashSet<>();
		heap = new PriorityQueue<kmeansHeap>();//this is for storing the bounds, Harmly
	}
	
	public cluster(double[] cluster, indexNode node, int dimen) {
		dimension = dimen;
		finalCentroid = new double[dimension];
		sumTotal =  new double[dimension];
		for(int i=0; i<dimension; i++) {
			finalCentroid[i] = cluster[i];
			sumTotal[i] = 0;
		}
		coveredPoints = new HashSet<>();
		covertedNodes = new HashSet<>();
		for(indexNode child: node.getNodelist()) {//add all the children
			covertedNodes.add(child);
			addSum(child.getSum());	
		}
	//	System.out.println("total is"+sumTotal[0]);
		heap = new PriorityQueue<kmeansHeap>();//this is for storing the bounds, Harmly
	}
	
	public cluster(double[] centroid, int dimen) {
		dimension = dimen;
		finalCentroid = new double[dimension];
		sumTotal =  new double[dimension];
		coveredPoints = new HashSet<>();
		covertedNodes = new HashSet<>();
		for(int i = 0; i < dimen; i++) {
			finalCentroid[i] = centroid[i];
			sumTotal[i] = 0;
		}
		heap = new PriorityQueue<kmeansHeap>();//this is for storing the bounds, Harmly
	}
	
	public boolean getCenterChanged() {
		return centerChanged;
	}

	
	
	
	/*
	 * update by incremental gap, get the mean value.
	 */
	double extractNewCentroidByMeansIncremental() {
		int numberPoints = coveredPoints.size();
		for(indexNode aIndexNode: covertedNodes) {
			numberPoints += aIndexNode.getTotalCoveredPoints();//all the points in the cluster
		}
		double []newfinalCentroid = new double[dimension];
		for(int i=0; i<dimension; i++) {
			newfinalCentroid[i] = 0;
		}
		if(numberPoints != 0) {
			for (int i = 0; i < dimension; i++) {
				newfinalCentroid[i] = sumTotal[i] / numberPoints;// no need to access the data
			}
			double drift = Util.EuclideanDis(newfinalCentroid, finalCentroid, dimension);
			finalCentroid = newfinalCentroid;
			for (int i = 0; i < dimension; i++) {
				finalCentroid[i] = newfinalCentroid[i];
			}
			return drift;
		} else
			return 0;
	}


	// get the centroid of all points in this cluster
	double[] extractMeans() {
		int numberPoints = coveredPoints.size();
		for(indexNode aIndexNode: covertedNodes) {
			numberPoints += aIndexNode.getTotalCoveredPoints();//all the points in the cluster
		}
		double []newfinalCentroid = new double[dimension];
		Arrays.fill(newfinalCentroid, 0);

		if(numberPoints != 0) {
			for (int i = 0; i < dimension; i++) {
				newfinalCentroid[i] = sumTotal[i] / numberPoints;// no need to access the data
			}
			return newfinalCentroid;
		} else
			return newfinalCentroid;
	}

	double medoid_extractNewCentroidByMeansIncremental(double[] medoidcenter) {
		int numberPoints = coveredPoints.size();
		for(indexNode aIndexNode: covertedNodes) {
			numberPoints += aIndexNode.getTotalCoveredPoints();//all the points in the cluster
		}
		if(numberPoints != 0) {
			double drift = Util.EuclideanDis(medoidcenter, finalCentroid, dimension);
			finalCentroid = medoidcenter;
			return drift;
		} else
			return 0;
	}

	/**
	 * update the centroid to new one, and calculate the distance as drift
	 * @param medoidcenter the coordinate of new centroid (it's the nearest point from the centroid we calculated before)
	 * @return drift: distance from the centroid in previous iteration to new centroid
	 */
	double extractNewCentroidByMeansIncremental1(double[] medoidcenter) {
		int numberPoints = coveredPoints.size();
		for(indexNode aIndexNode: covertedNodes) {
			numberPoints += aIndexNode.getTotalCoveredPoints();	// all the points in the cluster
		}
		if(numberPoints != 0) {
			double drift = Util.EuclideanDis(medoidcenter, finalCentroid, dimension);
			finalCentroid = medoidcenter;
			for (int i = 0; i < dimension; i++) {
				finalCentroid[i] = medoidcenter[i];
			}
			return drift;
		} else
			return 0;
	}


	void getCenterMeans(){

	}
	
	void minusSum(double []point) {
		for(int i=0; i<dimension; i++) {
			sumTotal[i] -= point[i];
		}
	}
	
	void addSum(double []point) {
		for(int i=0; i<dimension; i++) {
			sumTotal[i] += point[i];
		}
	}
	
	/*
	 * scan every trajectory to get the maximum distance as the radius.
	 */
	double getRadius(Set<Integer> candidates, double[][] datamapEuc, double[] nodeSum) {
		double max = 0;
		for(int idx: candidates) {
			double[] data = datamapEuc[idx-1];
			double dis = Util.EuclideanDis(finalCentroid, data, dimension);
			if(dis > max) {
				max = dis;
			}
			for(int i=0; i<dimension; i++) {
				nodeSum[i] += data[i];
			}
		}
		return max;
	}
	
	/*
	 * calculate the radius of each cluster in pami20 paper, add the time
	 */
	double pami20_calculate_radius(double[][] datamapEuc, boolean []pami20_flags) {
		double max = 0;
		for(int idx: coveredPoints) {
			double[] data = datamapEuc[idx-1];
			double dis = Util.EuclideanDis(finalCentroid, data, dimension);
			if(dis > max) {
				max = dis;
			}
		}
		return max;
	}
	
	/*
	 * scan every trajectory to get the maximum distance as the radius.
	 */
	double getSum(Set<Integer> candidates, Map<Integer, double[]> datamapEuc) {
		double sum = 0;
		for(int idx: candidates) {
			double[] data = datamapEuc.get(idx);
			double dis = Util.EuclideanDis(finalCentroid, data, dimension);
			sum += dis;	
		}
		return sum;
	}
	
	/*
	 * scan every trajectory to get the maximum distance as the radius.
	 */
	double[] getSumTotal() {
		return sumTotal;
	}
	
	void removeNode(indexNode node) {//remove specific node from index
		if(covertedNodes.contains(node)) {
			covertedNodes.remove(node);
			minusSum(node.getSum());
		}
	}
	
	void addNode(indexNode node) {//remove specific node from index
		if(!covertedNodes.contains(node)) {// this node never exists
			covertedNodes.add(node);
			addSum(node.getSum());
		}
	}
	
	public int computeNumberofNode() {
		return covertedNodes.size();
	}
	
	/*
	 * compute the sum of the cluster, this is for debug purpose.
	 */
	public double computeSum(double[][] datamapEuc, Set<Integer> allPointsAll) {
	//	System.out.println(Arrays.toString(finalCentroid));
		Set<Integer> allPoints = new HashSet<>(coveredPoints);
		Queue<indexNode> nodeq = new LinkedList<>(covertedNodes);
		while(!nodeq.isEmpty()) {
			indexNode aIndexNode = nodeq.poll();
			if(aIndexNode.getNodelist().isEmpty()) {
				allPoints.addAll(aIndexNode.getpointIdList());
			}else {
				nodeq.addAll(aIndexNode.getNodelist());
			}
		}
		allPointsAll.addAll(allPoints);
		double sum = 0;
		for(int pointid: allPoints) {
			if(finalCentroid!=null)
				sum += Util.EuclideanDis(datamapEuc[pointid-1], finalCentroid, dimension);
		}
		return sum;
	}
	
	/*
	 * compute the sum of the cluster, this is for debug purpose.
	 */
	public double computeFairSum(double[][] datamapEuc, Set<Integer> allPointsAll, int userID[], Map<Integer, Double> userNumber) {
	//	System.out.println(Arrays.toString(finalCentroid));
		Set<Integer> allPoints = new HashSet<>(coveredPoints);
		Queue<indexNode> nodeq = new LinkedList<>(covertedNodes);
		while(!nodeq.isEmpty()) {
			indexNode aIndexNode = nodeq.poll();
			if(aIndexNode.getNodelist().isEmpty()) {
				allPoints.addAll(aIndexNode.getpointIdList());
			}else {
				nodeq.addAll(aIndexNode.getNodelist());
			}
		}
		allPointsAll.addAll(allPoints);
		double sum = 0;
		for(int pointid: allPoints) {
			if(finalCentroid!=null) {
			//	System.out.println(userID[pointid-1]);
				sum += Util.EuclideanDis(datamapEuc[pointid-1], finalCentroid, dimension)*1/userNumber.get(userID[pointid-1]);
			}
		}
		return sum;
	}
	
	public void reset(indexNode node) {
		coveredPoints = new HashSet<>();
		covertedNodes = new HashSet<>();
		sumTotal = new double[dimension];
		if(node!=null) {
		//	covertedNodes.add(node);// add the node into candidates
			//the sum should be set
			for(indexNode child: node.getNodelist()) {//add all the children
				covertedNodes.add(child);
				addSum(child.getSum());
			}
		}
		sumdistance = 0;
	}
	
	/*
	 * update by scanning every point and node
	 */
	double extractNewCentroidByMeans(double[][] datamapDouble) {
		int numberPoints = coveredPoints.size();		
		double []newfinalCentroid = new double[dimension];
		for(int i=0; i<dimension; i++) {
			newfinalCentroid[i] = 0;
		}
		for(int traidx: coveredPoints) {
			double []point = datamapDouble[traidx-1];
			for(int i=0; i<dimension; i++) {
				newfinalCentroid[i] += point[i];
			}
		}		
		for(indexNode nodes: covertedNodes) {
			numberPoints += nodes.getTotalCoveredPoints();
			double []sum = nodes.getSum();
			for(int i=0; i<dimension; i++) {
				newfinalCentroid[i] += sum[i];
			}
		}
	//	System.out.print(numberPoints+",");
		if(numberPoints!=0) {
			for (int i = 0; i < dimension; i++) {
				newfinalCentroid[i] /= numberPoints;//compute the new
			}
			double drift = Util.EuclideanDis(newfinalCentroid, finalCentroid, dimension);
			finalCentroid = newfinalCentroid;
			for (int i = 0; i < dimension; i++) {
				finalCentroid[i] = newfinalCentroid[i];
			}
			return drift;
		} else
			return 0;
	}
	
	/*
	 * we define a new form of kmeans which considers the proportion of each person, mainly for location-based service.
	 * 
	 * the refinement is more tricky as it needs to access the owner information
	 */
	double fairkMeansRefinement(int userID[], Map<Integer, Double> userNumber, double[][] datamapDouble) {
		double numberPoints = 0;		
		double []newfinalCentroid = new double[dimension];
		for(int i=0; i<dimension; i++) {
			newfinalCentroid[i] = 0;
		}
		for(int traidx: coveredPoints) {
			double []point = datamapDouble[traidx-1];
			numberPoints += 1.0/userNumber.get(userID[traidx-1]);
			for(int i=0; i<dimension; i++) {
				newfinalCentroid[i] += point[i]/userNumber.get(userID[traidx-1]);// the update needs to consider the 1/|o|, where o is the owner
			}
		}		
		for(indexNode nodes: covertedNodes) {
			numberPoints += nodes.getTotalCoveredPointsFair();// the number also needs to be normalized
			double []sum = nodes.getSum();
			for(int i=0; i<dimension; i++) {
				newfinalCentroid[i] += sum[i];
			}
		}
		if(numberPoints!=0) {
			for (int i = 0; i < dimension; i++) {
				newfinalCentroid[i] /= numberPoints;//compute the new centroid, the number of points needs to be normalized
			}
			double drift = Util.EuclideanDis(newfinalCentroid, finalCentroid, dimension);
			finalCentroid = newfinalCentroid;
			for (int i = 0; i < dimension; i++) {
				finalCentroid[i] = newfinalCentroid[i];
			}
			return drift;
		} else
			return 0;
	}
	
	
	/*
	 * Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms
	 */
	double PAM() {
		
		// we can call the function to test, https://github.com/elki-project/elki/tree/master/elki-clustering/src/main/java/elki/clustering/kmedoids
		return 0;
	}
	
	/*
	 * 
	 */
	double FastPAM() {
		return 0;
	}
	
	/*
	 * NIPS 2020, this is an approximate algorithm, so we will not compare, we can 
	 */
	double BanditPam() {
		return 0;
	}
	
	/*
	 * we implement the PAM, FastPAM, and BanditPam here.
	 */
	double kMedoidsRefinement() {
		
		
		// compute the mean using existing fast refinement.
		
		// search the nearest neighbor as the existing centroid, based on the index,
		
		// this will be linear complexity, but can we prove it?
		
		// theory applied here, we do not need to build based on PAM as it is still minor changes 
		
		// we add a verification step for this, by ranking all the points
		
		
		return 0;
	}
	
	// we can also use histograms to prune grids that has a large 
}
