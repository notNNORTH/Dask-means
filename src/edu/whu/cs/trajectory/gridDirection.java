package edu.whu.cs.trajectory;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import edu.nyu.dss.similarity.EffectivenessStudy;

public class gridDirection {
	
	double minx;
	double miny;
	double range;
	int resolution;
	double unit = range/Math.pow(2, resolution);
	
	
	// grid with direction, for paper extension,
	
	// or quantization, PPQ-trajectory: spatio-temporal quantization for querying in large trajectory repositories
	
	// this is for general trajectories
	void quantizationkPaths(double [][]rawTrajectory) {
		
		// put this into a hashmap
		
		// we use z-curve, and the last two bits as the direction or more
		
		// compression of z-curve code
		
		
		Set<Integer> trajectoryIntegers = new HashSet<>();
		for(int i=0; i<rawTrajectory.length; i++) {
			int x = (int)((rawTrajectory[i][0]-minx)/unit);
			int y = (int)((rawTrajectory[i][1]-miny)/unit);
			int zcode = EffectivenessStudy.combine(x,y,resolution)*4;
			int direction = 0; // 16 possibilities, 4 bits to store, 
			zcode += direction; 
			
			// we need to add the nodes that passing through without jumping
			trajectoryIntegers.add(zcode);
		}
		
		// putting weight on the grids to show the direction, speed and volume
	}
	
	/*
	 * maping the data, encoding based on frequency, for the code that never shows
	 * which will save more space in terms of storage after using delta encoding
	 */
	void data_distribution() {
		
	}
	
	// also related to trajectory smoothing
	// Data-Driven Trajectory Smoothing
	void quantizationTorch() {// time stamp distribution, overlapped 
		
		// z-curve, choose the smooth grid set to denote each trajectories,
		
		// a starting timestamp and an ending timestamp
		
		// temporal index for filtering non-overlapped trajectories, for the applications
	}
	
	void OptimizedLORS(){
		// it has longer length than the points, we should think about the compression and encoding
		
		// whether use the Manhattan distance to move
		
		// 
		
		// eg. we find those common sub-trajectories shared by most trajectories, then reduce the number of cell ids.
		
		// we can call the LCS directly to conduct the search, based on the dynamic programming
		
		// still use our inverted index to search, 
		
	}
	
	/*
	 * mapping the trajectories to the POIs and conduct the intersection
	 * 
	 * A new way of mapping, and staying point detection
	 * 
	 * This is for POI matching, 
	 */
	void POIMatching(double [][]Pois, HashMap<Integer, double[][]> trajectories) {
		// denote each raw trajectory into POIs, then map matching to POIs, then we propose a novel measure 
		
		// when no pois, grids first, then clustering grids, conduct merge and connect, to construct a network in a more efficient way,
		
		// how to solve the sparse sampling problem, shortest path search
		
		// detect POIs, and conduct the mapping to nearest POIs, 
		
		// or construct the map in a coarse way, then fast map matching
		
		// we compute the similarity in an efficient way, given person's points
		
	}
	
	
	/*
	 * k-paths clustering, but based on grid-based trajectories, with the volume information
	 */
	void clustering() {
		
	}
	
	/*
	 * search the trajectories with new similarity measure
	 */
	void searchTorch() {
	
		
	}
	
	
	void fastkmeans() {
		// better model for algorithm selection
		
		// more algorithms, pick-means
		
		// Structured Inverted-File k-Means Clustering for High-Dimensional Sparse Data
		
		// // fair k-means evaluation, multiple algorithms, possible

	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
