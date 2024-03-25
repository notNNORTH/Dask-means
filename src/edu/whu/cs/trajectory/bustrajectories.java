package edu.whu.cs.trajectory;

public class bustrajectories {
	
	/*
	 * we get the route of the NYC and WHU, later we integrate to the search engine, for various data, we will publish our data
	 */
	
	/*
	 * pre-process the dataset to get the bus network, and the trajectory data in a double array
	 * and the bus trajectories data
	 */
	void preprocessing() {
		// serise the data to the hashmap, including the road network
	}
	
	/*
	 * compute the temporal similarity using the mapping to the nearest edge.
	 */
	void temporal_distance(double[]xa, double[]ya, double[]ta, double[]xb, double[]yb, double[]tb) {
		// check they have overlaps in two dimensions, if no return infinity
		
		// map the trajectories xb to yb
		
		//
	}
	
	/*
	 * find the top-k routes with the maximum trajectory simialrity
	 */
	void topksearch() {
		// give a set of bus routes with time table, measuring the similarity
		
		// 
	}
	
	/*
	 * we generate the time table first, then generate the top-k similarity, as an optimization
	 */
	void timeTableGeneration() {
		
	}
	
	/*
	 * routing based on trajectory search given the source and destination
	 */
	void busRouting() {
		
	}
	
	/*
	 *
	 */
	void dynamicRouting() {
		
	}
	
	void main() {
		
	}
}
