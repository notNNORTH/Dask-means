package edu.nyu.unik.expriments;

import au.edu.rmit.AutoIS;
import au.edu.rmit.HyperPoint;
import au.edu.rmit.trajectory.clustering.kmeans.kmeansAlgorithm;
import edu.wlu.cs.levy.cg.KeyDuplicateException;
import edu.wlu.cs.levy.cg.KeySizeException;


import java.io.IOException;
import java.util.LinkedList;

import static au.edu.rmit.trajectory.clustering.kmeans.kmeansAlgorithm.Max_Min_value;
import static au.edu.rmit.trajectory.clustering.kmeans.kmeansAlgorithm.ReadData;

public class kmeans_Smeans {
    public static void main(String[] args) throws IOException, KeySizeException, KeyDuplicateException {
        // TODO Auto-generated method stub
        int[] kvalue = new int[]{1000};     // 10, 100, 1000    the num of cluster centroid
        int testTime = 1;                   // test one time
        String[] paras = new String[7];
        paras[0] = "./dataset/US_twitter_average.txt";      // with data scale of 200,000
        //Convergence
        //Chicage
        //dataset/NYC_pickup_clean.csv
        //dataset/US_clustering_clean.txt
        //

        //Not convergence
        //assment
        //euro
        //dataset/US_twitter_average.txt

        paras[1] = "1000";
        paras[2] = "100000";
        paras[3] = "a";
        paras[4] = "assment";
        paras[5] = "0";
        paras[6] = "1";
        kmeansAlgorithm<?> runkmeans = new kmeansAlgorithm<>(paras);
        runkmeans.S_Means_experiments(kvalue, testTime);        // run the configuration that specified
        //runkmeans.KNN_search_test_wang();
        //String address = "dataset/Chicago_pickup_20m_clean1.csv";
        //HyperPoint[] sample = ReadData(address,100000);
        //LinkedList<double[]> list = Max_Min_value(sample);
        //int K = list.get(0).length;
        //HyperPoint min = new HyperPoint(list.get(0));
        //HyperPoint max = new HyperPoint(list.get(1));
        //AutoIS kd = new AutoIS(K, min, max);
        //System.out.println("kd begin"+"\n");
        //kd.ConstructionAutoIS(sample);
        //System.out.println("kd end"+"\n");

    }
}
