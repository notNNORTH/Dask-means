package edu.nyu.unik.expriments;

import au.edu.rmit.trajectory.clustering.kmeans.kmeansAlgorithm;
import edu.wlu.cs.levy.cg.KeyDuplicateException;
import edu.wlu.cs.levy.cg.KeySizeException;

import java.io.IOException;

public class knnExperiment {

    public 	knnExperiment() {
        // TODO Auto-generated constructor stub
    }

    // 1 5 10 50 100
    public static void main(String[] args) throws IOException, KeySizeException, KeyDuplicateException {
        // TODO Auto-generated method stub
        int[] kvalue = new int[]{10, 50, 100, 200, 400, 600, 800, 1000};//10, 100, 1000
        int[] scales = new int[] {100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000};
        int[] capacities = new int[] {10, 20, 30, 40, 50, 60};//capacity
        int[] dimensions = new int[] {2, 10, 20, 30, 40, 50};


        kvalue = new int[] {1, 2, 5, 10, 100, 1000, 10000};
        kvalue = new int[] {1};



        String[] paras = new String[7];
        paras[0] = "./dataset/1_2d_T_drive_No_duplication.txt";
        //paras[0] = "D:\\SURE\\embedding\\csvdata\\combined_data_16.csv";
        paras[1] = "10";
        paras[2] = "1000000";
        paras[3] = "a";
        paras[4] = "6";
        paras[5] = "0";
        paras[6] = "1";
        kmeansAlgorithm<?> runkmeans = new kmeansAlgorithm<>(paras);
        runkmeans.knn_experiments(kvalue);
    }
}
