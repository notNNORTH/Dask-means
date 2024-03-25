package au.edu.rmit.trajectory.clustering.kmeans;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class ImprovedPAM {

    public static void main(String[] args) {
        // Example usage
        List<double[]> dataPoints = new ArrayList<>();
        dataPoints.add(new double[]{1, 2});
        dataPoints.add(new double[]{3, 4});
        dataPoints.add(new double[]{5, 6});
        dataPoints.add(new double[]{7, 8});

        List<double[]> medoids = pamClustering(dataPoints, 2);
        System.out.println("Medoids: " + medoids);
    }

    public static List<double[]> pamClustering(List<double[]> dataPoints, int k) {
        List<double[]> medoids = new ArrayList<>();
        // part one: Randomly initialize medoids
        Random rand = new Random();
        for (int i = 0; i < k; i++) {
            int randomIndex = rand.nextInt(dataPoints.size());
            medoids.add(dataPoints.get(randomIndex));
        }

        boolean changed;
        do {
            changed = false;
            //????????cluster?
            // Assign each data point to the nearest medoid
            List<List<double[]>> clusters = new ArrayList<>();
            for (int i = 0; i < k; i++) {
                clusters.add(new ArrayList<>());
            }

            for (double[] point : dataPoints) {
                int nearestMedoidIndex = getNearestMedoidIndex(point, medoids);
                clusters.get(nearestMedoidIndex).add(point);
            }

            // Update medoids with the data point that minimizes the total distance within the cluster
            for (int i = 0; i < k; i++) {
                double[] currentMedoid = medoids.get(i);
                double currentMedoidCost = calculateTotalDistance(currentMedoid, clusters.get(i));

                for (double[] dataPoint : clusters.get(i)) {
                    double[] tempMedoid = dataPoint;
                    double tempMedoidCost = calculateTotalDistance(tempMedoid, clusters.get(i));

                    if (tempMedoidCost < currentMedoidCost) {
                        medoids.set(i, tempMedoid);
                        currentMedoidCost = tempMedoidCost;
                        changed = true;
                    }
                }
            }
        } while (changed);

        return medoids;
    }

    public static int getNearestMedoidIndex(double[] point, List<double[]> medoids) {
        int nearestMedoidIndex = 0;
        double minDistance = Double.MAX_VALUE;

        for (int i = 0; i < medoids.size(); i++) {
            double distance = calculateDistance(point, medoids.get(i));
            if (distance < minDistance) {
                minDistance = distance;
                nearestMedoidIndex = i;
            }
        }

        return nearestMedoidIndex;
    }

    public static double calculateDistance(double[] point1, double[] point2) {
        double sum = 0;
        for (int i = 0; i < point1.length; i++) {
            sum += Math.pow(point1[i] - point2[i], 2);
        }
        return Math.sqrt(sum);
    }
    public static double calculateTotalDistance(double[] medoid, List<double[]> dataPoints) {
        double totalDistance = 0;
        for (double[] point : dataPoints) {
            totalDistance += calculateDistance(medoid, point);
        }
        return totalDistance;
    }
}

