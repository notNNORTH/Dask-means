package edu.nyu.unik.expriments;

import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import au.edu.rmit.trajectory.clustering.kmeans.kmeansAlgorithm;
import edu.wlu.cs.levy.cg.KeyDuplicateException;
import edu.wlu.cs.levy.cg.KeySizeException;

public class kmeansEfficiency {

	public kmeansEfficiency() {
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) throws IOException, KeySizeException, KeyDuplicateException {
		// TODO Auto-generated method stub
		int[] kvalue = new int[]{10, 50, 100, 200, 400, 600, 800, 1000};//10, 100, 1000
		int[] scales = new int[] {100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000}; //1000, 5000, 10000, 50000, 100000, 500000, 1000000
		int[] capacities = new int[] {10, 20, 30, 40, 50, 60};//capacity
		int[] dimensions = new int[] {2, 10, 20, 30, 40, 50};

		// dimension kvalue scale capacities
		// runInBatch();
		// getInternalNodes();

		kvalue = new int[] {10, 1000, 10000};	//10, 100, 1000
		dimensions = new int[] {3};
		scales = new int[] {10000};
		capacities = new int[] {30};
		int testTime = 1;//test one time

		String[] paras = new String[7];
		paras[0] = "./dataset/2_2d_Porto_1277582.txt";	// data path
		paras[1] = "1000";			// k value
		paras[2] = "1000000";		// data scale
		paras[3] = "a";
		paras[4] = "Porto";	// output file name
		paras[5] = "0";				// first dimension
		paras[6] = "1";				// last dimension
		kmeansAlgorithm<?> runkmeans = new kmeansAlgorithm<>(paras);
		runkmeans.experiments(kvalue, testTime);// run the configuration that specified


	}

	public static void getInternalNodes() throws IOException, KeyDuplicateException, KeySizeException {
		String dataPath = "E:\\torch-clus\\dataset\\icde-dataset\\";
		List<String> csvFiles = getCSVFiles(dataPath);
		int counter = 0;
		for (String filePath : csvFiles) {
			if (counter > 0) {
				break;
			}

			String outputFileName = filePath.split("_")[1];
			outputFileName = outputFileName.split("\\.")[0];
			outputFileName = outputFileName.substring(0, outputFileName.length() - 1) + ".csv";

			String[] paras = new String[7];
			paras[0] = filePath;
			// System.out.println(paras[0]);
			paras[1] = getDimension(filePath);
			// System.out.println(paras[1]);
			paras[2] = getDataScale(filePath);
			// System.out.println(paras[2]);
			paras[3] = "a";
			paras[4] = getLogName(filePath);
			paras[5] = "0";
			paras[6] = Integer.toString(Integer.parseInt(paras[1]) - 1);
			kmeansAlgorithm<?> runkmeans = new kmeansAlgorithm<>(paras);


			runkmeans.getInternalNodes(2, outputFileName);

			counter += 1;
		}

	}

	public static void runInBatch() throws IOException, KeySizeException, KeyDuplicateException {
		String dataPath = "E:\\torch-clus\\dataset\\icde-dataset\\";
		List<String> csvFiles = getCSVFiles(dataPath);
		int counter = 0;
		for (String filePath : csvFiles) {
			// if (counter > 5) {
			// 	break;
			// }
			// System.out.println(filePath);

			String[] paras = new String[7];
			paras[0] = filePath;
			System.out.println(paras[0]);
			paras[1] = getDimension(filePath);
			// System.out.println(paras[1]);
			paras[2] = getDataScale(filePath);
			// System.out.println(paras[2]);
			paras[3] = "a";
			paras[4] = getLogName(filePath);
			paras[5] = "0";
			paras[6] = Integer.toString(Integer.parseInt(paras[1]) - 1);
			kmeansAlgorithm<?> runkmeans = new kmeansAlgorithm<>(paras);

			int testTime = 1;
			Random random = new Random();
			int randomK = random.nextInt(9901) + 100;
			// int randomK = 10;
			int[] kvalue = new int[] {randomK};	//10, 100, 1000
			runkmeans.experiments(kvalue, testTime);// run the configuration that specified

			// counter += 1;
		}
	}

	public static List<String> getCSVFiles(String directoryPath) throws IOException {
		List<String> csvFiles = new ArrayList<>();
		Path path = Paths.get(directoryPath);

		Files.walkFileTree(path, new SimpleFileVisitor<Path>() {
			@Override
			public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
				if (file.toString().endsWith(".csv")) {
					csvFiles.add(file.toString());
				}
				return FileVisitResult.CONTINUE;
			}
		});

		return csvFiles;
	}

	// input file path, then return the dimension of that file
	public static String getDimension(String filePath) {
		if (filePath.contains("Shapenet")) {
			return "3";
		} else {
			return "2";
		}
	}

	// input file path, then return date scale of that file
	public static String getDataScale(String filePath) {
		int underscoreIndex = filePath.lastIndexOf('_');

		if (underscoreIndex != -1) {
			return filePath.substring(underscoreIndex + 1, filePath.length() - 5);
		} else {
			return "No match found";
		}
	}

	// input file path, then return the log name of that file
	public static String getLogName(String filePath) {
		if (filePath.contains("Shapenet")) {
			return "Shapenet";
		} else {
			return "map";
		}
	}
}