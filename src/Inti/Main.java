package Inti;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class Main {
	// raw data
	static Instances data = null;

	// data with missing values

	static Instances withMiss = null;

	// the main model

	static Classifier model = null;

	public static void main(String... Args) throws Exception {
		// load file
		String filedir = "/home/juunnn/Project/Data Mining/korup.csv";
		loadData(filedir);
		// System.out.println(data.toSummaryString());

		// split data

		splitData(data);

		System.out.println(data.toSummaryString());

		// replace missing

		replaceMiss();

		// build model

		mainModel();

		// evaluation

		evaluation(data);

	}

	private static void loadData(String fileDir) throws IOException {
		File file = new File(fileDir);
		if (file.getName().toLowerCase().endsWith(CSVLoader.FILE_EXTENSION)) {
			CSVLoader loader = new CSVLoader();
			loader.setSource(file);
			data = loader.getDataSet();
		} else {
			data = new Instances(new BufferedReader(new FileReader(file)));
		}
	}

	private static void splitData(Instances data) throws Exception {
		// TODO Auto-generated method stub
		withMiss = new Instances(data, data.numInstances());
		for (int i = 0; i < data.numInstances(); i++) {
			if (data.instance(i).hasMissingValue()) {
				withMiss.add(data.instance(i));
				if (data.remove(i) == null) {
					System.out.println("Instances no " + i + " cannot be removed");
				}
			}
		}
		System.out.println(data.instance(0));
		System.out.println(data.instance(74));

		System.out.println("##############");
		// System.out.println(withMiss);

	}

	private static void replaceMiss() {
		// TODO Auto-generated method stub

	}

	private static void mainModel() throws Exception {
		// TODO Auto-generated method stub
		data.setClassIndex(data.numAttributes() - 1);
		model = new J48();
		model.buildClassifier(data);
		System.out.println("###############################################");
		System.out.println(model);
		System.out.println("###############################################");

	}

	private static void evaluation(Instances data) throws Exception {
		// TODO Auto-generated method stub
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(model, data, 10, new Random(0));
		System.out.println(eval.toSummaryString());
	}

}
