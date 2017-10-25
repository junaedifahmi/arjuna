package Inti;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.rules.FURIA;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
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

		replaceMiss(withMiss);

		// build model

		mainModel();

		// evaluation

		evaluation();

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
		for (Instance datum : data) {
			if (datum.hasMissingValue()) {
				withMiss.add(datum);
				data.remove(datum);
			}
		}
	}

	private static void replaceMiss(Instances MissData) throws Exception {
		for (Instance datum : MissData) {
			for (int i = 0; i < datum.numAttributes(); i++) {
				Attribute att = datum.attribute(i);
				if (datum.isMissing(att)) {
					ganti(datum, i);
				}
			}
		}

	}

	private static double ganti(Instance datum, int i) throws Exception {
		// TODO Auto-generated method stub
		Attribute att = datum.attribute(i);
		if (att.isNominal()) {
			model = new FURIA();
		} else {
			model = new Logistic();
		}

		data.setClassIndex(i);
		model.buildClassifier(data);
		return model.classifyInstance(datum);

	}

	private static void mainModel() throws Exception {
		data.setClassIndex(data.numAttributes() - 1);
		model = new J48();
		// ((J48) model).setUnpruned(false);
		model.buildClassifier(data);
		System.out.println(data);
	}

	private static void evaluation() throws Exception {
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(model, data, 10, new Random(0));
		System.out.println(eval.toSummaryString());
	}

}
