package IF4071.DecisionTreeLearning.Util;

import weka.core.Instances;

import java.io.*;
import java.util.Scanner;

import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import java.util.Random;

import IF4071.DecisionTreeLearning.MyC45.MyC45;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.Randomize;

public class Util {
    private Scanner input;

    public Instances ReadArff(String filename) throws Exception {
        BufferedReader reader = new BufferedReader(
                new FileReader(filename));
        Instances data = new Instances(reader);
        data.setClassIndex(data.numAttributes() - 1);
        reader.close();

        return data;
    }

    public Instances ReadCSV(String filename) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(filename));
        Instances data = loader.getDataSet();

        return data;
    }

    public Instances Resample(Instances data) throws Exception {
        Resample filter = new Resample();
        Instances newData;

        filter.setInputFormat(data);
        newData = Filter.useFilter(data, filter);

        return newData;
    }

    public Instances Randomize(Instances data) throws Exception {
        Randomize filter = new Randomize();
        Instances newData;

        filter.setInputFormat(data);
        newData = Filter.useFilter(data, filter);

        return newData;
    }

    public Instances Remove(Instances data, String attr) throws Exception {
        String[] options = new String[2];
        options[0] = "-R";  // "range"
        options[1] = attr;
        Remove filter = new Remove();
        filter.setOptions(options);
        Instances newData;

        filter.setInputFormat(data);
        newData = Filter.useFilter(data, filter);

        return newData;
    }

    public Classifier TenFoldsCrossValidation(Instances data, int idxClass) throws Exception{
        Classifier dtl = new MyC45();
        //((MyC45) dtl).setKelas(idxClass);
        dtl.buildClassifier(data);

        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(dtl, data, 10, new Random(1));

        System.out.println();
        System.out.println("=== Summary ===");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());

        return dtl;
    }

    public Classifier SplitTest(Instances data, int percent, int idxClass) throws Exception {
        int trainSize = (int) Math.round(data.numInstances() * percent / 100);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);

        Classifier dtl = new MyC45();
        //((MyC45) dtl).setKelas(idxClass);
        dtl.buildClassifier(train);
        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(dtl, test);

        System.out.println();
        System.out.println("=== Summary ===");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());

        return dtl;
    }

    public Classifier TrainingTest(Instances train, Instances test, int idxClass) throws Exception {
        Classifier dtl = new MyC45();
        //((MyC45) dtl).setKelas(idxClass);
        dtl.buildClassifier(train);
        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(dtl, test);

        System.out.println();
        System.out.println("=== Summary ===");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());

        return dtl;
    }

    public Classifier FullTrainingSchema(Instances data, int idxClass) throws Exception{
        Classifier dtl = new MyC45();
        //((MyC45) dtl).setKelas(idxClass);
        dtl.buildClassifier(data);

        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(dtl, data);

        System.out.println();
        System.out.println("=== Summary ===");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());

        return dtl;
    }

    public void saveModel(String filename, Classifier cls) throws Exception {
        ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream(filename));
        output.writeObject(cls);
        output.flush();
        output.close();
    }

    public Classifier loadModel(String filename) throws Exception{
        ObjectInputStream fileinput = new ObjectInputStream(new FileInputStream(filename));
        Classifier cls = (Classifier) fileinput.readObject();
        fileinput.close();
        return cls;
    }

    public void classifyData(Classifier model, Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(model, data);

        System.out.println();
        System.out.println("=== Summary ===");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
    }


}
