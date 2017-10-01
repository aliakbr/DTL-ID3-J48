import IF4071.DecisionTreeLearning.MyID3.MyID3;
import IF4071.DecisionTreeLearning.Util.*;
import IF4071.DecisionTreeLearning.MyC45.*;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class main {
    public static void main(String[] args) throws Exception {
        String filename = "./data/iris.arff";

        Util util = new Util();
        Instances data = util.ReadArff(filename);
        System.out.println("=====    Data    =====");
        System.out.println(data.toString());
        System.out.println("======================");
        System.out.println();

        data = util.Resample(data);
        System.out.println("===== Resampling =====");
        System.out.println(data.toString());
        System.out.println("======================");
        System.out.println();

        data = util.Randomize(data);
        System.out.println("===== Randomize  =====");
        System.out.println(data.toString());
        System.out.println("======================");
        System.out.println();

        Classifier id3 = new MyC45(true);
        id3 = util.SplitTest(id3, data, 80);
        util.saveModel("./models/myc45.v1.model", id3);

        Classifier loadid3;
        System.out.println("===== Load Model =====");
        loadid3 = util.loadModel("./models/myc45.v1.model");
    }
}
