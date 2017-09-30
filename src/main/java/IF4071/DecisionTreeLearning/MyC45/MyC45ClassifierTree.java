package IF4071.DecisionTreeLearning.MyC45;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

public class MyC45ClassifierTree{
    private Attribute splitAttribute;
    private MyC45ClassifierTree children[];
    private double classDistribution[];
    private Instances data;

    // Treshold for numeric value splitting
    private double treshold;

    public MyC45ClassifierTree(){
        setChildren(null);
        setClassDistribution(null);
        setSplitAttribute(null);
    }

    public void buildClassifier(Instances instances) throws Exception {
        // Handling missing value for numeric and nominal value

    }

    public double classifyInstance(Instance instance) throws Exception {
        return 0;
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
    }

    // Getter - Setter auto generated
    public double getTreshold() {
        return treshold;
    }

    public void setTreshold(double treshold) {
        this.treshold = treshold;
    }

    public Attribute getSplitAttribute() {
        return splitAttribute;
    }

    public void setSplitAttribute(Attribute splitAttribute) {
        this.splitAttribute = splitAttribute;
    }

    public MyC45ClassifierTree[] getChildren() {
        return children;
    }

    public void setChildren(MyC45ClassifierTree[] children) {
        this.children = children;
    }

    public double[] getClassDistribution() {
        return classDistribution;
    }

    public void setClassDistribution(double[] classDistribution) {
        this.classDistribution = classDistribution;
    }

    public Instances getDataset() {
        return data;
    }

    public void setDataset(Instances dataset) {
        this.data = dataset;
    }
}
