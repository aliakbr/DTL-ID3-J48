package IF4071.DecisionTreeLearning.MyC45;

import weka.classifiers.Classifier;
import weka.core.*;
import IF4071.DecisionTreeLearning.Util.Calculator;
import java.util.*;

public class MyC45ClassifierTree{
    private Attribute splitAttribute;
    private MyC45ClassifierTree children[];
    private double classDistribution[];
    private Instances data;
    private Integer classIndex;

    // Treshold for numeric value splitting
    private double treshold;

    public MyC45ClassifierTree(){
        setChildren(null);
        setClassDistribution(null);
        setSplitAttribute(null);
    }

    public void replaceMissingValues(Instances data) {
        Enumeration instanceEnum = data.enumerateInstances();
        while (instanceEnum.hasMoreElements()){
            Instance instance = (Instance) instanceEnum.nextElement();
            if (instance.hasMissingValue()){
                Enumeration attrEnum = instance.enumerateAttributes();
                while (attrEnum.hasMoreElements()){
                    Attribute attr = (Attribute) attrEnum.nextElement();
                    if (instance.isMissing(attr)){
                         double mostCommonValue = getCommonValue(data, attr);
                        instance.setValue(attr, mostCommonValue);
                    }
                }
            }
        }
    }

    private double getCommonValue(Instances data, Attribute attr) {
        if (attr.isNumeric()){
            // Get Median
           List<Double> values = new ArrayList<Double>();
            Enumeration instanceEnum = data.enumerateInstances();
            while (instanceEnum.hasMoreElements()) {
                Instance instance = (Instance) instanceEnum.nextElement();
                values.add(instance.value(attr));
            }

            Collections.sort(values);
            int size = values.size();
            double median;
            if (size % 2 == 0){
                median = 0.5 * (values.get(size / 2) + values.get((size / 2) + 1));
            }
            else{
                median = values.get(size / 2);
            }
            return median;
        }
        else{
            double distribution[] = new double[attr.numValues()];
            Enumeration instanceEnum = data.enumerateInstances();
            while (instanceEnum.hasMoreElements()) {
                Instance instance = (Instance) instanceEnum.nextElement();
                distribution[(int) instance.value(attr)]++;
            }

            return Utils.maxIndex(distribution);
        }
    }

    public void buildClassifier(Instances instances) throws Exception {
        // Set dataset
        setData(instances);

        // Handling missing value for numeric and nominal value
        replaceMissingValues(data);

        // Build Tree
        buildTree(data);

        // Prune Tree
        prune(data);
    }

    public double classifyInstance(Instance instance) throws Exception {
        if (getClassIndex() == null && Utils.eq(getClassIndex(), -1)){
            double splitAttrIdx;
            if (splitAttribute.isNumeric()){
                if (instance.value(splitAttribute) >= treshold ){
                    splitAttrIdx = 1.0;
                }
                else{
                    splitAttrIdx = 0.0;
                }
            }
            else{
                splitAttrIdx = instance.value(splitAttribute);
            }
            return getChildren()[(int) splitAttrIdx].classifyInstance(instance);
        }
        else{
            return classIndex;
        }
    }

    private void buildTree(Instances data) {
        // Set dataset
        setData(data);

        int numAttr = data.numAttributes();
        double[] gainRatios = new double[numAttr];
        Enumeration enumeration = data.enumerateAttributes();

        if (data.numInstances() == 0){
            setClassIndex(-1);
        }
        else {
            // Hitung gain ratio tiap atribut
            while (enumeration.hasMoreElements()) {
                Attribute attribute = (Attribute) enumeration.nextElement();
                if (attribute.isNominal()) {
                    gainRatios[attribute.index()] = Calculator.calcGainRatio(data, attribute);
                } else if (attribute.isNumeric()) {
                    setTreshold(searchTreshold(data, attribute)); // Cari treshold
                    gainRatios[attribute.index()] = Calculator.calcNumericGainRatio(data, attribute, treshold);
                }
            }

            // Temukan Gain Ratio Tertinggi
            int largestGainIdx= Utils.maxIndex(gainRatios);
            double gainRatio;
            if (data.attribute(largestGainIdx).isNominal()) {
                gainRatio = Calculator.calcGainRatio(data, data.attribute(largestGainIdx));
            } else {
                gainRatio = Calculator.calcNumericGainRatio(data, data.attribute(largestGainIdx), treshold);
            }

            // Generate Pohon
            if (Utils.eq(0, gainRatio)) {
                setClassDistribution(new double[data.numClasses()]);
                Enumeration instanceEnum = data.enumerateInstances();
                while (instanceEnum.hasMoreElements()) {
                    Instance instance = (Instance) instanceEnum.nextElement();
                    getClassDistribution()[(int) instance.classValue()]++;
                }
                Utils.normalize(getClassDistribution());
                setClassIndex(Utils.maxIndex(getClassDistribution()));
            } else {
                int numChild;
                setSplitAttribute(data.attribute(largestGainIdx));
                if (splitAttribute.isNumeric()){
                    numChild = 2;
                }
                else{
                    numChild = splitAttribute.numValues();
                }

                // Masukkan instance berdasarkan split
                setChildren(new MyC45ClassifierTree[numChild]);
                Instances[] splittedInstances;
                if (data.attribute(largestGainIdx).isNumeric()){
                    splittedInstances = Calculator.splitAttributeNumVal(data, getSplitAttribute(), treshold);
                }
                else{
                    splittedInstances = Calculator.splitDataByAttr(data, getSplitAttribute());
                }

                // Proses anak
                for (int i = 0; i < numChild; i++){
                    getChildren()[i] = new MyC45ClassifierTree();
                    getChildren()[i].buildTree(splittedInstances[i]);
                }

                for (int i = 0; i < numChild; ++i) {
                    MyC45ClassifierTree children = getChildren()[i];
                    if ((children.getClassIndex() != null) && Utils.eq(children.getClassIndex(), -1)){
                        double[] _classDistribution = new double[this.data.numClasses()];
                        Enumeration instanceEnum = this.data.enumerateInstances();
                        while (instanceEnum.hasMoreElements()) {
                            Instance instance = (Instance) instanceEnum.nextElement();
                            _classDistribution[(int) instance.classValue()]++;
                        }
                        Utils.normalize(_classDistribution);
                        int _decisonIndex = Utils.maxIndex(_classDistribution);

                        getChildren()[i].setClassIndex(_decisonIndex);
                        getChildren()[i].setClassDistribution(_classDistribution);

                    }
                }
            }
        }
    }

    private double searchTreshold(Instances data, Attribute attribute) {
        double[] treshold = new double[data.numInstances()];
        double[] gainRatio = new double[data.numInstances()];
        Enumeration instanceEnum = data.enumerateInstances();
        for (int i = 0; i < data.numInstances(); i++){
            if (data.instance(i).classValue() != data.instance(i++).classValue()){
                treshold[i] = (data.instance(i).value(attribute) + data.instance(i++).value(attribute))/2;
                gainRatio[i] = Calculator.calcNumericGainRatio(data, attribute, treshold[i]);
            }
        }

        double maxGainTres = treshold[Utils.maxIndex(gainRatio)];
        return maxGainTres;
    }

    private double calculateError(Instances instances) throws Exception {
        int numFalse = 0;
        int numTrue = 0;
        Enumeration instanceEnum = instances.enumerateInstances();
        while (instanceEnum.hasMoreElements()){
            Instance instance = (Instance) instanceEnum.nextElement();
            double predicted = classifyInstance(instance);
            double real_class = instance.classValue();
            if (Utils.eq(predicted, real_class)){
                numTrue += 1;
            }
            else{
                numFalse += 1;
            }
        }
        return (double) numTrue / (double) numFalse;
    }

    private void prune(Instances data) throws Exception {
        if (children != null) {
            // Calculate current error
            double currentError = calculateError(data);

            // Calculate pruned error
            double[] _classDistribution = new double[data.numClasses()];
            Enumeration instanceEnum = data.enumerateInstances();
            while (instanceEnum.hasMoreElements()) {
                Instance instance = (Instance) instanceEnum.nextElement();
                _classDistribution[(int) instance.classValue()]++;
            }
            Utils.normalize(_classDistribution);
            int _decisonIndex = Utils.maxIndex(_classDistribution);


            int numFalse = 0;
            int numTrue = 0;
            instanceEnum = data.enumerateInstances();
            while(instanceEnum.hasMoreElements()){
                Instance instance = (Instance) instanceEnum.nextElement();
                if (_decisonIndex == instance.classIndex()){
                    numTrue += 1;
                }
                else{
                    numFalse += 1;
                }
            }
            double pruned_error = (double) numTrue/ (double) numFalse;

            // Prune process
            if (pruned_error < currentError){
                setChildren(null);
                setSplitAttribute(null);
                setClassIndex(_decisonIndex);
                setClassDistribution(_classDistribution);
            }
        }
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

    public Instances getData() {
        return data;
    }

    public void setData(Instances data) {
        this.data = data;
    }

    public Integer getClassIndex() {
        return classIndex;
    }

    public void setClassIndex(Integer classIndex) {
        this.classIndex = classIndex;
    }
}
