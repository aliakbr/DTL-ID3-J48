package IF4071.DecisionTreeLearning.MyC45;

import org.w3c.dom.Attr;
import weka.classifiers.Classifier;
import weka.core.*;
import IF4071.DecisionTreeLearning.Util.Calculator;

import java.io.Serializable;
import java.util.*;

public class MyC45ClassifierTree implements Serializable {
    private Attribute splitAttribute;

    private MyC45ClassifierTree children[];

    private double classDistribution[];

    private Instances data;

    private double classIndex;

    // Treshold for numeric value splitting
    private double treshold;

    public MyC45ClassifierTree(){
        setChildren(null);
        setClassDistribution(null);
        setSplitAttribute(null);
    }

    public void buildClassifier(Instances instances) throws Exception {
        // Set dataset
        setData(instances);

        // Build Tree
        buildTree(data, new Vector<Attribute>());

        // Prune Tree
        prune();
    }

    public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {
        if (splitAttribute == null) {
            return classDistribution;
        } else {
            System.out.println("DEBUG instance: " + instance.toString());
            System.out.println("DEBUG attribute: " + splitAttribute.toString());
            System.out.println("DEBUG instance value: " + instance.value(splitAttribute));
            System.out.println("DEBUG attribute max: " + splitAttribute.numValues());
            System.out.println("DEBUG array max: " + children.length);
            System.out.println();
            return children[(int) instance.value(splitAttribute)].distributionForInstance(instance);
        }
    }

    public double classifyInstance(Instance instance) throws Exception {
        if (Utils.eq(classIndex, -1)){
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

    private void buildTree(Instances data, Vector<Attribute> attributes) {
        if (data.numInstances() == 0) {
            setClassIndex(-1.0);
        } else {
            double majorityClassValue = checkMajorityClass(data);
            // All examples are one class
            if (majorityClassValue != -1.0) {
                setClassIndex(majorityClassValue);
            } else {
                // Hapus atribut yang sudah dicek sebelumnya
                for (Attribute attr : attributes) {
                    data.deleteAttributeAt(attr.index());
                }

                // Hitung gain ratio
                double[] gainRatios = new double[data.numAttributes()];
                Enumeration enumAttr = data.enumerateAttributes();
                while (enumAttr.hasMoreElements()) {
                    Attribute attr = (Attribute) enumAttr.nextElement();
                    if (attr.isNumeric()) {
                        setTreshold(searchTreshold(data, attr));
                        gainRatios[attr.index()] = Calculator.calcNumericGainRatio(data, attr, getTreshold());
                    }
                    else {
                        gainRatios[attr.index()] = Calculator.calcGainRatio(data, attr);
                    }
                }


                int largestGainIdx = Utils.maxIndex(gainRatios);
                Vector<Attribute> nextAttributes = attributes;
                nextAttributes.add(data.attribute(largestGainIdx));
                double gainRatio = gainRatios[largestGainIdx];


                // Generate Pohon
                if (Utils.eq(0, gainRatio)) {
                    classDistribution = new double[data.numClasses()];

                    Enumeration instanceEnum = data.enumerateInstances();
                    while (instanceEnum.hasMoreElements()) {
                        Instance instance = (Instance) instanceEnum.nextElement();
                        classDistribution[(int) instance.classValue()]++;
                    }
                    Utils.normalize(classDistribution);
                    double majorityIndex = Utils.maxIndex(classDistribution);
                    setClassIndex(majorityIndex);
                } else {
                    splitAttribute = data.attribute(largestGainIdx);
                    int numChild;
                    if (splitAttribute.isNominal()){
                        numChild = splitAttribute.numValues();
                    }
                    else{
                        numChild = 2;
                    }

                    children = new MyC45ClassifierTree[numChild];
                    Instances[] splittedInstances;
                    if (splitAttribute.isNumeric()){
                        splittedInstances = Calculator.splitAttributeNumVal(data, getSplitAttribute(), getTreshold());
                    }
                    else{
                        splittedInstances = Calculator.splitDataByAttr(data, getSplitAttribute());
                    }

                    // Proses anak
                    for (int i = 0; i < numChild; i++) {
                        children[i] = new MyC45ClassifierTree();
                        children[i].buildTree(splittedInstances[i], nextAttributes);
                    }

                    for (int i = 0; i < numChild; ++i) {
                        MyC45ClassifierTree child = children[i];
                        if (Utils.eq(child.getClassIndex(), -1.0)) {
                            double[] _classDistribution = new double[data.numClasses()];
                            Enumeration instanceEnum = data.enumerateInstances();

                            while (instanceEnum.hasMoreElements()) {
                                Instance instance = (Instance) instanceEnum.nextElement();
                                _classDistribution[(int) instance.classValue()]++;
                            }

                            Utils.normalize(_classDistribution);
                            double majorityIndex = Utils.maxIndex(_classDistribution);

                            children[i].setClassIndex(majorityIndex);
                            children[i].setClassDistribution(_classDistribution);

                        }
                    }

                }
            }
        }
    }

    private double checkMajorityClass(Instances data){
        double ret = -1.00;

        Attribute attr = data.classAttribute();
        double distribution[] = new double[attr.numValues()];
        Enumeration instanceEnum = data.enumerateInstances();
        while (instanceEnum.hasMoreElements()) {
            Instance instance = (Instance) instanceEnum.nextElement();
            distribution[(int) instance.value(attr)]++;
        }

        boolean foundOne = false;
        for (int i = 0; i < attr.numValues(); i++){
            if (distribution[i] != 0){
                if (!foundOne && ret == -1.00) {
                    foundOne = true;
                    ret = i;
                } else if (foundOne && ret != -1.00){
                    foundOne = false;
                }
            }
        }

        if (!foundOne){
            ret = -1.00;
        }
        return ret;
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

    private void prune() throws Exception {
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
            double _decisonIndex = Utils.maxIndex(_classDistribution);


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
            else{
                for(MyC45ClassifierTree children: getChildren()){
                    children.prune();
                }
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

    public double getClassIndex() {
        return classIndex;
    }

    public void setClassIndex(double classIndex) {
        this.classIndex = classIndex;
    }
}
