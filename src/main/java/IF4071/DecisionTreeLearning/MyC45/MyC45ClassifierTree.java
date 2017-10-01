package IF4071.DecisionTreeLearning.MyC45;

//import com.sun.tools.doclets.formats.html.SourceToHTMLConverter;
import weka.core.*;
import IF4071.DecisionTreeLearning.Util.Calculator;

import java.io.Serializable;
import java.util.*;

public class MyC45ClassifierTree implements Serializable {
    private Attribute splitAttribute;

    private MyC45ClassifierTree children[];

    private double classDistribution[];

    private Instances data;

    private Instances trainingData;

    private Instances validationData;

    private double classIndex;

    // Treshold for numeric value splitting
    private double treshold;

    public MyC45ClassifierTree(){
        setChildren(null);
        setClassDistribution(null);
        setSplitAttribute(null);
    }

    public MyC45ClassifierTree(MyC45ClassifierTree new_root){
        setChildren(new_root.getChildren());
        setClassDistribution(new_root.getClassDistribution());
        setSplitAttribute(new_root.getSplitAttribute());
    }

    public void buildClassifier(Instances instances) throws Exception {
        Instances copy = new Instances(instances);
        setData(copy);

        int trainSize = Math.round(instances.numInstances() * 80 / 100);
        int validationSize = instances.numInstances() - trainSize;
        trainingData = new Instances(instances, 0, trainSize);
        validationData = new Instances(instances, trainSize, validationSize);

        // Build Tree
        buildTree(trainingData, new Vector<Attribute>());

        System.out.println("Prune");
        // Prune Tree
        prune();
    }

    public double[] distributionForInstance(Instance instance){
        if (splitAttribute == null) {
            return classDistribution;
        } else {
            if (splitAttribute.isNumeric()){
                if (instance.value(splitAttribute) >= treshold){
                    return children[1].distributionForInstance(instance);
                }
                else{
                    return children[0].distributionForInstance(instance);
                }
            }
            else{
                return children[(int) instance.value(splitAttribute)].distributionForInstance(instance);
            }
        }
    }

    public double classifyInstance(Instance instance) throws Exception {
        if (splitAttribute != null){
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

    public void buildTree(Instances data, Vector<Attribute> attributes) {
        // Set dataset
        Instances copy = new Instances(data);
        setData(copy);
        int trainSize = Math.round(copy.numInstances() * 80 / 100);
        int validationSize = copy.numInstances() - trainSize;
        trainingData = new Instances(copy, 0, trainSize);
        validationData = new Instances(copy, trainSize, validationSize);

        if (data.numInstances() == 0) {
            classDistribution = new double[data.numClasses()];
            setClassIndex(-1.0);
        } else {
            double majorityClassValue = checkMajorityClass(data);
            // All examples are one class
            if (majorityClassValue != -1.0) {
                Enumeration instanceEnum = data.enumerateInstances();

                classDistribution = new double[data.numClasses()];
                while (instanceEnum.hasMoreElements()) {
                    Instance instance = (Instance) instanceEnum.nextElement();
                    classDistribution[(int) instance.classValue()]++;
                }
                Utils.normalize(classDistribution);

                setClassIndex(majorityClassValue);
            } else {
                // Hitung gain ratio
                double[] gainRatios = new double[data.numAttributes()];
                double[] treshold = new double[data.numAttributes()];
                Enumeration enumAttr = data.enumerateAttributes();
                while (enumAttr.hasMoreElements()) {
                    Attribute attribute = (Attribute) enumAttr.nextElement();

                    boolean sameAttr = false;
                    // Atribut yang sudah dicek sebelumnya tidak dicek kembali
                    for (Attribute attr : attributes){
                        if (attribute.index() == attr.index()){
                            sameAttr = true;
                            break;
                        }
                    }
                    if (attribute.isNumeric()) {
                        if (attribute.index() != data.classIndex() && !sameAttr) {
                            treshold[attribute.index()] = searchTreshold(data, attribute);
                            gainRatios[attribute.index()] = Calculator.calcNumericGainRatio(data, attribute, treshold[attribute.index()]);
                        } else {
                            gainRatios[attribute.index()] = -1;
                        }

                    }
                    else {
                        if (attribute.index() != data.classIndex() && !sameAttr) {
                            gainRatios[attribute.index()] = Calculator.calcGainRatio(data, attribute);
                        } else {
                            gainRatios[attribute.index()] = -1;
                        }
                    }
                }


                int largestGainIdx = Utils.maxIndex(gainRatios);
                setTreshold(treshold[largestGainIdx]);
                double gainRatio = gainRatios[largestGainIdx];

                // Atribut yang tidak dipertimbangkan selanjutnya
                Vector<Attribute> nextAttributes = attributes;
                nextAttributes.add(data.attribute(largestGainIdx));

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
                        splittedInstances = Calculator.splitDataByAttrNum(data, getSplitAttribute(), getTreshold());
                    }
                    else{
                        splittedInstances = Calculator.splitDataByAttr(data, getSplitAttribute());
                    }

//                    System.out.println("DEBUG attribute: " + splitAttribute.toString());
//                    System.out.println("DEBUG max instance: " + splittedInstances.length);
//                    if (splitAttribute.isNumeric()){
//                        System.out.println("DEBUG treshold : " + getTreshold());
//                    }
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
        double[] _treshold = new double[data.numInstances()];
        double[] gainRatio = new double[data.numInstances()];
        for (int i = 0; i < data.numInstances()-1; i++){
            int next_index = i + 1;
            double currentClass = data.instance(i).classValue();
            double nextClass = data.instance(next_index).classValue();
            if(currentClass != nextClass) {
                double current_attr_val = data.instance(i).value(attribute);
                double next_attr_val = data.instance(next_index).value(attribute);
                _treshold[i] = (current_attr_val + next_attr_val)/2;
                gainRatio[i] = Calculator.calcNumericGainRatio(data, attribute, _treshold[i]);
            }
        }
        double res = (double) _treshold[Utils.maxIndex(gainRatio)];
        return res;
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

        return (double) numFalse / (double) (numTrue + numFalse);
    }

    private void prune() throws Exception {
        if (children != null) {
            // Calculate current error
            double currentError = calculateError(validationData);

            // Calculate pruned error
            double[] _classDistribution = new double[validationData.numClasses()];
            Enumeration instanceEnum = validationData.enumerateInstances();
            while (instanceEnum.hasMoreElements()) {
                Instance instance = (Instance) instanceEnum.nextElement();
                _classDistribution[(int) instance.classValue()]++;
            }
            Utils.normalize(_classDistribution);
            double _decisonIndex = Utils.maxIndex(_classDistribution);


            int numFalse = 0;
            int numTrue = 0;
            instanceEnum = validationData.enumerateInstances();
            while(instanceEnum.hasMoreElements()){
                Instance instance = (Instance) instanceEnum.nextElement();
                if (_decisonIndex == instance.classIndex()){
                    numTrue += 1;
                }
                else{
                    numFalse += 1;
                }
            }
            double pruned_error = (double) numFalse / (double) (numTrue + numFalse);

            // Prune process
            System.out.println("Curr Error : "+currentError);
            System.out.println("Pruned Error : "+pruned_error);
            if (pruned_error < currentError){
                System.out.println("DEBUG Attribute pruned : "+splitAttribute.toString());
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
        else{
            System.out.println("No Pruning");
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
