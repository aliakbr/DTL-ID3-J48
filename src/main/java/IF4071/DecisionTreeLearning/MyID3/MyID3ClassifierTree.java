package IF4071.DecisionTreeLearning.MyID3;

import IF4071.DecisionTreeLearning.Util.Calculator;
import weka.core.*;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.Vector;

public class MyID3ClassifierTree implements Serializable {

    private MyID3ClassifierTree[] children;

    private Attribute splitAttribute;

    private double classValue;

    private double[] classDistribution;

    public MyID3ClassifierTree(){
        children = null;
        splitAttribute = null;
        classDistribution = null;
    }

    public double classifyInstance(Instance instance) {
        if (Utils.eq(classValue, -1)){
            double splitAttrIdx;

            splitAttrIdx = instance.value(splitAttribute);

            return children[(int) splitAttrIdx].classifyInstance(instance);
        } else {
            return classValue;
        }
    }

    public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("MyID3: missing values not supported");
        }

        if (splitAttribute == null) {
            return classDistribution;
        } else {
            return children[(int) instance.value(splitAttribute)].distributionForInstance(instance);
        }
    }

    public void buildTree(Instances data, Vector<Attribute> attributes){
        if (data.numInstances() == 0) {
            classDistribution = new double[data.numClasses()];
            classValue = -1;
        } else {
            double majorityClassValue = checkMajorityClass(data);
            // All examples are one class
            if (majorityClassValue != -1.00) {
                Enumeration instanceEnum = data.enumerateInstances();

                classDistribution = new double[data.numClasses()];
                while (instanceEnum.hasMoreElements()) {
                    Instance instance = (Instance) instanceEnum.nextElement();
                    classDistribution[(int) instance.classValue()]++;
                }
                Utils.normalize(classDistribution);

                classValue = majorityClassValue;
            } else {
                double[] infoGains = new double[data.numAttributes()];

                Enumeration enumeration = data.enumerateAttributes();

                // Hitung gain ratio tiap atribut
                while (enumeration.hasMoreElements()) {
                    Attribute attribute = (Attribute) enumeration.nextElement();

                    boolean sameAttr = false;
                    // Atribut yang sudah dicek sebelumnya tidak dicek kembali
                    for (Attribute attr : attributes){
                        if (attribute.index() == attr.index()){
                            sameAttr = true;
                        }
                    }
                    if (attribute.index() != data.classIndex() && !sameAttr) {
                        infoGains[attribute.index()] = Calculator.calcInfoGain(data, attribute);
                    } else {
                        infoGains[attribute.index()] = -1;
                    }
                }

                int largestGainIdx = Utils.maxIndex(infoGains);

                // Atribut yang tidak dipertimbangkan selanjutnya
                Vector<Attribute> nextAttributes = attributes;
                nextAttributes.add(data.attribute(largestGainIdx));



                // Generate Pohon
                if (Utils.eq(0, infoGains[largestGainIdx])){
                    classDistribution = new double[data.numClasses()];

                    Enumeration instanceEnum = data.enumerateInstances();
                    while (instanceEnum.hasMoreElements()) {
                        Instance instance = (Instance) instanceEnum.nextElement();
                        classDistribution[(int) instance.classValue()]++;
                    }
                    Utils.normalize(classDistribution);

                    classValue = Utils.maxIndex(classDistribution);


                } else {
                    splitAttribute = data.attribute(largestGainIdx);
                    int numChild = splitAttribute.numValues();

                    children = new MyID3ClassifierTree[numChild];
                    Instances[] splittedInstances = Calculator.splitDataByAttr(data, splitAttribute);

                    // Proses anak
                    for (int i = 0; i < numChild; i++){
                        children[i] = new MyID3ClassifierTree();
                        children[i].buildTree(splittedInstances[i], nextAttributes);
                    }

                    for (int i = 0; i < numChild; ++i) {
                        MyID3ClassifierTree child = children[i];
                        if (Utils.eq(child.getClassValue(), -1)){
                            double[] _classDistribution = new double[data.numClasses()];
                            Enumeration instanceEnum = data.enumerateInstances();

                            while (instanceEnum.hasMoreElements()) {
                                Instance instance = (Instance) instanceEnum.nextElement();
                                _classDistribution[(int) instance.classValue()]++;
                            }

                            Utils.normalize(_classDistribution);
                            double majorityIndex = Utils.maxIndex(_classDistribution);

                            children[i].setClassValue(majorityIndex);
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

    public MyID3ClassifierTree[] getChildren() {
        return children;
    }

    public void setChildren(MyID3ClassifierTree[] children) {
        this.children = children;
    }

    public Attribute getSplitAttribute() {
        return splitAttribute;
    }

    public void setSplitAttribute(Attribute splitAttribute) {
        this.splitAttribute = splitAttribute;
    }

    public double getClassValue() {
        return classValue;
    }

    public void setClassValue(double classValue) {
        this.classValue = classValue;
    }

    public double[] getClassDistribution() {
        return classDistribution;
    }

    public void setClassDistribution(double[] classDistribution) {
        this.classDistribution = classDistribution;
    }
}
