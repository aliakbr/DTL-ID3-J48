package IF4071.DecisionTreeLearning.Util;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

public class Calculator {
    // Entropy calculator
    public static double calculateEntropy(Instances data) {
        if (data.numInstances() == 0){
            return 0.0;
        }
        else{
            int totalClass = data.numClasses();
            //System.out.println("DEBUG total class: " + totalClass);
            double[] classCount = new double[totalClass];
            Enumeration instanceEnum = data.enumerateInstances();
            while (instanceEnum.hasMoreElements()){
                Instance instance = (Instance) instanceEnum.nextElement();
                //System.out.println("DEBUG class index: " + instance.classIndex());
                //System.out.println("DEBUG class value: " + instance.classValue());
                classCount[(int) instance.classValue()]++;
            }
            double entropy = 0.0;

            for (int i = 0; i < classCount.length; i++){
                double frac = classCount[i]/data.numInstances();
                entropy += -1 * frac * Utils.log2(frac);
            }

            return entropy;
        }
    }

    // Split data by attr
    public static Instances[] splitDataByAttr(Instances data, Attribute attr){
        Instances[] splittedData = new Instances[attr.numValues()];
        for (int i = 0; i < attr.numValues(); i++){
            splittedData[i] = new Instances(data, data.numInstances()); //Initialize splitted instances
        }

        Enumeration instanceEnum = data.enumerateInstances();
        while (instanceEnum.hasMoreElements()){
            Instance instance = (Instance) instanceEnum.nextElement();
            splittedData[(int) instance.value(attr)].add(instance);
        }

        for (Instances instances : splittedData) {
            instances.compactify(); // to reduce array size to fit num of instances
        }

        return splittedData;
    }

    // Gain Ratio Calculator
    public static double calcInfoGain(Instances data, Attribute attr){

        double infoGain=0.0;
        Instances[] splitData = splitDataByAttr(data, attr);
        infoGain = calculateEntropy(data);
        for (int i = 0; i < attr.numValues(); i++) {
            if (splitData[i].numInstances() > 0) {
                infoGain -= (double) splitData[i].numInstances() /
                        (double) data.numInstances() * calculateEntropy(splitData[i]);
            }
        }
        return infoGain;
    }

    // Information Gain Calculator
    public static double calcGainRatio (Instances data, Attribute attr) {
        double infogain = calcInfoGain(data, attr);
        if (Utils.eq(0.0, infogain)){
            return 0.0;
        }
        else{
            double intrinsicValue = 0.0;
            Instances[] splitData;
            splitData = splitDataByAttr(data, attr);
            for (int i = 0; i < splitData.length; ++i) {
                if (splitData[i].numInstances()>0) {
                    double frac = (double) splitData[i].numInstances() / (double) data.numInstances();
                    intrinsicValue -= frac * Utils.log2(frac);
                }
            }
            return calcInfoGain(data, attr) / intrinsicValue;
        }
    }

    // Numeric handler

    //Split by attr val
    public static Instances[] splitAttributeNumVal(Instances data, Attribute attr, double treshold){
        Instances[] splittedData = new Instances[2];
        for (int i = 0; i < 2; i++){
            splittedData[i] = new Instances(data, data.numInstances()); //Initialize splitted instances
        }

        Enumeration instanceEnum = data.enumerateInstances();
        while (instanceEnum.hasMoreElements()){
            Instance instance = (Instance) instanceEnum.nextElement();
            if (instance.value(attr) >= treshold){
                splittedData[1].add(instance);
            }
            else{
                splittedData[0].add(instance);
            }
        }

        for (Instances instances : splittedData) {
            instances.compactify(); // to reduce array size to fit num of instances
        }

        return splittedData;
    }

    // Numeric Information Gain
    public static double numericInformationGain(Instances data, Attribute attr, double treshold){
        double infoGain = 0.0;
        Instances[] splitData = splitAttributeNumVal(data, attr, treshold);
        infoGain = calculateEntropy(data);
        for (int i = 0; i < 2; i++) {
            if (splitData[i].numInstances() > 0) {
                infoGain -= (double) splitData[i].numInstances() /
                        (double) data.numInstances() * calculateEntropy(splitData[i]);
            }
        }
        return infoGain;
    }

    //Split In Attribute value for Numeric
    public static double SplitInAttributeNumeric(Instances data, Attribute attr, double treshold){
        double splitValue = 0.0;
        Instances[] splitData = splitAttributeNumVal(data, attr, treshold);
        for (int i = 0; i < splitData.length; i++){
            double frac = splitData[i].numInstances()/data.numInstances();
            splitValue += -1 * frac * Utils.log2(frac);
        }
        return splitValue;
    }

    public static double calcNumericGainRatio (Instances data, Attribute attr, double threshold){
        double infogain = numericInformationGain(data, attr,threshold);
        if (Utils.eq(0.0, infogain)) return 0.0;
        return numericInformationGain(data, attr, threshold) / SplitInAttributeNumeric(data, attr, threshold);
    }
}
