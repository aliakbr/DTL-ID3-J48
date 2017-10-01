package IF4071.DecisionTreeLearning.Util;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

public class Calculator {
    // Entropy calculator
    public static double calculateEntropy(Instances data) {
        if (data.numInstances() == 0) return 0.0;

        double[] classCounts = new double[data.numClasses()];
        Enumeration instanceEnum = data.enumerateInstances();
        while (instanceEnum.hasMoreElements()) {
            Instance inst = (Instance) instanceEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }

        double entropy = 0;
        for (int i = 0; i < data.numClasses(); i++) {
            double fraction = classCounts[i] / data.numInstances();
            if (fraction != 0) {
                entropy -= fraction * Utils.log2(fraction);
            }
        }

        return entropy;
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
                double frac = (double) splitData[i].numInstances() / (double) data.numInstances();
                infoGain -=  frac * calculateEntropy(splitData[i]);
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
            return infogain / intrinsicValue;
        }
    }

    // Numeric handler

    //Split by attr val
    public static Instances[] splitDataByAttrNum(Instances data, Attribute attr, double treshold){
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

//        System.out.println("Data Below "+ treshold + " Count : "+splittedData[0].numInstances());
//        System.out.println("Data Below "+ treshold + " Count : "+splittedData[1].numInstances());

        return splittedData;
    }

    // Numeric Information Gain
    public static double numericInformationGain(Instances data, Attribute attr, double treshold){
        double infoGain = 0.0;
        Instances[] splitData = splitDataByAttrNum(data, attr, treshold);
        infoGain = calculateEntropy(data);
        for (int i = 0; i < 2; i++) {
            if (splitData[i].numInstances() > 0) {
                double frac = (double) splitData[i].numInstances() / (double) data.numInstances();
                infoGain -=  frac * calculateEntropy(splitData[i]);
            }
        }
        return infoGain;
    }

    public static double calcNumericGainRatio (Instances data, Attribute attr, double treshold){
        double infogain = numericInformationGain(data, attr,treshold);
        if (Utils.eq(0.0, infogain)){
            return 0.0;
        }
        else{
            double intrinsicValue = 0.0;
            Instances[] splitData;
            splitData = splitDataByAttrNum(data, attr, treshold);
            for (int i = 0; i < 2; ++i) {
                if (splitData[i].numInstances() > 0) {
                    double frac = (double)splitData[i].numInstances()/(double)data.numInstances();
                    intrinsicValue -= frac * Utils.log2(frac);
                }
            }

            double value = infogain/intrinsicValue;
            return value;
        }
    }
}
