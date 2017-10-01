package IF4071.DecisionTreeLearning.MyC45;

import weka.classifiers.AbstractClassifier;
import weka.core.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;

public class MyC45 extends AbstractClassifier {
    public MyC45ClassifierTree root;
    public MyC45RuleClassifier ruleClassifier = new MyC45RuleClassifier();
    boolean rule = false;

    public MyC45(){
        root = new MyC45ClassifierTree();
    }
    public MyC45(Instances instances) {
        root = new MyC45ClassifierTree();
        root.setData(instances);
    }

    public MyC45(boolean rule){
        this.rule = rule;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);

        Instances data = new Instances(instances);
        data.deleteWithMissingClass();

        // Handling missing value for numeric and nominal value
        data = replaceMissingValues(data);

        if (!rule) {
            root.buildClassifier(data);
        }
        else{
            ruleClassifier.buildClassifier(data);
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (!rule) {
            return root.classifyInstance(instance);
        }
        else{
            return ruleClassifier.classifyInstance(instance);
        }
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        result.setMinimumNumberInstances(0);

        return result;
    }

    public Instances replaceMissingValues(Instances input) {
        Instances data = new Instances(input);
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
        return data;
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
}
