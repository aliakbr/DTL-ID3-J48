package IF4071.DecisionTreeLearning.MyID3;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;

import java.util.List;
import java.util.Vector;

public class MyID3 extends AbstractClassifier {

    private MyID3ClassifierTree root;

    public void buildClassifier(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);

        Instances data = new Instances(instances);
        data.deleteWithMissingClass();
        root = new MyID3ClassifierTree();
        root.buildTree(instances, new Vector<Attribute>());
    }

    public double classifyInstance(Instance instance) throws Exception {
        return root.classifyInstance(instance);
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        return root.distributionForInstance(instance);
    }

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        result.setMinimumNumberInstances(0);

        return result;
    }
}
