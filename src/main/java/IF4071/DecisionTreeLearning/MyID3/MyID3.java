package IF4071.DecisionTreeLearning.MyID3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Instance;
import weka.core.Instances;

public class MyID3 implements Classifier {

    private MyID3ClassifierTree root;

    public void buildClassifier(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);

        Instances data = new Instances(instances);
        data.deleteWithMissingClass();
        root = new MyID3ClassifierTree();
        root.buildTree(instances);
    }

    public double classifyInstance(Instance instance) throws Exception {
        return 0;
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
    }

    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities((CapabilitiesHandler) this);
        result.disableAll();

        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.DATE_ATTRIBUTES);

        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        result.setMinimumNumberInstances(0);

        return result;
    }
}
