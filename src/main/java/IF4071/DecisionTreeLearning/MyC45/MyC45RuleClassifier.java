package IF4071.DecisionTreeLearning.MyC45;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.io.Serializable;
import java.util.*;

public class MyC45RuleClassifier implements Serializable {
    MyC45ClassifierTree root;
    ArrayList<MyC45Rule> rule_list = new ArrayList<MyC45Rule>();
    private Instances data;

    private Instances trainingData;

    private Instances validationData;

    public void generateRuleFromPath(MyC45Rule curr_rule, MyC45ClassifierTree curr_path){
        if ((root.getSplitAttribute()) != null && (curr_path.getChildren() != null)){
            for (int i = 0; i < curr_path.getChildren().length; i++){
                if (curr_rule == null){
                    curr_rule = new MyC45Rule();
                }
                Attribute attr = curr_path.getSplitAttribute();
                if (attr.isNumeric()){
                    curr_rule.addRuleNum(attr, curr_path.getTreshold(), i);
                }
                else{
                    curr_rule.addRule(attr, attr.value(i));
                }
                if (curr_path.getChildren()[i] != null) {
                    MyC45Rule child_rule = new MyC45Rule(curr_rule);
                    generateRuleFromPath(child_rule, curr_path.getChildren()[i]);
                    addRule_list(curr_rule);
                }
                else{
                    addRule_list(curr_rule);
                }
            }
        }
        else{
            System.out.println("Class value "+curr_path.getClassIndex());
            curr_rule.setClassValue(curr_path.getClassIndex());
            addRule_list(curr_rule);
            System.out.println("Class value set "+ curr_rule.getClassValue());
        }
    }

    public void buildClassifier(Instances instances){
        Instances data = new Instances(instances);
        Instances copy = new Instances(instances);
        setData(data);

        int trainSize = Math.round(instances.numInstances() * 80 / 100);
        int validationSize = instances.numInstances() - trainSize;
        trainingData = new Instances(instances, 0, trainSize);
        validationData = new Instances(instances, trainSize, validationSize);

        root = new MyC45ClassifierTree();

        // Buat pohon
        root.buildTree(data, new Vector<Attribute>());

        // Generate rule
        generateRuleFromPath(new MyC45Rule(), root);

        // Hapus semua rule dengan class initial -1.0
        removeInitialValue();

        // Pruning
        ArrayList<MyC45Rule> newRuleList = prune(rule_list);
        rule_list = newRuleList;

        System.out.println(rule_list.toString());
        for (MyC45Rule rule: rule_list){
            System.out.println(rule.toString());
        }
    }

    public double classifyInstance(Instance instance){
        // MASIH SALAH
        double result = 0.0;
        for (MyC45Rule rule: rule_list){
            if (!Utils.eq(rule.classifyInstance(instance), -1.0)){
                result = rule.classifyInstance(instance);
            }
        }
        return result;
    }

    public double classifyInstanceWithRule(Instance instance, ArrayList<MyC45Rule> rule_list){
        // MASIH SALAH
        double result = 0.0;
        for (MyC45Rule rule: rule_list){
            if (!Utils.eq(rule.classifyInstance(instance), -1.0)){
                result = rule.classifyInstance(instance);
            }
        }
        return result;
    }

    public String toString(){
        StringBuilder output = new StringBuilder();
        for (MyC45Rule rule: rule_list){
            output.append(rule.toString());
        }
        output.append("\n");
        return output.toString();
    }

    public double calcError(Instances instances, ArrayList<MyC45Rule> rule){
        int numFalse = 0;
        int numTrue = 0;
        Enumeration instanceEnum = instances.enumerateInstances();
        while (instanceEnum.hasMoreElements()){
            Instance instance = (Instance) instanceEnum.nextElement();
            double predicted = classifyInstanceWithRule(instance, rule);
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

    public ArrayList<MyC45Rule> prune(ArrayList<MyC45Rule> rule) {
        List<ArrayList<MyC45Rule>> ruleList = new ArrayList<ArrayList<MyC45Rule>>();
        ruleList.add(rule);
        List<Double> akurasiList = new ArrayList<Double>();
        akurasiList.add(calcError(validationData, rule));

        for (int i = 0; i < rule.size(); i++){
            for(Iterator<Map.Entry<Attribute, Object>> it = rule.get(i).getRuleValue().entrySet().iterator(); it.hasNext(); ) {
                Map.Entry<Attribute, Object> entry = it.next();
                Attribute key = entry.getKey();
                Object value = entry.getValue();

                // Hapus Rule
                ArrayList<MyC45Rule> ruleNew = new ArrayList<MyC45Rule>(rule);
                for (int j = 0; j < ruleNew.size(); j++) {
                    boolean found = false;
                    for (Iterator<Map.Entry<Attribute, Object>> it2 = ruleNew.get(i).getRuleValue().entrySet().iterator(); it.hasNext(); ) {
                        Map.Entry<Attribute, Object> entry1 = it2.next();
                        Attribute key1 = entry1.getKey();
                        Object value1 = entry1.getValue();
                        if ((key == key1) && (value == value1)){
                            found = true;
                            it2.remove();
                            break;
                        }
                    }
                    if (found){
                        break;
                    }
                }

                akurasiList.add(calcError(validationData, ruleNew));
                ruleList.add(ruleNew);
            }
        }

        int largestIdx = 0;
        Double max = 0.0;
        for (int i = 0; i < akurasiList.size(); i++){
            if (max < akurasiList.get(i)){
                largestIdx = i;
                max = akurasiList.get(i);
            }
        }
        return ruleList.get(largestIdx);
    }

    public MyC45ClassifierTree getRoot() {
        return root;
    }

    public void setRoot(MyC45ClassifierTree root) {
        this.root = root;
    }

    public ArrayList<MyC45Rule> getRule_list() {
        return rule_list;
    }

    public void setRule_list(ArrayList<MyC45Rule> rule_list) {
        this.rule_list = rule_list;
    }

    private void addRule_list(MyC45Rule rule){
        boolean same = false;

        Vector<MyC45Rule> ruleToRemove = new Vector<MyC45Rule>();
        for (MyC45Rule curr : rule_list){
            if (curr.ruleValue.equals(rule.ruleValue)){
                same = true;
                if (curr.classValue != rule.classValue && curr.getClassValue() == -1.0){
                    ruleToRemove.add(curr);
                    same = false;
                }
            }
        }

        for (MyC45Rule curr : ruleToRemove){
            rule_list.remove(curr);
        }

        if (!same){
            rule_list.add(rule);
        }
    }

    private void removeInitialValue() {
        Vector<MyC45Rule> ruleToRemove = new Vector<MyC45Rule>();

        for (MyC45Rule curr : rule_list){
            if (curr.getClassValue() == -1.0){
                ruleToRemove.add(curr);
            }
        }

        for (MyC45Rule curr : ruleToRemove){
            rule_list.remove(curr);
        }
    }

    public Instances getData() {
        return data;
    }

    public void setData(Instances data) {
        this.data = data;
    }
}
