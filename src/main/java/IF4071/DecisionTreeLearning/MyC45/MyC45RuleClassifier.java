package IF4071.DecisionTreeLearning.MyC45;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Vector;

public class MyC45RuleClassifier implements Serializable {
    MyC45ClassifierTree root;
    ArrayList<MyC45Rule> rule_list = new ArrayList<MyC45Rule>();

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

        root = new MyC45ClassifierTree();

        // Buat pohon
        root.buildTree(data, new Vector<Attribute>());

        // Generate rule
        generateRuleFromPath(new MyC45Rule(), root);
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

    public String toString(){
        StringBuilder output = new StringBuilder();
        for (MyC45Rule rule: rule_list){
            output.append(rule.toString());
        }
        output.append("\n");
        return output.toString();
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

        for (MyC45Rule curr : rule_list){
            if (curr.ruleValue.equals(rule.ruleValue) && curr.classValue == rule.classValue){
                same = true;
            }
        }

        if (!same){
            rule_list.add(rule);
        }
    }
}
