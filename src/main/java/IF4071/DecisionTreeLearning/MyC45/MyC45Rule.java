package IF4071.DecisionTreeLearning.MyC45;

import org.w3c.dom.Attr;
import weka.core.Attribute;
import weka.core.Instance;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;

public class MyC45Rule implements Serializable{
    double classValue;
    Map<Attribute, Object> ruleValue = new HashMap<Attribute, Object>();

    public MyC45Rule(){
        Map<Attribute, Object> ruleValue = new HashMap<Attribute, Object>();
    }

    public MyC45Rule(MyC45Rule new_rule){
        classValue = new_rule.getClassValue();
        ruleValue = new_rule.getRuleValue();
    }

    public void addRule(Attribute attr, Object value){
        ruleValue.put(attr, value);
    }

    public void setClassValue(double classValue) {
        this.classValue = classValue;
    }

    public void setRuleValue(Map<Attribute, Object> ruleValue) {
        this.ruleValue = ruleValue;
    }

    public double getClassValue() {
        return classValue;
    }

    public Map<Attribute, Object> getRuleValue() {
        return ruleValue;
    }

    public double classifyInstance(Instance instance){
        boolean found = true;
        Enumeration attrEnum = instance.enumerateAttributes();
        while(attrEnum.hasMoreElements()){
            Attribute attr = (Attribute) attrEnum.nextElement();
            if (attr.isNumeric()){
                double value = (Double) ruleValue.get(attr);
                double treshold = Math.abs(value);
                if (value < 0){
                    if (instance.value(attr) >= treshold){
                        found = false;
                        break;
                    }
                }
                else{
                    if (instance.value(attr) < treshold){
                        found = false;
                        break;
                    }
                }
            }
            else{
                if ((Double) ruleValue.get(attr) != instance.value(attr)){
                    found = false;
                    break;
                }
            }
        }
        if (found){
            return classValue;
        }
        else{
            return -1.0; //Rule fail
        }
    }

    public void addRuleNum(Attribute attr, double treshold, int i) {
        Double tmp = new Double(treshold);
        if (i == 0){
            // Penanda lebih kecil
            tmp = -1 * tmp;
        }
        addRule(attr,tmp);
    }

    public String toString(){
        StringBuilder stringBuilder = new StringBuilder();
        for (Map.Entry<Attribute, Object> entry : ruleValue.entrySet()) {
            Attribute key = entry.getKey();
            Object value = entry.getValue();

            stringBuilder.append("(key: "+key.toString() + " value: "+ value.toString() + "), ");
        }
        stringBuilder.append("class : "+classValue);
        stringBuilder.append("\n");
        return stringBuilder.toString();
    }
}
