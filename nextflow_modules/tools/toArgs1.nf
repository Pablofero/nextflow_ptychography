def toArgs1(nextflow.script.ScriptBinding$ParamsMap p){//only one (1) level deep!!
    // Converts the params custom map to a command line argument format,  adding an -- for the flag/key and quoting the arguments with ' (permalink to custom map: https://github.com/nextflow-io/nextflow/blob/8e5129d69d9cf9458ce92baa5654cdbf96484006/modules/nextflow/src/main/groovy/nextflow/script/ScriptBinding.groovy)
    //the ? : is a ternary_operator, see https://groovy-lang.org/operators.html#_ternary_operator
    // the .each{} is a groovy closures, see https://www.nextflow.io/docs/latest/script.html#closures and  http://groovy-lang.org/closures.html
    // single quotes avoid interpretation of the contents by bash
    results = []
    p.each{results << "--$it.key "+"${it.value.getClass()!=java.util.ArrayList ? '\''+it.value.toString()+'\'' : list_to_quoted_string(it.value)}"}
    return results.join(' ')
}

def list_to_quoted_string(java.util.ArrayList a){
    // single quotes avoid interpretation of the contents by bash
    // the .each{} is a groovy closures, see https://www.nextflow.io/docs/latest/script.html#closures and  http://groovy-lang.org/closures.html
    resultss = []
    a.each{resultss <<  '\''+"$it"+ '\''}
    return resultss.join(' ')
}

// def filterSpecialChar(String S){// based on https://stackoverflow.com/a/18619944
//     Map<String, String> map = new HashMap<String, String>();
//     // map.put("(", "\\(");
//     // map.put(")", "\\)");
//     for (Map.Entry<String, String> entry : map.entrySet()) {
//         S = S.replace(entry.getKey(), entry.getValue());
//     }
//     return S
// }