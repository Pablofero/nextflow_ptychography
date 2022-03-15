def toArgs1(nextflow.script.ScriptBinding$ParamsMap p){//only one (1) level deep!!
    
    results = []
    p.each{results << "--$it.key ${it.value.getClass()!=java.util.ArrayList ? '\''+it.value.toString()+'\'' : (it.value).toString().replaceAll(",","").replace("[","").replace("]","")}"}
    return results.join(' ')
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