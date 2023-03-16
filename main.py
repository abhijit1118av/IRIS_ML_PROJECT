from flask import Flask,render_template,request
import pickle
import json 
import numpy as np
import CONFIG
with open(CONFIG.model_path,'rb') as file:
    model=pickle.load(file)
with open(CONFIG.asset_path,"r") as file:
    asset=json.load(file)

app=Flask(__name__)
@app.route('/')
def default():

    return render_template("front.html")

@app.route('/get_data',methods=["POST","GET"])
def integrate():
    input_data=request.form
    print(input_data)
    data=np.zeros(len(asset["columns"]))
    data[0]=input_data["sepal_length"]
    data[1]=input_data["sepal_width"]
    data[2]=input_data["petal_length"]
    data[3]=input_data["petal_width"]
    result=model.predict([data])
    print(result)
    if result[0]==1:
        iris_value="SETOSA"
    elif result[0]==2:
        iris_values="VERSICOLOR"
    else:
        iris_value="VERGINICA"
 

    return render_template('front.html',PREDICT_VALUE=iris_value)





if __name__=="__main__":
    app.run(host=CONFIG.host_name,port=CONFIG.port_value)