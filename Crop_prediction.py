# imports 
import gradio as gr
import pandas as pd 
import numpy as np
from ydata_profiling import ProfileReport 
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import KMeansSMOTE
import random
from imblearn.pipeline import Pipeline
import joblib
import sqlite3
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
ID=random.randint(10000,99999)
def Eda_Report(data):
    profile = ProfileReport(data, title="EDA Report", explorative=True)
    profile.to_file("eda_report.html") 
    
def Data_devsion(data):
    x=data.drop(columns=['label']);
    y=data['label']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25);
    numerix_feature=x.select_dtypes(include=['number']).columns.to_list();
    categorical_feature=x.select_dtypes(include=['object']).columns.to_list();
    lb=LabelEncoder();
    y_train_numeric=lb.fit_transform(y_train);
    y_test=lb.transform(y_test)
    joblib.dump(lb,'label_encoder.pkl')  
    prosessor=ColumnTransformer([('numerical_feature',StandardScaler(),numerix_feature),('categorical_feature',OneHotEncoder(handle_unknown='ignore'),categorical_feature)],remainder='passthrough')
    smote=KMeansSMOTE();
    models={
        'Adaboost':AdaBoostClassifier(random_state=42),
        'GradiendtBoosting':GradientBoostingClassifier(random_state=42,learning_rate=0.1),
        'xgboost':XGBClassifier(),
        'lightbgm':LGBMClassifier(),
        'Randomforest':RandomForestClassifier(random_state=42),
        'logistic_regrsor':LogisticRegression(),
        'Naive_Bias':GaussianNB()
    
        } 
    result=[]
    for name,model in models.items():
        model.fit(x_train,y_train_numeric)
        p=model.predict(x_test)
        acc=accuracy_score(y_test,p)
        f1=f1_score(y_test,p,average='weighted')
        result.append({'Model':model,'acc_score':acc,'f1_score':f1})
    results_df = pd.DataFrame(result)
    joblib.dump(results_df,'model_performances.pkl',)
    mod=results_df.sort_values(by='f1_score',ascending=False).iloc[0]['Model']
    pipeline=Pipeline([('prosessor',prosessor),('smote',smote),
                   ('model',mod)])
    pipeline.fit(x_train,y_train_numeric)
    joblib.dump(pipeline,"Best_model_pipeline.pkl")  
    return results_df
      
def Data_Base_intialization():   
    conn = sqlite3.connect("crops.db");
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS crop_Predictor (
        id INTEGER NOT NULL ,
        N INTEGER NOT NULL,
        P INTEGER NOT NULL,
        K INTEGER NOT NULL,
        temperature REAL NOT NULL,
        humidity REAL NOT NULL,
        ph REAL NOT NULL,
        rainfall REAL NOT NULL,
        Crop_Predicted Varchar(20) NOT NULL
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Model_perfomance (
      id INTEGER PRIMARY KEY ,
        MODEL_NAME  VARCHAR (20) ,
        ACCURACY_SCORE FLOAT(5) NOT NULL ,
        F1_SCORE FLOAT(5) NOT NULL  
    );
    """)
    conn.commit()
    conn.close()
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
        dataji={
          "N":[N],  
          "P":[P],
          "K":[K],
          "temperature":[temperature],
          "humidity":[humidity],
          "ph":[ph],
          "rainfall":[rainfall]
        }
        dataji=pd.DataFrame(dataji)
        M=joblib.load("Best_model_pipeline.pkl")
        enc=joblib.load('label_encoder.pkl');
        result=enc.inverse_transform(M.predict(dataji))
        conn = sqlite3.connect("crops.db")
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO crop_Predictor 
        (id,N, P, K, temperature, humidity, ph, rainfall, Crop_Predicted)
        VALUES (?,?, ?, ?, ?, ?, ?, ?, ?)
        """, (ID,N, P, K, temperature, humidity, ph, rainfall, result[0]))
        conn.commit()
        conn.close()
        return result[0].upper()    
def interface():
    custom_css = """
    body {
      background-color: #11141a !important;
      font-family: "Poppins", sans-serif !important;
      color: #ffffff;
    }

    #main-container {
      width: 420px;
      margin: auto;
      padding: 25px;
      border-radius: 20px;
      background-color: #1b1f27;
      box-shadow: 0 0 15px rgba(0, 255, 136, 0.05);
      transition: box-shadow 0.3s ease;
    }

    #main-container:hover {
      box-shadow: 0 0 25px rgba(0, 255, 136, 0.6);
    }
    #secondry-box {
      height: 8in; /* original height */
    }



    .title {
      text-align: center;
      font-size: 22px;
      font-weight: 700;
      color: #00ff88;
      margin-bottom: 5px;
    }

    .subtitle {
      text-align: center;
      color: #a9a9a9;
      font-size: 13px;
      margin-bottom: 20px;
    }

    label {
      font-size: 13px !important;
      color: #bfbfbf !important;
      font-weight: 600 !important;
      margin-top: 10px !important;
    }

    input[type="number"] {
      background-color: #0e1015 !important;
      color: #ffffff !important;
      border: 1px solid #333 !important;
      border-radius: 20px !important;
      text-align: center !important;
      height: 40px !important;
      width: 100% !important;
      box-shadow: inset 0 0 8px rgba(0, 0, 0, 0.5);
    }

    #predict-btn {
      background-color: #00ff88 !important;
      color: #0e1015 !important;
      font-weight: 700;
      font-size: 15px;
      border-radius: 25px;
      width: 100%;
      height: 45px;
      margin-top: 20px;
      transition: 0.3s ease;
    }

    #predict-btn:hover {
      background-color: #00cc70 !important;
      transform: scale(1.03);
    }

    #result-box {
      background-color: #0e1015;
      color: #00ff88;
      border-radius: 20px;
      padding: 15px;
      margin-top: 25px;
      text-align: center;
      font-weight: 600;
      border: 1px solid #333;
      box-shadow: 0 0 10px rgba(0, 255, 136, 0.1);
      transition: box-shadow 0.4s ease;
    }

    #result-box.glow {
      box-shadow: 0 0 25px rgba(0, 255, 136, 0.6), 0 0 40px rgba(0, 255, 136, 0.3);
    }
    """

    with gr.Blocks(css=custom_css) as demo:
        with gr.Column(elem_id="main-container"):
         gr.Markdown("""
         <h2 class='title'>Crop Yield Prediction</h2>
         <p class='subtitle'>
         Enter the environmental and nutrient parameters to get a personalized crop prediction.
         </p>
         """)
         with gr.Column(elem_id="secondry-box"):
             N = gr.Number(label="Nitrogen (N)", value=90, info="mg/kg")
             P = gr.Number(label="Phosphorus (P)", value=42, info="mg/kg")
             K = gr.Number(label="Potassium (K)", value=43, info="mg/kg")
             temperature = gr.Number(label="Temperature", value=20.88, info="°C")
             humidity = gr.Number(label="Humidity", value=80.86, info="%")
             ph = gr.Number(label="pH Value", value=6.5, info="pH")
             rainfall = gr.Number(label="Rainfall", value=202.9, info="mm")
         predict_btn = gr.Button("Predict Crop", elem_id="predict-btn")
         result = gr.Markdown(elem_id="result-box")
         predict_btn.click(
                predict_crop, inputs=[N, P, K, temperature, humidity, ph, rainfall], outputs=result
        )
        
    demo.launch(share=True,inline=False,inbrowser=True,pwa=True)
    
def Data_Base_Closing(results_df):  
    conn = sqlite3.connect("crops.db");  
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO Model_perfomance (id,MODEL_NAME, ACCURACY_SCORE, F1_SCORE)
    VALUES (?,?, ?, ?)
    """, (ID,str(results_df.sort_values(by='f1_score',ascending=False).iloc[0]['Model']),
      results_df.sort_values(by='f1_score',ascending=False).iloc[0]['f1_score'],
      results_df.sort_values(by='f1_score',ascending=False).iloc[0]['acc_score'] ))
    conn.commit()
    conn.close()  
  
data=pd.read_csv('C:/datascience to ai/ml/project/Crop_recommendation.csv'); 
Eda_Report(data);
result_df=Data_devsion(data);
Data_Base_intialization();
Data_Base_Closing(result_df)
interface();