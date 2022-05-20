import streamlit as st
from PIL import Image
import pandas as pd 
from streamlit_multipage import MultiPage

import functions1

from sklearn.utils import estimator_html_repr
from streamlit_option_menu import option_menu

import plotly.express as px  
import numpy as np
import plotly.figure_factory as ff
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode,iplot
import streamlit as st 
import chart_studio.plotly as py
import plotly.graph_objects as go


st.set_page_config(
    layout= 'wide')



st.title('Heart Attack Risk Predictor')

#Menu Bar
selected = option_menu(
        menu_title = None,
        options = ["Home", "EDA", "VIZ", "Model"],
        default_index=0,
        orientation ='horizontal'
        )
##########################################################
if selected == "Home":
    
    #st.image()
    #image = Image.open(r"C:\Users\BC\Documents\imageee.jpg")
    st.image("https://www.linkpicture.com/q/hhhhh.jpg", width=1200)
    st.markdown("""---""")
############################################################
if selected == "EDA":
   
    st.write('<p style="font-size:130%">Import Dataset</p>', unsafe_allow_html=True)

    file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
    dataset = st.file_uploader(label = '')
    if dataset is None:
        st.write('')
    
    use_defo = st.checkbox('Use example Dataset')
    if use_defo:
        dataset = r"C:\Users\BC\Downloads\heart.csv"

    if dataset:
        if file_format == 'csv' or use_defo:
            df = pd.read_csv(dataset)
        else:
            df = pd.read_excel(dataset)

    
    st.subheader('Dataframe:')
    n, m = df.shape
    st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)   
    st.dataframe(df)


    all_vizuals = ['Info', 'NA Info', 'Descriptive Analysis', 'Target Analysis', 
                   'Distribution of Numerical Columns', 
                   'Box Plots',  'Outlier Analysis']
          
    vizuals = st.sidebar.multiselect("Choose which visualizations you want to see 👇", all_vizuals)

    if 'Info' in vizuals:
        st.subheader('Info:')
        c1, c2, c3 = st.columns([1, 2, 1])
        c2.dataframe(functions1.df_info(df))
    
    if 'NA Info' in vizuals:
        st.subheader('NA Value Information:')
        if df.isnull().sum().sum() == 0:
            st.write('There is not any NA value in your dataset.')
        else:
            c1, c2, c3 = st.columns([0.5, 2, 0.5])
            c2.dataframe(functions1.df_isnull(df), width=1500)
            functions1.space(2)
            

    if 'Descriptive Analysis' in vizuals:
        st.subheader('Descriptive Analysis:')
        st.dataframe(df.describe())
        
    if 'Target Analysis' in vizuals:
        st.subheader("Select target column:")    
        target_column = st.selectbox("", df.columns, index = len(df.columns) - 1)
    
        st.subheader("Histogram of target column")
        fig = px.histogram(df, x = target_column)
        c1, c2, c3 = st.columns([0.5, 2, 0.5])
        c2.plotly_chart(fig)


    num_columns = df.select_dtypes(exclude = 'object').columns
    cat_columns = df.select_dtypes(include = 'object').columns

    if 'Distribution of Numerical Columns' in vizuals:

        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = functions1.sidebar_multiselect_container('Choose columns for Distribution plots:', num_columns, 'Distribution')
            st.subheader('Distribution of numerical columns')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_num_cols)):
                        break

                    fig = px.histogram(df, x = selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width = True)
                    i += 1

    

    if 'Box Plots' in vizuals:
        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = functions1.sidebar_multiselect_container('Choose columns for Box plots:', num_columns, 'Box')
            st.subheader('Box plots')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:
                    
                    if (i >= len(selected_num_cols)):
                        break
                    
                    fig = px.box(df, y = selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width = True)
                    i += 1

    if 'Outlier Analysis' in vizuals:
        st.subheader('Outlier Analysis')
        c1, c2, c3 = st.columns([1, 2, 1])
        c2.dataframe(functions1.number_of_outliers(df))

    
####################################################
if selected == 'VIZ':
    df= pd.read_csv(r"C:\Users\BC\Downloads\heart.csv")

    col1, col2= st.columns([9,9])

    

    
    fig0 = px.histogram(df,x= "target", color="target")
            
    fig0.update_layout(title_text='Distribution of target')
    fig0.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)'})
    

        
    col1.plotly_chart(fig0)
#####################################################
    fig1 = px.histogram(df, x = "target", color = "sex", barmode = "group")
    fig1.update_layout(title_text='Target vs Sex')
    fig1.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)'})
    col2.plotly_chart(fig1)

    

	
######################################################


    fig2=px.histogram(df,x= "target", color="exang", barmode="group")

    fig2.update_layout(title_text='Heart Attack vs Exercise Induced Angina')
    fig2.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)'})
    col1.plotly_chart(fig2)
    #f
    
	
#######################################################
    fig3=px.histogram(df,x="target", color="fbs", barmode="group")

    fig3.update_layout(title_text='Heart Attack vs fasting blood sugar')
    fig3.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)'})
    col2.plotly_chart(fig3)

#########################################################
from sklearn.svm import SVC
df=pd.read_csv(r"C:\Users\BC\Downloads\heart.csv")
SVC_clf=SVC()
X=df.drop("target",axis=1).values
Y=df.target.values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,test_size=0.2)
SVC_clf.fit(X_train,Y_train)
SVC_score=SVC_clf.score(X_test,Y_test)
SVC_Y_pred=SVC_clf.predict(X_test)
from pyngrok import ngrok
import streamlit as st
import base64
import sklearn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scal=MinMaxScaler()


def preprocess(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal ):   
 
    
    # Pre-processing user input   
    if sex=="male":
        sex=1 
    else: sex=0
    
    
    if cp=="Typical angina":
        cp=0
    elif cp=="Atypical angina":
        cp=1
    elif cp=="Non-anginal pain":
        cp=2
    elif cp=="Asymptomatic":
        cp=2
    
    if exang=="Yes":
        exang=1
    elif exang=="No":
        exang=0
 
    if fbs=="Yes":
        fbs=1
    elif fbs=="No":
        fbs=0
 
    if slope=="Upsloping: better heart rate with excercise(uncommon)":
        slope=0
    elif slope=="Flatsloping: minimal change(typical healthy heart)":
          slope=1
    elif slope=="Downsloping: signs of unhealthy heart":
        slope=2  
 
    if thal=="fixed defect: used to be defect but ok now":
        thal=6
    elif thal=="reversable defect: no proper blood movement when excercising":
        thal=7
    elif thal=="normal":
        thal=2.31

    if restecg=="Nothing to note":
        restecg=0
    elif restecg=="ST-T Wave abnormality":
        restecg=1
    elif restecg=="Possible or definite left ventricular hypertrophy":
        restecg=2


    user_input=[age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal]
    user_input=np.array(user_input)
    user_input=user_input.reshape(1,-1)
    user_input=scal.fit_transform(user_input)
    prediction = SVC_clf.predict(user_input)

    return prediction


if selected == "Model":
    df=pd.read_csv(r"C:\Users\BC\Downloads\heart.csv")
    # following lines create boxes in which user can enter data required to make prediction
    age=st.selectbox ("Age",range(1,121,1))
    sex = st.radio("Select Gender: ", ('male', 'female'))
    cp = st.selectbox('Chest Pain Type',("Typical angina","Atypical angina","Non-anginal pain","Asymptomatic")) 
    trestbps=st.selectbox('Resting Blood Sugar',range(60,500,10))
    restecg=st.selectbox('Resting Electrocardiographic Results',("Nothing to note","ST-T Wave abnormality","Possible or definite left ventricular hypertrophy"))
    chol=st.selectbox('Serum Cholestoral in mg/dl',range(150,750,50))
    fbs=st.radio("Fasting Blood Sugar higher than 120 mg/dl", ['Yes','No'])
    thalach=st.selectbox('Maximum Heart Rate Achieved',range(1,300,5))
    exang=st.selectbox('Exercise Induced Angina',["Yes","No"])
    oldpeak=st.number_input('Oldpeak')
    slope = st.selectbox('Heart Rate Slope',("Upsloping: better heart rate with excercise(uncommon)","Flatsloping: minimal change(typical healthy heart)","Downsloping: signs of unhealthy heart"))
    ca=st.selectbox('Number of Major Vessels Colored by Flourosopy',range(0,5,1))
    thal=st.selectbox('Thalium Stress Result',range(1,8,1))



    
    pred=preprocess(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal)

    if st.button("Predict"):    
        if pred[0] == 0:
           st.error('Warning! You have high risk of getting a heart attack!')
    
    else:
        st.success('You have lower risk of getting a heart disease!')
      
    
    
   



    
