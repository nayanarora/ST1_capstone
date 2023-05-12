# Commented out IPython magic to ensure Python compatibility.
#Import Required Python Packages and read data
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno # To visualize missing value
import plotly.graph_objects as go # To Generate Graphs
import plotly.express as px # To Generate box plot for statistical representation
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import missingno as msno
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from plotly.subplots import make_subplots
from sklearn.exceptions import DataDimensionalityWarning
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
import sklearn
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocess_kgptalkie as ps
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")

# Use local CSS for building the Get in touch with me form and to hide the streamlit icon from the bottom of the screen. 
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("/Users/nayanarora/Desktop/ST/capstone/myCap/style.css")

# ---- HEADER SECTION ----
with st.container():
    st.markdown('##')
    st.subheader("Hi, I am Nayan Arora :wave:")
    st.title("This is my Capstone Project for ST1")
    st.write("---")
    st.subheader("This project performs an Exploratory & Predictive Data Analyses on the dataset - Coronavirus Tweets NLP (Text Classification)")
    st.write("The dataset can be found here >> [Coronavirus Tweets NLP - Text Classification](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification)" )


df = pd.read_csv('/Users/nayanarora/Desktop/ST/capstone/myCap/Corona_NLP_train.csv', encoding='latin-1')

# ---- Exploratory data anaylyses ----
with st.container():
    st.write("---")
    st.header("Exploratory Data Analyses (EDA)")
    st.write("##")
    st.write(
            """
            The EDA will help answer the following 5 questions by studying and understanding the Dataset:
            - How machine learning algorithms can be used to predict data?
            - How is a language processing model used for improving user experience?
            - Whether similar datasets are used to train a language processing model like ChatGPT?
            - What kind of data is absolutely necessary for traing a NLP model?
            - How can this data be biased? 
            """
        )

#1. Checking description(first 5 and last 5 rows)
    st.write((df.head())) #first 5 rows of dataframe

    st.write(df.tail()) #last 5 rows

#rows and columns - data shape (attributes and samples)
    st.write(df.shape)

# name of the attributes
    st.write(df.columns)

#unique values for each attribute
    st.write(df.nunique())

#Complete info about data frame
    st.write(df.info())

# Drop duplicates
    st.write("Dataframe after dropping duplicates: ",df.drop_duplicates())
    st.write(df.shape)

#Null values

    null= df.isnull().sum().sort_values(ascending=False)
    total =df.shape[0]
    percent_missing= (df.isnull().sum()/total).sort_values(ascending=False)

    missing_data= pd.concat([null, percent_missing], axis=1, keys=['Total missing', 'Percent missing'])

    missing_data.reset_index(inplace=True)
    missing_data= missing_data.rename(columns= { "index": " column name"})
 
    st.write("Null Values in each column:\n")
    st.write(missing_data)

    st.write("Missing data as white lines")

    st.write((msno.matrix(df,color=(0.3,0.36,0.44))))

    st.write('Total tweets in this data: ')
    st.write(format(df.shape[0]))
    st.write('Total Unique Users in this data:')
    st.write(format(df['UserName'].nunique()))
    st.write('Total Unique Sentiments elements in this data:')
    st.write(df.Sentiment.unique())
    st.write('Total value count for each Unique Sentiment element in this data:')
    st.write(df.Sentiment.value_counts())

# We will copy the text in another column so that the original text is also there for comparison
    st.subheader("To Simplify the analyses process")
    st.write("""
                We will purely focus on the three major class categories - Positive, Negative and Neutral
                - All elements from Extremely Positive and Positive would be combined as one.
                - All elements from Extremely Negative and Negative would be combined as one.
                - All neutral elements would stay as before. 
              """)
    df['text'] = df.OriginalTweet
    df["text"] = df["text"].astype(str)

# Data has 5 classes, let's convert them to 3

    def classes_def(x):
        if x ==  "Extremely Positive":
            return "positive"
        elif x == "Extremely Negative":
            return "negative"
        elif x == "Negative":
            return "negative"
        elif x ==  "Positive":
            return "positive"
        else:
            return "neutral"
    
    df['sentiment']=df['Sentiment'].apply(lambda x:classes_def(x))
    target=df['sentiment']
    st.write('After the simplication, the total distribution (in percentage) for each Unique Sentiment element in this data is as follows: ')
    st.write(df.sentiment.value_counts(normalize= True))


# pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
#restart kernel
#re-run import libraries and data
    st.write("##")
    st.write("##")
    st.write("##")
    st.write("""
                Below is a full analysis report on EDA using pandas profiling. 
                - To be able to run it on your local computer use - pip install streamlit pandas profiling.
                - Then, import pandas_profiling
                - And lastly do - from streamlit_pandas_profiling import st_profile_report 
              """)
    
    profile = ProfileReport(df,title="Twitter tweets NLP EDA",
                        html={'style':{'full_width':True}})
    # profile.to_notebook_iframe()
    st_profile_report(profile)

    st.write("##")
    st.write("##")
    st.write("##")
    st.write("""
                Below is a more in depth data analysis prepared to analyse it for any kind of bias that can make this dataset imperfect for training a model. 
                - For example - more data for positive tweets than negative or neutral. 
                - And other possible interpretations using visualizations. 
                - All this analyses is done on a copied set of data for 'OriginalTweet' as 'text' & 'Sentiment' as 'sentiment'.
              """)
    class_df = df.groupby('Sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)
    st.write(class_df.style.background_gradient())
    
    # df.head()

    percent_class=class_df.text
    labels= class_df.Sentiment
    colors = ['#17C37B','#F92969','#FACA0C', '#cc00ff', '#0066ff']
    my_pie,_,_ = plt.pie(percent_class,radius = 1.0,labels=labels,colors=colors,autopct="%.1f%%")

    plt.setp(my_pie, width=0.6, edgecolor='white')
    class_df = df.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)
    st.write(class_df.style.background_gradient())
    left_column, right_column = st.columns(2)
    with left_column: 
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(plt.show())

    percent_class = class_df.text
    labels=class_df.sentiment

    colors = ['#17C37B','#F92969','#FACA0C']

    my_pie,_,_ = plt.pie(percent_class,radius = 1.0,labels=labels,colors=colors,autopct="%.1f%%")
    plt.setp(my_pie, width=0.6, edgecolor='white') 
    with left_column: 
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(plt.show())
    
    st.write("##")
    st.write("##")
    st.write("##")
    st.write("""
                Comparing the newly Developed training set to the Original set of data               
              """)

    fig=make_subplots(1,2,subplot_titles=('Train set','Original set'))
    x=df.sentiment.value_counts()
    fig.add_trace(go.Bar(x=x.index,y=x.values, marker_color=['#17C37B','#F92969','#FACA0C']),row=1,col=1)
    x=df.Sentiment.value_counts()
    st.write(fig.add_trace(go.Bar(x=x.index,y=x.values, marker_color=['#17C37B','#F92969','#FACA0C', '#cc00ff', '#0066ff']),row=1,col=2))

    with left_column: 
        fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))

        char_len=df[df['sentiment']=="positive"]['text'].str.len()
        ax1.hist(char_len,color='#17C37B')
        ax1.set_title('Positive')

        char_len=df[df['sentiment']=="negative"]['text'].str.len()
        ax2.hist(char_len,color='#F92969')
        ax2.set_title('Negative')

        char_len=df[df['sentiment']=="neutral"]['text'].str.len()
        ax3.hist(char_len,color='#FACA0C')
        ax3.set_title('Neutral')

        fig.suptitle('Character count in tweets')
        st.pyplot(plt.show())

        fig,(p1, p2, p3)=plt.subplots(1,3,figsize=(12,6))

        word_count=df[df['sentiment']=="positive"]['text'].str.split().map(lambda x: len(x))
        p1.hist(word_count,color='#17C37B')
        p1.set_title('Positive words')

        word_count=df[df['sentiment']=="negative"]['text'].str.split().map(lambda x: len(x))
        p2.hist(word_count,color='#F92969')
        p2.set_title('Negative words')

        word_count=df[df['sentiment']=="neutral"]['text'].str.split().map(lambda x: len(x))
        p3.hist(word_count,color='#FACA0C')
        p3.set_title('Neutral words')

        fig.suptitle('Word count for tweets')
        st.pyplot(plt.show())


# ---- Predictive data anaylyses ----
with st.container():
    st.write("---")
    st.header("Predictive Data Analyses (PDA)")
    st.write("##")
    st.write(
            """
            The first step in my Predictive Data Analysis is to clean the data of all special characters, url's, emails and others.
             - This is achieved using a pre-processing library imported in this implementation. 
             - To run on your local computer use !pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force-reinstall
             - And Then use - import preprocess_kgptalkie as ps
            """
        )
   
    #Normalizing and scaling using LabelEncoder
    tweet_data = df.copy()
    le = preprocessing.LabelEncoder()
    UserName = le.fit_transform(list(tweet_data["UserName"]))
    ScreenName = le.fit_transform(list(tweet_data["ScreenName"])) 
    Location = le.fit_transform(list(tweet_data["Location"])) 
    TweetAt = le.fit_transform(list(tweet_data["TweetAt"])) 
    OriginalTweet = le.fit_transform(list(tweet_data["OriginalTweet"])) 
    Sentiment = le.fit_transform(list(tweet_data["Sentiment"])) 


# pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force-reinstall
    st.write("Orignial data - head part")
    st.write(df[["text","sentiment"]])
    st.write("Data is going through the pre-processing algorithm...... It may take a few minutes")

    def get_clean(x):
        x = str(x).lower().replace('\\', '').replace('_', ' ')
        x = ps.cont_exp(x)
        x = ps.remove_emails(x)
        x = ps.remove_urls(x)
        x = ps.remove_html_tags(x)
        x = ps.remove_accented_chars(x)
        x = ps.remove_special_chars(x)
        x = re.sub("(.)\\1{2,}", "\\1", x)
        return x

    df['text']=df['text'].apply(lambda x:get_clean(x))

    st.write("After data pre processing - head part below")
    st.write(df[["text","sentiment"]])


with st.container():
    st.write("---")
    st.header("Model Preparation")
    st.write("##")
    st.write(
            """Process followed:
            - Convert the dataframe to training and validation/test subsets by taking a random sample of 80% of the dataframe and defining it as train subset. 
            - Create the validation/test set by dropping all of the rows that comprise the training set from the dataframe.
            - Create y_train by using using the last column of train (target class). 
            - Create x_train by using all of the columns in train except the last one.
            - The validation set of y_val and x_val or (y_test and x_test), can be created using the same methodology that used to create y_train and X_train.
            """
        )

    tfidf = TfidfVectorizer(max_features = 5000)

    x = df['text']
    y = df['sentiment']
    num_folds = 5
    seed = 7
    scoring = 'accuracy'
    x = tfidf.fit_transform(x)


# Model Test/Train
# Splitting what we are trying to predict into 4 different arrays -
# X train is a section of the x array(attributes) and vise versa for Y(features)
# The test data will test the accuracy of the model created
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)
#splitting 20% of our data into test samples. If we train the model with higher data it already has seen that information and knows
#size of train and test subsets after splitting
    st.write("After the split and TDF Vectorization of x to a maximum of 5000 features, the shape of our training and testing subsets are as below")
    st.write(np.shape(x_train), np.shape(x_test))


    st.write("Predictive analytics model development by comparing different Scikit-learn classification algorithms below:")
# Predictive analytics model development by comparing different Scikit-learn classification algorithms
    models = []
    models.append(('DT', DecisionTreeClassifier()))
    models.append(('SVM', LinearSVC()))
    models.append(('GBM', GradientBoostingClassifier()))
    models.append(('RF', RandomForestClassifier()))
    # evaluate each model in turn
    results = []
    names = []
    st.write("This process will take a few minutes (upto 8mins) to run. ")
    st.write("Performance on Training set: (results are in percent)")

    for name, model in models:
        kfold = KFold(n_splits=num_folds,shuffle=True,random_state=seed)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        msg += '\n'
        st.write(msg)

    left_column,right_column = st.columns(2)
    with left_column:
        st.write("Compare Algorithms Performance using the graph below")
        fig = plt.figure(figsize=(10,6))
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        st.pyplot(plt.show())
    with right_column:
        st.empty()

#Model Evaluation of best performing model, by testing with 
#independent/external test data set. 
# Make predictions on validation/test dataset
    models=[]
    models.append(('DT', DecisionTreeClassifier()))
    models.append(('SVM', LinearSVC()))
    models.append(('GBM', GradientBoostingClassifier()))
    models.append(('RF', RandomForestClassifier()))
    dt = DecisionTreeClassifier()
    nb = GaussianNB()
    gb = GradientBoostingClassifier()
    rf = RandomForestClassifier()
    clf = LinearSVC()
    
    best_model = clf
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    st.write("Best Model Accuracy Score on Test Set:")
    st.write(accuracy_score(y_test, y_pred))

#Model Performance Evaluation Metric 1 - Classification Report
    st.write(classification_report(y_test, y_pred, labels=[1,2,3]))

    left_column,right_column = st.columns(2)
    with left_column:
#Model Performance Evaluation Metric 2
#Confusion matrix 
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        st.pyplot(plt.show())
    with right_column:
        st.empty()

    clf = LinearSVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
# print(classification_report(y_test, y_pred))

#OUR TEST DATA
# haha = "shit. hate an unhealthy political debate"
# #haha = get_clean(haha)
# my_vec = tfidf.transform([haha])
# my_vec.shape
# clf.predict(my_vec)
# #OUR TEST DATA
# haha = "Wow, what a beautiful day"
# #haha = get_clean(haha)
# my_vec = tfidf.transform([haha])
# print(clf.predict(my_vec))
# #OUR TEST DATA
# haha = "how are you doing"
# #haha = get_clean(haha)
# my_vec = tfidf.transform([haha])
# print(clf.predict(my_vec))

    # choice = 'y' 
    # while(choice.islower() != 'n'):       
    # if st.button("Predict"):
        # user_input = st.text_area("Enter anything")
        # user_input = get_clean(user_input)
    
        # st.write(clf.predict(my_vec))

with st.form("my_form"):
    st.write("Inside the form")
    user_input = st.text_input("Enter text you wish to predict")
    user_input = get_clean(user_input)
    my_vec = tfidf.transform([user_input])
    submitted = st.form_submit_button("Predict")
    ans = (clf.predict(my_vec))
    if submitted:
        st.write("Prediction", ans)


with st.container():
    st.write("---")
    st.header("Get In Touch With Me!")
    st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = """
    <form action="https://formsubmit.co/u3249907@uni.canberra.edu.au" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()