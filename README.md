# ST1_capstone
Temporary repository for ST1 capstone code 

Things to note:
1. The data pre processing takes a while because of the large dataset
2. The algorithm comparison and overall-prediction would take around 5 mins or more, depending on the processing speed of your local machine. 
3. Towards the end of all analyses, you will have the option to enter the text of your own choice and predict if the entered text is Positive, negative or neutral. 


Important:
 - 4. On hitting predict, the program will appear to rerun all the analyses. Do not freak out. It is the way it is developed for now and needs further refinements to improve overall runtime. It just takes lonoger because of the Streamlit refresh but it will give you an answer 100%.


External libraries you may need to install on your local computer:
- pip install streamlit pandas profiling
- Then, import pandas_profiling
And lastly do - from streamlit_pandas_profiling import st_profile_report

For preprocessing:
  - !pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force-reinstall
  - And Then use - import preprocess_kgptalkie as ps
 
