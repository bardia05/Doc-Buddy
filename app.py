import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from sklearn.preprocessing import StandardScaler
import os 
from dotenv import load_dotenv
import google.generativeai as gen_ai

## Loading the Model
Diabetese_model = pickle.load(open('D:/ML PROJECTS/Streamlit_app/Final_Combination/Diabetes_model.pkl','rb'))
Heart_model = pickle.load(open('D:/ML PROJECTS/Streamlit_app/Final_Combination/HeartDisease_model.pkl','rb'))
Parkinsons_model = pickle.load(open('D:/ML PROJECTS/Streamlit_app/Final_Combination/Parkinsons_model.pkl','rb'))

## Side bar Navigator 
 #Streamlit paging krre ha 
st.set_page_config(
    page_title= 'Doc-Buddy',
    page_icon= ':syringe:',
    layout= 'centered'
)
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'CareBot'],

                            icons= ['activity','heart','person','robot'],

                            default_index=0,
                        ) # Default Index = 0 means --> jb ap web khologe to 'Diaetes Prediction' dikhega, 1 kroge to heart,2 kroge to parkinsons
    


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = Diabetese_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)



if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = Heart_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = Parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

## Care BOT button
if selected == 'CareBot':
    
## Loading the environment
    load_dotenv()

    GOOGLE_API_KEY = os.getenv('GOOGLE_GEMINI_API')

    # Setting up the GOOGLE GEMINI pro- AI model
    gen_ai.configure(api_key = GOOGLE_API_KEY)
    model = gen_ai.GenerativeModel('gemini-2.0-flash')

    ## Function to translate roles between the Gemini-pro and Streamlit terminilogy
    def translate_rule_for_streamlit(user_role):
        if user_role == 'model':
            return 'Assistant'
        else:
            return user_role
        
    ## Initialize chat session in Stramlit if not already exist
    if 'chat_session' not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])


    ## Displaying Chatbot Title on the Page
    st.title('CareBot 🤖❤️')

    # Display Chat History-->
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_rule_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # input field for user's message
    user_prompt = st.chat_input('Ask CareBot...')
    if user_prompt:
        # Add user's message to chat and display it 
        st.chat_message('user').markdown(user_prompt)

        # Send users message to gemini-pro and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)

        #Displays gemini-pro response
        with st.chat_message('assistant'):
            st.markdown(gemini_response.text)


