import os 
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
import PyPDF2  # For reading PDFs
import docx

## Loading the environment
load_dotenv()

#Streamlit paging krre ha 
st.set_page_config(
    page_title= 'CareBot',
    page_icon= ':syringe:',
    layout= 'centered'
)

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
# user_prompt = st.chat_input('Ask CareBot...')
# if user_prompt:
#     # Add user's message to chat and display it 
#     st.chat_message('user').markdown(user_prompt)

#     # Send users message to gemini-pro and get the response
#     gemini_response = st.session_state.chat_session.send_message(user_prompt)

#     #Displays gemini-pro response
#     with st.chat_message('assistant'):
#         st.markdown(gemini_response.text)



# import streamlit as st
# Custom CSS for styling chat input and buttons inline
st.markdown("""
    <style>
        div[data-testid="stChatInput"] {
            display: flex;
            align-items: bottom;
        }
        div[data-testid="stChatInput"] > div:first-child {
            flex-grow: 1; /* Make chat input take most space */
        }
        div[data-testid="stChatInput"] > div {
            margin-left: 1px;
        }
    </style>
""", unsafe_allow_html=True)

# Layout for chat input and buttons
col1, col2, col3, col4 = st.columns([7, 1, 1, 1])  # Adjust proportions as needed

with col1:
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

with col2:
    file_button = st.button("➕", help="Attach a file")

# Handle button clicks
uploaded_file = None
if file_button:
    uploaded_file = st.file_uploader("Upload a file", type=["png", "jpg", "pdf", "docx", "txt"])

# Function to extract text from uploaded file
def extract_text(file):
    file_type = file.type

    if file_type == "text/plain":  # TXT File
        return file.read().decode("utf-8")

    elif file_type == "application/pdf":  # PDF File
        pdf_reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        return text

    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # DOCX File
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    else:
        return "File type not supported for text extraction."

# Extract and display text
if uploaded_file:
    extracted_text = extract_text(uploaded_file)
    st.write("### Extracted Text:")
    st.write(extracted_text)




