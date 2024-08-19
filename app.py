import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from PIL import Image




load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))




def get_conversational_chain():
    """Initialize the chain for the QA
    Returns:
        langchain.chains.combine_documents.stuff.StuffDocumentsChain: Chain for QA
    """
    
    prompt_template = """
    provide a proper linkedin post content for the provided Senario, make sure to maintain the professionalism in the post, 
    Use the Context as a guide,Just provide the post content as text. Do not add any other formattings.If the Senario is not appropriate for a LinkedIn post inform the user that it is better to not use it.
    don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Senario: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.0)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)


    return chain


def user_input(user_question):
    """Provide answers for the user questions
    Args:
        user_question (String): User's question 
    """
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    with st.chat_message("assistant"):
        st.write(response["output_text"])
        
    st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})


def main():
    """Main function to initialize the Streamlit"""

    summerize = False

    st.set_page_config("PDFQuery")
    st.markdown(
        """
        <style>
        .title {
            text-align: center;
        }
        button[title="View fullscreen"]{
            visibility: hidden;}
        .e1nzilvr1 {display: none}
        </style>
        """,
        unsafe_allow_html=True
    )
    

    
    image = Image.open('Linked in Sight.png')
    with st.columns(3)[1]:
        # st.header("")
        st.image(image)
        st.markdown(
            """
            <div style="text-align: center; margin-bottom:20%">
            <small style="color: white;text-align: center;"> 
                Turn your ideas into impactful LinkedIn posts with AI precision.
            <br>
                
            </small>
            """,
            unsafe_allow_html=True
        )
        
        

    # st.markdown("<h1 class='title'>PDFQuery<h1>",unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a Question Related to Uploaded Files")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})
        user_input(user_question)        

    # with st.sidebar:
        
        # title_container = st.container()
        # col1, col2,col3 = st.columns([4,9,8])
        # with title_container:
        #     with col1:
        #         st.image(image, width=48)
        #     with col2:
        #         st.markdown('<h1 style="color: white;padding:0;margin-top:10px;"> PDF Query </h1>',
        #                     unsafe_allow_html=True)
            # st.image(image)
        
        # st.title("Doc Genie")
        # pdf_docs = st.file_uploader("", accept_multiple_files=True)
        # if st.button("Submit & Process"):
        #     with st.spinner("Processing..."):
        #         raw_text = get_pdf_text(pdf_docs)
        #         text_chunks = get_text_chunks(raw_text)
        #         get_vector_store(text_chunks)
        #         st.success("Done")
        #         summerize = True
    
    # if summerize:
    #     user_input("Summerize the context")
                



if __name__ == "__main__":
    main()