import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplate import css, bot_template, user_template
from langchain.llms import HuggingFaceHub



#extracting text from pdf(s)
def get_pdf_text(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


#extracting chunks from text
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator= "\n", chunk_size = 1000, chunk_overlap = 200, length_function = len)
    chunks = text_splitter.split_text(raw_text)
    return chunks


#vectorization of text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore


#convo chain / memory for followup ques
def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI(openai_api_key = "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")
    llm = HuggingFaceHub(huggingfacehub_api_token = "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", repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain


#handling user ques
def handle_userinput(user_ques):
    response = st.session_state.conversation({"question": user_ques})
    # st.write(response)
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html = True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html = True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Multiple PDF(s) ChatBot", page_icon=":books:")

    st.write(css, unsafe_allow_html = True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Multiple PDF(s) ChatBot 😃")
    st.write("This llm based chatbot uses instructor embedding from huggingface, FAISS based local vector storage. And with Langchain at its base. Using Streamlit for suave & polished ui.")
    user_ques = st.text_input("Ask a question about your pdf(s):")
    
    if user_ques:
        handle_userinput(user_ques)





    with st.sidebar:
        st.subheader("Your Files")
        pdf_doc = st.file_uploader("Upload Pdf(s) here and click Proceed", accept_multiple_files = True)
        if st.button("Proceed"):
            with st.spinner("Processing"):
                #pdf text
                raw_text = get_pdf_text(pdf_doc)
                # st.write(raw_text)

                #text chunk
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                #vector store
                vectorstore = get_vectorstore(text_chunks)

                #convo chain/ memory
                st.session_state.conversation = get_conversation_chain(vectorstore)





if __name__ == '__main__':
    main()
