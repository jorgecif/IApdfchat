import os
import streamlit as st

from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    ImageCaptionLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document

import pytube


# -------------------------
# Configuración OpenAI
# -------------------------

OPENAI_API_KEY = st.secrets["openai_api_key"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Interfaz
# -------------------------

st.header("Sube tu archivo y haz tus preguntas")
st.subheader(
    "Tipos de archivo soportados: PDF / DOCX / TXT / JPG / PNG / YouTube"
)

# -------------------------
# Modelo LLM
# -------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True,
)

# -------------------------
# Funciones auxiliares
# -------------------------


def load_version_history():
    try:
        with open("version_history.txt", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "No version history available."


# -------------------------
# Sidebar
# -------------------------

with st.sidebar:

    uploaded_files = st.file_uploader(
        "Please upload your files",
        accept_multiple_files=True,
    )

    youtube_url = st.text_input("YouTube URL")

    with st.expander("Version History"):
        st.write(load_version_history())

    st.info(
        "Refresh the browser if you want to start a completely new session.",
        icon="ℹ️",
    )

# -------------------------
# Procesamiento
# -------------------------

if uploaded_files or youtube_url:

    st.write(
        f"Number of files uploaded: "
        f"{len(uploaded_files) if uploaded_files else 0}"
    )

    if "processed_data" not in st.session_state:

        documents = []

        # -------------------------
        # Archivos cargados
        # -------------------------

        if uploaded_files:

            for uploaded_file in uploaded_files:

                file_path = os.path.join(
                    os.getcwd(),
                    uploaded_file.name,
                )

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                try:

                    if file_path.lower().endswith(
                        (".png", ".jpg", ".jpeg")
                    ):

                        image_loader = ImageCaptionLoader(
                            path_images=[file_path]
                        )

                        image_documents = image_loader.load()

                        documents.extend(image_documents)

                    elif file_path.lower().endswith(
                        (".pdf", ".docx", ".txt")
                    ):

                        loader = UnstructuredFileLoader(file_path)

                        loaded_documents = loader.load()

                        documents.extend(loaded_documents)

                except Exception as e:
                    st.error(
                        f"Error procesando {uploaded_file.name}: {str(e)}"
                    )

        # -------------------------
        # YouTube
        # -------------------------

        if youtube_url:

            try:

                youtube_video = pytube.YouTube(youtube_url)

                stream = (
                    youtube_video.streams
                    .filter(only_audio=True)
                    .first()
                )

                stream.download(
                    filename="youtube_audio.mp4"
                )

                with open(
                    "youtube_audio.mp4",
                    "rb"
                ) as audio_file:

                    transcript = (
                        client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                        )
                    )

                youtube_document = Document(
                    page_content=transcript.text,
                    metadata={"source": youtube_url},
                )

                documents.append(youtube_document)

            except Exception as e:
                st.error(
                    f"Error procesando YouTube: {str(e)}"
                )

        # -------------------------
        # Validación
        # -------------------------

        if not documents:
            st.warning(
                "No se pudo extraer contenido de los archivos."
            )
            st.stop()

        # -------------------------
        # Chunking
        # -------------------------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
        )

        document_chunks = text_splitter.split_documents(
            documents
        )

        embeddings = OpenAIEmbeddings()

        vectorstore = Chroma.from_documents(
            document_chunks,
            embeddings,
        )

        st.session_state.processed_data = {
            "document_chunks": document_chunks,
            "vectorstore": vectorstore,
        }

    else:

        document_chunks = (
            st.session_state.processed_data[
                "document_chunks"
            ]
        )

        vectorstore = (
            st.session_state.processed_data[
                "vectorstore"
            ]
        )

    # -------------------------
    # QA Chain
    # -------------------------

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # -------------------------
    # Chat
    # -------------------------

    if prompt := st.chat_input("Haz tu pregunta"):

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        result = qa(
            {
                "question": prompt,
                "chat_history": st.session_state.chat_history,
            }
        )

        answer = result["answer"]

        st.session_state.chat_history.append(
            (
                prompt,
                answer,
            )
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

else:

    st.write(
        "Carga archivos o proporciona una URL de YouTube."
    )
