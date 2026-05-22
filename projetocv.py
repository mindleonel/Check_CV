# Importação principal do Streamlit
import streamlit as st

# Configuração inicial da página
st.set_page_config(page_title="LynxTalent", page_icon="")

# Carregamento do ficheiro CSS personalizado
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

import utils
from ui import render_cv_summary, render_match_results
from utils import calcular_match, extract_summary_from_cv
import os, time, tempfile, hashlib, re
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Função para extrair resumo básico de um currículo
def extract_summary_from_cv(text: str):
    summary = {
        "nome": "",
        "email": "",
        "telefone": "",
        "skills_detectadas": [],
        "texto_cv": text
    }
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    tel_match = re.search(r"(\+?\d{1,3})?[\s-]?\(?\d{2,3}\)?[\s-]?\d{4,5}[\s-]?\d{4}", text)
    summary["email"] = email_match.group(0) if email_match else "Não identificado"
    summary["telefone"] = tel_match.group(0) if tel_match else "Não identificado"
    skills_keywords = ["Python", "SQL", "Docker", "AWS", "React", "Node", "Linux", "Kubernetes"]
    summary["skills_detectadas"] = [s for s in skills_keywords if s.lower() in text.lower()]
    return summary

# Função para calcular a compatibilidade entre o currículo e os requisitos
def calcular_match(cv_text: str, requisitos: list):
    if not requisitos:
        return 0.0, [], [], []
    encontrados = []
    faltantes = []
    for r in requisitos:
        if r.lower() in cv_text.lower():
            encontrados.append(r)
        else:
            faltantes.append(r)
    score = round((len(encontrados) / len(requisitos)) * 100, 1)
    return score, encontrados, faltantes, requisitos
# Função para gerar um hash dos ficheiros enviados
def get_uploads_hash(uploads):
    hasher = hashlib.md5()
    for file in uploads:
        hasher.update(file.getvalue())
    return hasher.hexdigest()

# Função para processar PDFs enviados e configurar o retriever
def config_retriever(uploads):
    docs = []
    summaries = []
    if not uploads:
        st.warning("Nenhum arquivo foi enviado.")
        return None
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploads:
        try:
            file_path = os.path.join(temp_dir.name, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            if not loaded_docs:
                st.warning(f"O arquivo '{file.name}' não contém texto válido.")
                continue
            docs.extend(loaded_docs)
            texto_cv = " ".join([doc.page_content for doc in loaded_docs])
            resumo = extract_summary_from_cv(texto_cv)
            resumo["nome_arquivo"] = file.name
            resumo["texto_cv"] = texto_cv
            summaries.append(resumo)
        except Exception as e:
            st.error(f"Erro ao processar '{file.name}': {e}")
    if not docs:
        st.error("Nenhum documento válido foi processado.")
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    st.session_state.cv_summaries = summaries
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4})

# Função para configurar o modelo Groq
def model_groq(model="llama3-70b-8192", temperature=0.1):
    return ChatGroq(model=model, temperature=temperature, max_tokens=None)
# Função para configurar a cadeia RAG
def config_rag_chain(model_class, retriever):
    llm = model_groq()
    prompt_template = PromptTemplate.from_template(
        "És um assistente que analisa currículos. Pergunta: {input} \nContexto: {context}"
    )
    qa_chain = create_stuff_documents_chain(llm, prompt_template)
    history_aware = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=ChatPromptTemplate.from_messages([
            ("system", "Reformule a pergunta com contexto"),
            MessagesPlaceholder("chat_history"),
            ("human", "Question: {input}")
        ])
    )
    return create_retrieval_chain(history_aware, qa_chain)
# Função para avaliar o perfil do candidato usando IA
def avaliar_perfil_com_ia(llm, texto_cv, requisitos):
    prompt = f"""
Você é um assistente de RH especialista em análise de currículos.
Requisitos da vaga:
{', '.join(requisitos)}
Currículo do candidato:
{texto_cv[:3000]}
Analise o currículo considerando os requisitos e responda:
1. Score de compatibilidade com os requisitos (0 a 100).
2. Nível de experiência (Básico, Intermediário ou Avançado).
3. Se o candidato é compatível com a vaga (Sim ou Não).
4. Justifique brevemente a avaliação.

Responda em JSON com as chaves:
"score", "nivel", "compatibilidade", "justificativa".
"""
    resposta = llm.invoke(prompt)
    import json
    try:
        result = json.loads(resposta.content if hasattr(resposta, "content") else resposta)
        return result
    except Exception:
        return {
            "score": 0.0,
            "nivel": "Desconhecido",
            "compatibilidade": "Não",
            "justificativa": "A IA não conseguiu processar o currículo corretamente."
        }
# Carregamento das variáveis de ambiente
load_dotenv()
# Inicialização da lista de resumos dos currículos
if "cv_summaries" not in st.session_state:
    st.session_state.cv_summaries = []
# Inicialização do histórico do chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá, sou o seu assistente virtual! Como posso ajudar?")
    ]
# Inicialização da lista de documentos
if "docs_list" not in st.session_state:
    st.session_state.docs_list = None

# Inicialização do retriever
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Configuração da barra lateral
with st.sidebar:
    st.header("Currículos & Vaga")

    uploads = st.file_uploader(
        "Enviar Currículos (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("---")

    st.subheader("Requisitos da Vaga")

    areas_tecnicas = [
        "Desenvolvimento Backend",
        "Desenvolvimento Frontend",
        "DevOps",
        "Segurança da Informação",
        "Data Science",
        "Machine Learning",
        "Administração de Redes",
        "Administração de Sistemas",
        "Cloud Computing",
        "Suporte Técnico",
        "UI/UX Design",
        "QA/Testes",
        "Banco de Dados",
        "Gestão de Projetos",
        "Engenharia de Dados",
        "Arquitetura de Software"
    ]
    requisitos_selecionados = st.multiselect("Áreas técnicas desejadas", areas_tecnicas)
    habilidades_digitadas = st.text_input(
        "Habilidades específicas (opcional)",
        placeholder="Ex: Python, SQL"
    )
    requisitos_texto = [h.strip() for h in habilidades_digitadas.split(",") if h.strip()]
    requisitos_list = requisitos_selecionados + requisitos_texto
    executar_match = st.button("Verificar compatibilidade")

# Verificação inicial de envio de currículos
if not uploads:
    st.info("Envie ao menos 1 currículo PDF.")
    st.stop()

# Execução da análise de compatibilidade
if executar_match:
    if not requisitos_list:
        st.warning("Selecione pelo menos uma área técnica ou digite uma habilidade.")
        st.stop()
    current_hash = get_uploads_hash(uploads)
    if st.session_state.docs_list != current_hash:
        st.session_state.docs_list = current_hash
        retriever = config_retriever(uploads)
        if retriever is None:
            st.stop()
        st.session_state.retriever = retriever
    cv_summaries = st.session_state.get("cv_summaries", [])
    if not cv_summaries:
        st.warning("Nenhum currículo foi processado.")
        st.stop()
    st.subheader("Relatório de Compatibilidade com a Vaga")
    # Avaliação individual de cada currículo
    for resumo in cv_summaries:
        texto = resumo.get("texto_cv", "")
        avaliacao_ia = avaliar_perfil_com_ia(model_groq(), texto, requisitos_list)
        score = round(avaliacao_ia.get("score", 0.0), 1)
        nivel_raw = avaliacao_ia.get("nivel")
        compat_raw = avaliacao_ia.get("compatibilidade")
        just_raw = avaliacao_ia.get("justificativa")
        nivel = f"Nível de experiência: {nivel_raw}" if nivel_raw else "Nível de experiência: não identificado com clareza."
        compatibilidade = (
            "O candidato é compatível com a vaga."
            if compat_raw and compat_raw.lower() == "sim"
            else "O candidato não é compatível com a vaga."
        )
        justificativa = (
            just_raw
            if just_raw
            else "A IA não conseguiu justificar com detalhes. Verifique o CV manualmente."
        )
        score_bar = score / 100
        _, encontrados, faltantes, todos = calcular_match(texto, requisitos_list)
        with st.expander(f"{resumo['nome_arquivo']} — Match: {score}%"):
            st.markdown(f"**Email:** {resumo['email']}")
            st.markdown(f"**Telefone:** {resumo['telefone']}")
            st.markdown(f"**Score de compatibilidade (IA):** `{score}%`")
            st.progress(score_bar)
            st.markdown(f"**{nivel}**")
            st.markdown(f"**{compatibilidade}**")
            st.markdown(f"**Justificativa:** {justificativa}")
            st.markdown("#### Pontos Fortes")
            if encontrados:
                st.markdown(", ".join(encontrados))
            else:
                st.markdown("*Nenhum requisito atendido*")
            st.markdown("#### Pontos a Melhorar")
            if faltantes:
                st.markdown(", ".join(faltantes))
            else:
                st.markdown("*Todos os requisitos foram atendidos!*")
            st.markdown("#### Comparativo de Requisitos")
            comparativo = [
                {
                    "Requisito": r,
                    "Status": "Encontrado" if r in encontrados else "Não encontrado"
                }
                for r in todos
            ]
            st.dataframe(comparativo, use_container_width=True)
# Entrada de mensagens do chat
user_query = st.chat_input("Digite sua mensagem aqui...")
# Processamento das mensagens do chat
if user_query and uploads:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)
    with st.chat_message("AI"):
        current_hash = get_uploads_hash(uploads)
        if st.session_state.docs_list != current_hash:
            st.session_state.docs_list = current_hash
            st.session_state.retriever = config_retriever(uploads)
        retriever = st.session_state.retriever
        if retriever is None:
            st.error("Retriever não foi inicializado.")
            st.stop()
        rag_chain = config_rag_chain(model_class="groq", retriever=retriever)
        result = rag_chain.invoke({
            "input": user_query,
            "chat_history": st.session_state.chat_history
        })
        resposta = result["answer"]
        st.write(resposta)
        for idx, doc in enumerate(result["context"]):
            file = os.path.basename(doc.metadata["source"])
            page = doc.metadata.get("page", "?")
            with st.popover(f"Fonte {idx} — {file} p. {page}"):
                st.caption(doc.page_content)
        st.session_state.chat_history.append(AIMessage(content=resposta))

# Visualização do histórico do chat
if st.session_state.chat_history:
    with st.expander("Histórico do chat"):
        for msg in st.session_state.chat_history:
            role = "Assistente" if isinstance(msg, AIMessage) else "Humano"
            st.markdown(f"**{role}:** {msg.content}")
# Tempo de execução
start = time.time()
end = time.time()

print("Tempo total de execução:", round(end - start, 2), "segundos")
