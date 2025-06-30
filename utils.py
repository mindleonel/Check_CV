import re
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings

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

    skills_keywords = [
        "Python", "SQL", "Docker", "AWS", "React", "Node",
        "Linux", "Kubernetes"
    ]
    summary["skills_detectadas"] = [s for s in skills_keywords if s.lower() in text.lower()]
    return summary

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

def exportar_resultados_cv(cv_summaries, formato="xlsx"):
    if not cv_summaries:
        return None

    df = pd.DataFrame(cv_summaries)
    caminho = f"lynxtalent_resultados.{formato}"

    if formato == "xlsx":
        df.to_excel(caminho, index=False)
    elif formato == "csv":
        df.to_csv(caminho, index=False)
    else:
        raise ValueError("Formato não suportado. Use 'xlsx' ou 'csv'.")

    return caminho

def get_embeddings_model():
    # Troque aqui o modelo, se desejar usar outro compatível (ex: BAAI/bge-m3)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")