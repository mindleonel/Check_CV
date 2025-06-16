import re
import pandas as pd

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

def calcular_match(cv_text: str, requisitos: list):
    if not requisitos:
        return 0.0, [], [], []
    encontrados = [r for r in requisitos if r.lower() in cv_text.lower()]
    faltantes = [r for r in requisitos if r.lower() not in cv_text.lower()]
    score = round((len(encontrados) / len(requisitos)) * 100, 1)
    return score, encontrados, faltantes, requisitos

def exportar_resultados_cv(summaries: list, filename: str = "resultados_cvs", formato: str = "csv"):
    """Exporta os resultados dos currículos para CSV ou XLSX."""
    if not summaries:
        return None

    dados = []
    for resumo in summaries:
        dados.append({
            "Arquivo": resumo.get("nome_arquivo", ""),
            "Email": resumo.get("email", ""),
            "Telefone": resumo.get("telefone", ""),
            "Score": resumo.get("score", ""),
            "Nível": resumo.get("nivel", ""),
            "Compatibilidade": resumo.get("compatibilidade", ""),
            "Justificativa": resumo.get("justificativa", ""),
            "Pontos Fortes": ", ".join(resumo.get("encontrados", [])),
            "Pontos a Melhorar": ", ".join(resumo.get("faltantes", [])),
        })

    df = pd.DataFrame(dados)
    file_path = f"/mnt/data/{filename}.{formato}"

    if formato == "xlsx":
        df.to_excel(file_path, index=False)
    else:
        df.to_csv(file_path, index=False)

    return file_path