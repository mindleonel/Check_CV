import streamlit as st

def style_match_score(score):
    if score >= 90:
        return "✅ Compatível (Excelente)"
    elif score >= 70:
        return "🟡 Compatível (Intermediário)"
    else:
        return "❌ Não Compatível"

def score_color(score):
    if score >= 90:
        return "#28a745"  # Verde
    elif score >= 70:
        return "#ffc107"  # Amarelo
    else:
        return "#dc3545"  # Vermelho

def render_cv_summary(summaries):
    st.markdown("### 📄 Resumo dos Currículos")
    for s in summaries:
        with st.expander(f"📁 {s.get('nome_arquivo', 'Sem nome')}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**📧 Email:** `{s.get('email', 'Não informado')}`")
                st.markdown(f"**📱 Telefone:** `{s.get('telefone', 'Não informado')}`")
            with col2:
                skills = s.get('skills_detectadas', [])
                st.markdown("**🧠 Skills Detectadas:**")
                if skills:
                    st.markdown("✅ " + " | ".join(skills))
                else:
                    st.markdown("_Nenhuma detectada_")

def render_match_results(cv_summaries, requisitos_list, calcular_match_fn, avaliar_perfil_fn, model_fn):
    st.markdown("## 🧮 Resultados da Análise de Compatibilidade")

    for resumo in cv_summaries:
        texto = resumo.get("texto_cv", "")
        avaliacao = avaliar_perfil_fn(model_fn(), texto, requisitos_list)
        resumo["score"] = round(avaliacao.get("score", 0.0), 1)
        resumo["nivel"] = avaliacao.get("nivel", "Desconhecido")
        resumo["compatibilidade"] = avaliacao.get("compatibilidade", "Não")
        resumo["justificativa"] = avaliacao.get("justificativa", "Sem justificativa fornecida.")
        score = round(avaliacao.get("score", 0.0), 1)
        nivel = avaliacao.get("nivel", "Desconhecido")
        compat = avaliacao.get("compatibilidade", "Não")
        justificativa = avaliacao.get("justificativa", "Sem justificativa fornecida.")
        cor = score_color(score)
        status = style_match_score(score)

        _, encontrados, faltantes, todos = calcular_match_fn(texto, requisitos_list)

        with st.expander(f"📂 {resumo.get('nome_arquivo', 'Sem nome')} — 🎯 Match: {score}%", expanded=False):
            st.markdown(f"**📧 Email:** {resumo.get('email', '-')}")
            st.markdown(f"**📞 Telefone:** {resumo.get('telefone', '-')}")
            st.markdown(f"**🎯 Score de compatibilidade (IA):** `{score}%`")
            st.progress(score / 100)

            st.markdown(f"**🧠 Nível de experiência:** {nivel}")
            compat_text = "✅ O candidato é compatível com a vaga." if compat.lower() == "sim" else "❌ O candidato não é compatível com a vaga."
            st.markdown(f"**{compat_text}**")
            st.markdown(f"**📌 Justificativa:** {justificativa}")

            st.markdown("#### ✅ Pontos Fortes")
            st.markdown(", ".join(encontrados) if encontrados else "*Nenhum requisito atendido*")

            st.markdown("#### ⚠️ Pontos a Melhorar")
            st.markdown(", ".join(faltantes) if faltantes else "*Todos os requisitos foram atendidos!*")

            st.markdown("#### 📋 Comparativo de Requisitos")
            comparativo = [{"Requisito": r, "Status": "✔️ Encontrado" if r in encontrados else "❌ Não encontrado"} for r in todos]
            st.dataframe(comparativo, use_container_width=True)

    # Botão de exportação ao final
    # Ao final da função
    st.markdown("---")
    if st.button("🧮 Gerar Relatório", key="btn_gerar"):
        from utils import exportar_resultados_cv
        caminho = exportar_resultados_cv(cv_summaries, formato="xlsx")
        if caminho:
            st.success("✅ Relatório gerado com sucesso!")
            with open(caminho, "rb") as f:
                st.download_button(
                    "📥 Baixar Relatório Excel",
                    data=f,
                    file_name="lynxtalent_resultados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="btn_download"
                )
