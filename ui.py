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
        with st.expander(f"📁 {s['nome_arquivo']}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**📧 Email:** `{s['email']}`")
                st.markdown(f"**📱 Telefone:** `{s['telefone']}`")
            with col2:
                skills = s['skills_detectadas']
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
        score = round(avaliacao.get("score", 0.0), 1)
        nivel = avaliacao.get("nivel", "Desconhecido")
        compat = avaliacao.get("compatibilidade", "Não")
        justificativa = avaliacao.get("justificativa", "Sem justificativa fornecida.")
        cor = score_color(score)
        status = style_match_score(score)

        # Match de palavras-chave
        _, encontrados, faltantes, todos = calcular_match_fn(texto, requisitos_list)

        with st.expander(f"📂 {resumo['nome_arquivo']} — 🎯 Match: {score}%", expanded=False):
            st.markdown(f"**📧 Email:** {resumo['email']}")
            st.markdown(f"**📞 Telefone:** {resumo['telefone']}")
            st.markdown(f"**🎯 Score de compatibilidade (IA):** `{score}%`")
            st.progress(score / 100)

            st.markdown(f"**🧠 Nível de experiência:** {nivel}")
            compat_text = "✅ O candidato é compatível com a vaga." if compat.lower() == "sim" else "❌ O candidato não é compatível com a vaga."
            st.markdown(f"**{compat_text}**")
            st.markdown(f"**📌 Justificativa:** {justificativa}")

            st.markdown("#### ✅ Pontos Fortes")
            if encontrados:
                st.markdown(", ".join(encontrados))
            else:
                st.markdown("*Nenhum requisito atendido*")

            st.markdown("#### ⚠️ Pontos a Melhorar")
            if faltantes:
                st.markdown(", ".join(faltantes))
            else:
                st.markdown("*Todos os requisitos foram atendidos!*")

            st.markdown("#### 📋 Comparativo de Requisitos")
            comparativo = [
                {"Requisito": r, "Status": "✔️ Encontrado" if r in encontrados else "❌ Não encontrado"}
                for r in todos
            ]
            st.dataframe(comparativo, use_container_width=True)