import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Amostra de textos para demonstração
textos_exemplo = [
    "Urgente! Seu cartão de crédito foi clonado.",
    "Ganhe um prêmio urgente em dinheiro via PIX.",
    "Olá, confirmação da reunião de amanhã na faculdade."
]

# Lista de stop words em português
stop_words_pt = [
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
    "é", "com", "não", "uma", "os", "no", "se", "na", "por",
    "mais", "as", "dos", "como", "mas", "ao", "ele", "das",
    "à", "seu", "sua", "ou", "quando", "muito", "foi"
]

# Inicializa o vetorizador
vetorizador = CountVectorizer(
    lowercase=True,
    stop_words=stop_words_pt
)

# Aprende o vocabulário e transforma os textos em matriz numérica
X_exemplo_vec = vetorizador.fit_transform(textos_exemplo)

# Exibe o vocabulário e sua quantidade real
vocabulario = vetorizador.get_feature_names_out()

print("Vocabulário aprendido:", vocabulario)
print("Quantidade de termos únicos:", len(vocabulario))
print("Dimensão da matriz (Linhas x Colunas):", X_exemplo_vec.shape)

# Visualização estruturada com Pandas
df_bow_exemplo = pd.DataFrame(
    X_exemplo_vec.toarray(),
    columns=vocabulario,
    index=["Texto 1", "Texto 2", "Texto 3"]
)

print("\nRepresentação tabular do Bag of Words (BoW):")
print(df_bow_exemplo)