import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Lista de stop words em português
stop_words_pt = [
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
    "é", "com", "não", "uma", "os", "no", "se", "na", "por",
    "mais", "as", "dos", "como", "mas", "ao", "ele", "das",
    "à", "seu", "sua", "ou", "quando", "muito", "foi"
]

# Base de mensagens
mensagens = [
    # Mensagens legítimas (Ham - Classe 0)
    ("Olá João, tudo bem? Podemos marcar nossa reunião do projeto amanhã?", 0),
    ("Segue em anexo o relatório final da disciplina de banco de dados.", 0),
    ("Lembrete: a aula de aprendizado de máquina começa às 19h no laboratório.", 0),
    ("Você conseguiu rodar o script no ambiente virtual com uv?", 0),
    ("Favor confirmar presença na conferência de tecnologia até quinta.", 0),
    ("Professor, segue a entrega da atividade de programação em Python.", 0),
    ("Almoço confirmado hoje ao meio-dia na praça de alimentação?", 0),
    ("Atualização de segurança realizada com sucesso em seu servidor Linux.", 0),
    ("Preciso de ajuda com a correção dos testes de integração no backend.", 0),
    ("Bom dia! Poderia revisar a documentação da API antes do deploy?", 0),
    ("Seu pacote dos Correios chegou na portaria do condomínio.", 0),
    ("Reunião de alinhamento da squad reagendada para sexta-feira.", 0),

    # Mensagens fraudulentas (Spam - Classe 1)
    ("URGENTE! Seu cartão de crédito foi clonado. Clique no link para desbloquear agora.", 1),
    ("Parabéns! Você ganhou um prêmio de 50 mil reais no sorteio do PIX. Resgate já!", 1),
    ("Aviso bancário: sua conta possui pendência crítica. Acesse o link seguro imediatamente.", 1),
    ("Empréstimo pré-aprovado sem consulta ao SPC e Serasa com taxa zero. Clique aqui.", 1),
    ("Você foi selecionado para receber uma recompensa exclusiva. Cadastre seu cartão.", 1),
    ("Oferta imperdível! Compre o produto com 90 por cento de desconto só hoje.", 1),
    ("ATENÇÃO: Sua chave PIX precisa de recadastramento urgente para evitar bloqueio.", 1),
    ("Trabalhe de casa e ganhe 5 mil reais por semana sem experiência. Saiba mais.", 1),
    ("Ganhe seguidores e curtidas ilimitadas no seu perfil agora mesmo de graça.", 1),
    ("Último aviso! Seu saldo promocional expira em 1 hora. Clique para resgatar.", 1),
    ("Sua fatura do cartão está atrasada. Pague via PIX com desconto especial no link.", 1),
    ("Parabéns cliente! Seu número foi premiado na promoção anual da loja.", 1)
]

# Criação do DataFrame
df = pd.DataFrame(mensagens, columns=["texto", "rotulo"])

print(f"Total de mensagens: {len(df)}")
print(f"Distribuição das classes:\n{df['rotulo'].value_counts()}")

# 1. Divisão estratificada: 70% treino e 30% teste
X_treino_raw, X_teste_raw, y_treino, y_teste = train_test_split(
    df["texto"],
    df["rotulo"],
    test_size=0.30,
    random_state=42,
    stratify=df["rotulo"]
)

# 2. Vetorização
vectorizer = CountVectorizer(
    lowercase=True,
    stop_words=stop_words_pt
)

# Treino: aprende o vocabulário
X_treino_vec = vectorizer.fit_transform(X_treino_raw)

# Teste: utiliza o vocabulário já aprendido
X_teste_vec = vectorizer.transform(X_teste_raw)

print(f"Dimensão do treino: {X_treino_vec.shape}")
print(f"Dimensão do teste: {X_teste_vec.shape}")
print(
    "Total de termos no vocabulário: "
    f"{len(vectorizer.get_feature_names_out())}"
)

# 3. Treinamento do Naive Bayes Multinomial
classificador_nb = MultinomialNB(alpha=1.0)
classificador_nb.fit(X_treino_vec, y_treino)

print("\nClassificador Naive Bayes treinado com sucesso!")
print(
    f"Log Prior das Classes (ln P(y)): "
    f"{classificador_nb.class_log_prior_}"
)

# 4. Classificação de novas mensagens
mensagens_novas = [
    "Parabéns! Clique no link e resgate seu prêmio em dinheiro agora via PIX.",
    "Boa tarde, professor. Gostaria de tirar uma dúvida sobre a entrega da atividade.",
    "Aviso importante da segurança do cartão: transferência suspeita identificada.",
    "Pessoal, vamos nos encontrar no laboratório às 14h para finalizar o backend."
]

# Vetorização das novas mensagens
X_novas_vec = vectorizer.transform(mensagens_novas)

# Predições e probabilidades
previsoes = classificador_nb.predict(X_novas_vec)
probabilidades = classificador_nb.predict_proba(X_novas_vec)

nomes_classes = ["Legítimo (Ham)", "Spam"]

print("-" * 80)

for msg, pred, prob in zip(
    mensagens_novas,
    previsoes,
    probabilidades
):
    prob_ham = prob[0] * 100
    prob_spam = prob[1] * 100

    print(f'Mensagem: "{msg}"')
    print(f"Classificação: {nomes_classes[pred]}")
    print(
        f"Probabilidades: Ham = {prob_ham:.2f}% | "
        f"Spam = {prob_spam:.2f}%"
    )
    print("-" * 80)