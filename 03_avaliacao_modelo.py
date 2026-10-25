import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Lista de stop words em português
stop_words_pt = [
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
    "é", "com", "não", "uma", "os", "no", "se", "na", "por",
    "mais", "as", "dos", "como", "mas", "ao", "ele", "das",
    "à", "seu", "sua", "ou", "quando", "muito", "foi"
]

# Base de mensagens: mesma utilizada no arquivo 02
mensagens = [
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

# 1. Preparação dos dados
df = pd.DataFrame(mensagens, columns=["texto", "rotulo"])

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

X_treino_vec = vectorizer.fit_transform(X_treino_raw)
X_teste_vec = vectorizer.transform(X_teste_raw)

# 3. Treinamento do Multinomial Naive Bayes
classificador_nb = MultinomialNB(alpha=1.0)
classificador_nb.fit(X_treino_vec, y_treino)

# 4. Avaliação do Multinomial Naive Bayes
y_pred = classificador_nb.predict(X_teste_vec)

print("=== RELATÓRIO DE CLASSIFICAÇÃO: MULTINOMIAL NAIVE BAYES ===")
print(
    classification_report(
        y_teste,
        y_pred,
        labels=[0, 1],
        target_names=["Ham (0)", "Spam (1)"],
        digits=4,
        zero_division=0
    )
)

# 5. Matriz de confusão
fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay.from_estimator(
    classificador_nb,
    X_teste_vec,
    y_teste,
    labels=[0, 1],
    display_labels=["Ham", "Spam"],
    cmap=plt.cm.Blues,
    ax=ax
)

ax.set_title("Matriz de Confusão - Multinomial Naive Bayes")
plt.tight_layout()
plt.show()

# 6. Treinamento do Bernoulli Naive Bayes
classificador_bernoulli = BernoulliNB(alpha=1.0)
classificador_bernoulli.fit(X_treino_vec, y_treino)

# 7. Avaliação do Bernoulli Naive Bayes
y_pred_bernoulli = classificador_bernoulli.predict(X_teste_vec)

print("\n=== RELATÓRIO DE CLASSIFICAÇÃO: BERNOULLI NAIVE BAYES ===")
print(
    classification_report(
        y_teste,
        y_pred_bernoulli,
        labels=[0, 1],
        target_names=["Ham (0)", "Spam (1)"],
        digits=4,
        zero_division=0
    )
)