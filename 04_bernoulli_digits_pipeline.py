import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer


# ============================================================
# PARTE 1 - CLASSIFICAÇÃO DE DÍGITOS MANUSCRITOS
# ============================================================

# 1. Carregamento do dataset embutido
digitos = load_digits()
X_digitos, y_digitos = digitos.data, digitos.target

print("=== DATASET DE DÍGITOS MANUSCRITOS ===")
print(f"Total de amostras: {X_digitos.shape[0]}")
print(
    "Total de características por amostra (pixels 8x8): "
    f"{X_digitos.shape[1]}"
)
print(f"Classes disponíveis: {digitos.target_names}")

# 2. Visualização de seis imagens
fig, axes = plt.subplots(1, 6, figsize=(10, 2.5))

for i, ax in enumerate(axes):
    ax.imshow(
        digitos.images[i],
        cmap=plt.cm.gray_r,
        interpolation="nearest"
    )
    ax.set_title(f"Dígito: {digitos.target[i]}")
    ax.axis("off")

plt.suptitle(
    "Amostras de Dígitos Manuscritos (Matrizes 8x8)",
    fontsize=11
)
plt.tight_layout()
plt.show()

# 3. Divisão estratificada: 70% treino e 30% teste
X_tr_dig, X_te_dig, y_tr_dig, y_te_dig = train_test_split(
    X_digitos,
    y_digitos,
    test_size=0.30,
    random_state=42,
    stratify=y_digitos
)

# 4. Treinamento com Multinomial Naive Bayes
classificador_digitos = MultinomialNB(alpha=1.0)
classificador_digitos.fit(X_tr_dig, y_tr_dig)

# 5. Avaliação do modelo multiclasse
y_pred_dig = classificador_digitos.predict(X_te_dig)

print("\n=== DESEMPENHO NO DATASET DE DÍGITOS (MULTICLASSE) ===")
print(
    classification_report(
        y_te_dig,
        y_pred_dig,
        labels=list(digitos.target_names),
        digits=4,
        zero_division=0
    )
)


# ============================================================
# PARTE 2 - PIPELINE PARA DETECÇÃO DE SPAM
# ============================================================

# 1. Lista de stop words em português
stop_words_pt = [
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
    "é", "com", "não", "uma", "os", "no", "se", "na", "por",
    "mais", "as", "dos", "como", "mas", "ao", "ele", "das",
    "à", "seu", "sua", "ou", "quando", "muito", "foi"
]

# 2. Base de mensagens: mesma utilizada nos arquivos 02 e 03
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

# 3. Preparação dos dados
df = pd.DataFrame(mensagens, columns=["texto", "rotulo"])

X_spam = df["texto"]
y_spam = df["rotulo"]

X_treino_raw, X_teste_raw, y_treino, y_teste = train_test_split(
    X_spam,
    y_spam,
    test_size=0.30,
    random_state=42,
    stratify=y_spam
)

# 4. Construção do Pipeline: vetorização + classificador
pipeline_spam = Pipeline([
    (
        "vetorizador",
        CountVectorizer(
            lowercase=True,
            stop_words=stop_words_pt
        )
    ),
    (
        "classificador",
        MultinomialNB(alpha=1.0)
    )
])

# 5. Treinamento direto nos textos brutos
pipeline_spam.fit(X_treino_raw, y_treino)

print("\nPipeline treinado com sucesso!")

# 6. Avaliação nos textos de teste
y_pred_pipe = pipeline_spam.predict(X_teste_raw)

print("\n=== AVALIAÇÃO VIA PIPELINE ===")
print(
    classification_report(
        y_teste,
        y_pred_pipe,
        labels=[0, 1],
        target_names=["Ham", "Spam"],
        digits=4,
        zero_division=0
    )
)

# 7. Inferência em novas mensagens
novas_mensagens = [
    "Parabéns! Você ganhou um voucher exclusivo para compras via PIX.",
    "Reunião de alinhamento com a equipe reagendada para amanhã cedo.",
    "Aviso da sua conta bancária: fatura vencida disponível para pagamento imediato."
]

previsoes_pipe = pipeline_spam.predict(novas_mensagens)
probabilidades_pipe = pipeline_spam.predict_proba(novas_mensagens)

# 8. Exibição das classificações e probabilidades
print("-" * 80)

for msg, pred, prob in zip(
    novas_mensagens,
    previsoes_pipe,
    probabilidades_pipe
):
    rotulo = "Spam" if pred == 1 else "Legítimo (Ham)"
    indice_classe = list(pipeline_spam.classes_).index(pred)
    confianca = prob[indice_classe] * 100

    print(f"Texto: '{msg}'")
    print(f"Classificação: {rotulo} (Confiança: {confianca:.2f}%)")
    print("-" * 80)