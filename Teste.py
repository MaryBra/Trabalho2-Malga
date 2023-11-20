import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

def print_weights(W1, W2):
    print("Pesos W1:")
    print(W1)
    print("\nPesos W2:")
    print(W2)

def predict(X, W1, W2):
    bias = 1
    Xb = np.insert(X, 0, bias, axis=0)  # Inserir o bias diretamente no vetor de entrada
    o1 = np.tanh(W1.dot(Xb))
    o1b = np.insert(o1, 0, bias)
    Y = np.tanh(W2.dot(o1b))
    return Y


# Carregar o conjunto de dados a partir de um arquivo CSV
# Certifique-se de que o arquivo CSV possui colunas mencionadas abaixo
# Substitua 'seuarquivo.csv' pelo nome do seu arquivo CSV
dataset = pd.read_csv('diabetes.csv')

# Extrair os arrays de atributos e rótulos do conjunto de dados
atributos = dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values
rotulos = dataset['Outcome'].values

# Configurações da rede neural
numEpocas = 100
q = len(rotulos)
eta = 0.01
m = atributos.shape[1]
N = 4
L = 1

# Inicia aleatoriamente as matrizes de pesos.
W1 = np.random.random((N, m + 1))
W2 = np.random.random((L, N + 1))

# Array para armazenar os erros.
E = np.zeros(q)
Etm = np.zeros(numEpocas)

# bias
bias = 1

# Entrada do Perceptron.
X = atributos.T

# Treinamento
for i in range(numEpocas):
    for j in range(q):
        Xb = np.insert(X[:, j], 0, bias)  # Inserir o bias diretamente no vetor de entrada
        o1 = np.tanh(W1.dot(Xb))
        o1b = np.insert(o1, 0, bias)
        Y = np.tanh(W2.dot(o1b))
        e = rotulos[j] - Y
        E[j] = (e.transpose().dot(e))/2

        # Error backpropagation.
        delta2 = np.diag(e).dot((1 - Y*Y))
        vdelta2 = (W2.transpose()).dot(delta2)
        delta1 = np.diag(1 - o1b*o1b).dot(vdelta2)

        # Atualização dos pesos.
        W1 = W1 + eta*(np.outer(delta1[1:], Xb))
        W2 = W2 + eta*(np.outer(delta2, o1b))

    Etm[i] = E.mean()

# Visualização do erro
plt.xlabel("Épocas")
plt.ylabel("Erro Médio")
plt.plot(Etm, color='b')
plt.show()

# Avaliação da rede no conjunto de teste
# Vamos pegar o primeiro exemplo do conjunto de dados para testar a previsão
for i in range(len(rotulos)):
    exemplo_teste = atributos[i, :]
    previsao = predict(exemplo_teste, W1, W2)
    print(f"Previsão para o exemplo {i + 1}: {previsao}")

# Pergunta relacionada ao dataset
# Exemplo: Qual é a média de idade das pessoas no conjunto de dados?
media_idade = np.mean(atributos[:, -1])
print(f"Média de idade no conjunto de dados: {media_idade}")
