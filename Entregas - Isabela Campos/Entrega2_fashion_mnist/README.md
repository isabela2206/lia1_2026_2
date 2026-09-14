# Aula 09 — Classificação de Imagens com Rede Neural Convolucional

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FRAR5Kjrr7sjdhr-Q6KIC_JJ4B2uw4VR)

**Autora:** Isabela
**Disciplina:** Laboratório de Inovação e Automação
**Data de entrega:** 14/09/2026

---

## Sobre esta entrega

Esta pasta contém a adaptação do pipeline de classificação de imagens desenvolvido em aula. O notebook original percorria o ciclo completo de um projeto de IA sobre o dataset **CIFAR-10**; aqui o mesmo pipeline é aplicado ao **Fashion-MNIST**, com o objetivo de demonstrar que a estrutura de um projeto de visão computacional é reaproveitável entre problemas distintos.

```
Dados → Pré-processamento → Modelo → Treinamento → Validação → Teste → Inferência
```

O Fashion-MNIST reúne 70.000 imagens em escala de cinza de 28×28 pixels, distribuídas uniformemente em 10 categorias de vestuário: camiseta, calça, pulôver, vestido, casaco, sandália, camisa, tênis, bolsa e bota.

## Conteúdo da pasta

| Arquivo | Descrição |
|---|---|
| `Aula_09_Adaptado_Fashion_MNIST.ipynb` | Notebook completo, com código, documentação e análise dos resultados |
| `imagens/roupa.jpg` | Imagem de teste para a inferência externa (Seção 8) |
| `README.md` | Este arquivo |

## Como executar

O notebook foi desenvolvido no Google Colab e roda integralmente na CPU, sem necessidade de GPU. O tempo total de execução é de aproximadamente 3 minutos.

1. Abrir o notebook pelo botão **Open in Colab** acima, ou fazer o download e o upload no Colab (`Arquivo → Fazer upload de notebook`).
2. Executar com `Ambiente de execução → Executar tudo`.
3. A Seção 8 é a única que exige ação manual: o upload de uma imagem de peça de roupa para `/content/roupa.jpg`. Um exemplo está disponível na pasta `imagens/`.

Todas as dependências (TensorFlow, NumPy, Matplotlib) já vêm pré-instaladas no ambiente do Colab. O dataset é baixado automaticamente pela API do Keras, sem necessidade de arquivos locais.

## Imagem para a inferência externa (Seção 8)

Para dispensar o upload manual, execute a célula abaixo **antes** da Seção 8. Ela baixa direto para `/content/roupa.jpg` a imagem versionada nesta pasta, que é exatamente a mesma utilizada no desenvolvimento:

```python
!wget -q -O /content/roupa.jpg https://raw.githubusercontent.com/isabela2206/lia1_2026_2/main/Entregas%20-%20Isabela%20Campos/Entrega2_fashion_mnist/roupa.jpg

# Conferir o download
from IPython.display import Image
Image("/content/roupa.jpg", width=200)
```

> ⚠️ O `%20` no endereço corresponde aos espaços do nome da pasta e é obrigatório. O domínio precisa ser `raw.githubusercontent.com` — o link da interface do GitHub (`github.com/.../blob/...`) devolve a página HTML e faria o `load_img` falhar.

**Alternativa sem depender do repositório.** Imagem de licença livre hospedada no Wikimedia Commons:

```python
!wget -q -O /content/roupa.jpg "https://commons.wikimedia.org/wiki/Special:FilePath/T-Shirt_Wikipedia_white.jpg"
```

Crédito: *T-Shirt Wikipedia white.jpg*, Wikimedia Commons, licença [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0).

### Características de uma imagem adequada

O modelo foi treinado sobre imagens padronizadas, e a previsão só é confiável quando a entrada se aproxima dessas condições:

| Requisito | Motivo |
|---|---|
| Peça isolada, sem pessoa vestindo | o dataset contém apenas a peça recortada |
| Centralizada e ocupando quase todo o quadro | o redimensionamento para 28×28 é feito sem recorte inteligente |
| Fundo liso e **claro**, com a peça **escura** | a Seção 8 inverte as intensidades para reproduzir o padrão do dataset (fundo preto, objeto claro) |
| Proporção próxima de quadrada | o `target_size=(28, 28)` distorce imagens muito alongadas |

Uma fotografia de camiseta escura sobre uma folha branca, tirada de cima com o celular, atende a todos os critérios. Caso a peça seja clara sobre fundo escuro, basta comentar a linha `img_array = 1.0 - img_array` na Seção 8.

A discussão sobre por que o desempenho cai em imagens externas está na análise que acompanha a Figura 5 do notebook.

## Arquitetura do modelo

Rede convolucional sequencial com dois blocos de convolução e um classificador denso:

| Camada | Saída | Parâmetros |
|---|---|---|
| `Conv2D(32, 3×3)` + ReLU | 26 × 26 × 32 | 320 |
| `MaxPooling2D(2×2)` | 13 × 13 × 32 | 0 |
| `Conv2D(64, 3×3)` + ReLU | 11 × 11 × 64 | 18.496 |
| `MaxPooling2D(2×2)` | 5 × 5 × 64 | 0 |
| `Flatten` | 1.600 | 0 |
| `Dense(64)` + ReLU | 64 | 102.464 |
| `Dense(10)` + softmax | 10 | 650 |

Treinamento com otimizador Adam, função de perda `sparse_categorical_crossentropy`, 5 épocas e 20% do conjunto de treino reservado para validação.

## Resultados

| Métrica | CIFAR-10 (original) | Fashion-MNIST (esta entrega) |
|---|---|---|
| Acurácia no teste | ≈ 67% | ≈ 90% |
| Parâmetros treináveis | 167.562 | 121.930 |
| Maior confusão | cat ↔ dog | Shirt ↔ T-shirt/top |

A diferença de acurácia não decorre de melhoria no modelo — a arquitetura é a mesma —, mas da dificuldade intrínseca de cada problema. O Fashion-MNIST apresenta peças centralizadas, padronizadas e recortadas sobre fundo uniforme, enquanto o CIFAR-10 reúne fotografias naturais com variação de pose, escala, iluminação e fundo.

Em ambos os datasets, os erros concentram-se em classes que **compartilham a forma global e diferem apenas em detalhes finos**, imperceptíveis na baixa resolução. A análise completa está na Seção 6.2 do notebook.

## Estrutura do notebook

1. Bibliotecas e carregamento dos dados
2. Visualização dos dados
3. Pré-processamento
4. Construção do modelo — decisões de projeto, redução espacial e camada de saída
5. Treinamento e validação
6. Avaliação do modelo — curvas de aprendizado, matriz de confusão e análise dos erros
7. Inferência com uma imagem do conjunto de teste
8. Inferência com uma imagem externa
9. Síntese: o que muda ao trocar de dataset
10. Limitações e próximos passos
11. Referências

## Referências

XIAO, H.; RASUL, K.; VOLLGRAF, R. **Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms.** arXiv:1708.07747, 2017. Disponível em: https://arxiv.org/abs/1708.07747

ZALANDO RESEARCH. **Fashion-MNIST dataset.** Disponível em: https://github.com/zalandoresearch/fashion-mnist

KERAS. **Datasets API — fashion_mnist.** Disponível em: https://keras.io/api/datasets/fashion_mnist/

KRIZHEVSKY, A. **Learning multiple layers of features from tiny images.** Technical Report, University of Toronto, 2009.
