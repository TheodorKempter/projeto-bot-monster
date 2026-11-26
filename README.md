# Classificador de Imagens: Monster Energy

Projeto de Visão Computacional desenvolvido de ponta a ponta em Python para classificar imagens em duas categorias: **`monster`** (latas de Monster Energy) e **`nao_monster`**.

O objetivo do projeto é aplicar e documentar todo o fluxo real de um pipeline de Machine Learning — desde a exploração e higienização de imagens brutas até o treinamento com *Transfer Learning* em PyTorch e análise de explicabilidade com **Grad-CAM**.

---

## Tecnologias Utilizadas

* **Python 3**
* **PyTorch & Torchvision:** carregamento de dados, *Data Augmentation* e *Transfer Learning* (ResNet18)
* **Pillow (PIL):** inspeção, conversão de canais e padronização de imagens
* **Scikit-learn:** divisão estratificada (`train_test_split`) e métricas de avaliação
* **PyTorch Grad-CAM:** visualização de mapas de calor (*heatmaps*) para explicabilidade do modelo
* **Matplotlib & NumPy:** visualização de amostras, erros e matrizes

---

## Estrutura do Projeto

```text
projeto-bot-monster/
├── data/
│   ├── raw/            # Imagens originais intocadas (formatos e tamanhos variados)
│   ├── processed/      # Imagens padronizadas (224x224, RGB, .jpg com padding)
│   └── split/          # Divisão estratificada (80% train / 20% test)
├── notebooks/
│   ├── 01_exploracao_imagens.ipynb   # Diagnóstico, pré-processamento e split
│   └── 02_treinamento.ipynb          # Pipeline PyTorch, treino, avaliação e Grad-CAM
├── src/                # Scripts de produção/inferência (em desenvolvimento)
├── requirements.txt    # Dependências do ambiente virtual (venv)
└── README.md
```

---

## Pipeline e Decisões Técnicas

### 1. Exploração e Pré-processamento
A base inicial conta com **56 imagens** (24 `monster` e 32 `nao_monster`). A inspeção inicial revelou extensões misturadas (`.jpg`, `.png`, `.avif`), canais de cor incompatíveis (`RGB` e `RGBA`) e altas resoluções verticais (`1816x4032`, `3000x4000`).

Para padronizar os dados sem sobrescrever os originais:
* Todas as imagens foram convertidas para **RGB** e salvas em **`.jpg`**.
* O redimensionamento para **`224x224`** foi feito utilizando **`ImageOps.pad` (*Letterbox Padding*)** em vez de um *resize* direto, preservando a proporção real das latas sem achatá-las ou cortar objetos próximos às bordas.
* A divisão em **Treino (44 imagens)** e **Teste (12 imagens)** foi feita de forma **estratificada** (`test_size=0.2`, `random_state=42`) para manter o balanceamento entre as classes.

### 2. Modelagem (Transfer Learning)
Devido ao tamanho reduzido da base, optou-se por **Transfer Learning** com a arquitetura **ResNet18** pré-treinada no *ImageNet*:
* **Congelamento da base:** todos os pesos das camadas convolucionais foram congelados (`requires_grad = False`) para reaproveitar os extratores visuais já treinados.
* **Nova camada de classificação:** a camada final (`fc`) foi substituída por uma `nn.Linear(512, 2)`.
* **Data Augmentation:** aplicado exclusivamente no conjunto de treino (`RandomHorizontalFlip` e `RandomRotation(15)`), seguido pela normalização padrão do *ImageNet*.
* **Treinamento:** otimizador `Adam` (`lr=0.001`) e função de perda `CrossEntropyLoss`, convergindo para um *Loss* de treino abaixo de `0.20`.

---

## Resultados (Conjunto de Teste)

Avaliado nas **12 imagens** mantidas isoladas durante todo o treinamento:

* **Acurácia Geral:** `83.3%` (10 acertos de 12)
* **Recall (`monster`):** `100%` (5 de 5 latas identificadas corretamente)
* **Precision (`monster`):** `71%` (2 falsos positivos)

**Matriz de Confusão:**

| Real \ Previsto | `monster` | `nao_monster` | Total Real |
| :--- | :---: | :---: | :---: |
| **`monster`** | **5** | 0 | 5 |
| **`nao_monster`** | 2 | **5** | 7 |

---

## Análise de Erros e Explicabilidade (Grad-CAM)

Para verificar se o modelo estava realmente aprendendo características do produto (e não "colando" pelo fundo da mesa ou iluminação), foi aplicado **Grad-CAM** na última camada convolucional (`layer4[-1]`) sobre os dois falsos positivos do teste:

1. **Desenho do logo do Monster (falso positivo):** o mapa de calor concentrou-se exatamente no centro das três garras verdes, confirmando que a rede aprendeu o padrão visual do logotipo, mesmo fora de uma lata física.
2. **Copo cilíndrico sobre a mesa (falso positivo):** o mapa de calor confirmou que o modelo **ignorou o fundo** (teclado RGB e monitor), mas focou na borda superior e no formato cilíndrico do copo, confundindo-o com uma lata.

---

## Limitações Atuais

* **Tamanho da base de dados:** com apenas 56 imagens (12 no teste), cada acerto ou erro altera a acurácia em ~8,3%, tornando as métricas sensíveis a pequenas variações.
* **Falta de *Hard Negatives*:** o conjunto `nao_monster` ainda possui poucos exemplos visualmente parecidos com o alvo (outras marcas de energéticos, refrigerantes em lata, copos altos e desenhos/logos isolados).
* **Camadas convolucionais congeladas:** sem *Fine-Tuning* nas camadas mais profundas, o modelo depende estritamente das representações genéricas do *ImageNet*.
* **Inferência restrita ao Notebook:** o modelo ainda não está salvo em disco nem encapsulado em um script pronto para classificar fotos novas fora do Jupyter.

---

## Próximos Passos

- [ ] **Expandir o dataset** com mais fotos de latas de Monster em ângulos/ambientes variados e adicionar *hard negatives* (outras latas, copos e logos desenhados) na classe `nao_monster`.
- [ ] **Testar *Fine-Tuning*** descongelando a `layer4` da ResNet18 com uma taxa de aprendizado menor (ex: `lr=1e-4`) para refinar a distinção entre latas e copos.
- [ ] **Persistir o modelo treinado** (`torch.save`) e criar um script de inferência em `src/` para classificar novas imagens via linha de comando.
- [ ] **Adicionar validação cruzada** ou conjunto de validação separado para monitorar *overfitting* época a época.

---

## Como Executar

1. Clone o repositório e crie o ambiente virtual:
   ```powershell
   git clone https://github.com/TheodorKempter/projeto-bot-monster.git
   cd projeto-bot-monster
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Instale as dependências:
   ```powershell
   pip install -r requirements.txt
   ```

3. Execute os notebooks em ordem (`01_exploracao_imagens.ipynb` e `02_treinamento.ipynb`) selecionando o kernel do `venv`.