# Dataset para Classificação de Sons Emergenciais em Dispositivos Vestíveis para Deficientes Auditivos
*(Dataset for Classification of Emergency Sounds in Wearable Devices for the Hearing Impaired)*

Trabalho de Conclusão de Curso (TCC) em Engenharia de Computação pela Universidade Federal do Recôncavo da Bahia (UFRB).

**Autor:** Gabriel Sampaio de Oliveira, 
**Orientador:** Prof. Dr. Igor Dantas dos Santos Miranda

---

## 📝 Resumo

A escassez de conjuntos de dados adequados para a classificação de sons emergenciais em tempo real representa um desafio para o desenvolvimento de tecnologias assistivas para pessoas com deficiência auditiva. Este trabalho apresenta a construção e validação de um novo dataset de áudio, contendo 2.849 clipes de 1 segundo em nove classes de sons críticos. Para validar a eficácia do dataset, quatro arquiteturas de redes neurais com complexidade crescente (MLP, CNN, ResNet-18 e ResNet-50) foram implementadas e comparadas. Os resultados confirmaram a viabilidade do dataset, com as acurácias variando de 79,30% (MLP) a um máximo de 86,67% (ResNet-18). A análise comparativa indicou que, embora a ResNet-18 tenha obtido o melhor desempenho, a arquitetura CNN 2D apresentou uma performance robusta (81,75%) com uma complexidade significativamente menor. O estudo valida o dataset como um recurso eficaz para pesquisas na área e conclui que a arquitetatura CNN 2D representa o compromisso mais pragmático entre acurácia e eficiência computacional para uma futura implementação em dispositivos vestíveis de baixo custo.

## 📦 O Dataset: UFRB-ESD

Este repositório contém o **código-fonte** e os experimentos. O **dataset UFRB-ESD (UFRB Emergency Sound Dataset)**, contendo os 2.849 arquivos de áudio `.wav` e metadados, está permanentemente hospedado no Zenodo para garantir a sua citação e acessibilidade.

### **➡️ Link para o Dataset (Zenodo):**

**https://doi.org/10.5281/zenodo.17517988**

*(O dataset é distribuído sob a licença Creative Commons Attribution 4.0 International)*

---

## 💻 Conteúdo do Repositório

Este repositório contém o notebook Jupyter (`modelos.ipynb`) utilizado para realizar todos os experimentos descritos no artigo. O notebook inclui:

* Carregamento e pré-processamento dos dados de áudio.
* Extração de características (MFCCs).
* Definição, treinamento e avaliação dos 4 modelos:
    1.  MLP (Baseline não convolucional)
    2.  CNN 2D (Baseline convolucional)
    3.  ResNet-18 (Melhor acurácia)
    4.  ResNet-50
* Geração de todas as métricas, matrizes de confusão e curvas ROC.

## 🛠 Requisitos

O notebook foi desenvolvido em ambiente Google Colab e utiliza as seguintes bibliotecas principais:

* `tensorflow`
* `librosa`
* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`

## 🚀 Como Usar

1.  Baixe o dataset UFRB-ESD do link do Zenodo acima.
2.  Faça o upload do dataset para um local de sua preferência (ex: Google Drive).
3.  Abra o notebook `modelos.ipynb` no Google Colab ou em um ambiente Jupyter local.
4.  Ajuste as variáveis de caminho no início do notebook (ex: `DATASET_PATH`, `MODELOS_PATH`) para apontar para onde você salvou os dados.
5.  Execute as células em sequência para replicar os resultados.

## 🎓 Como Citar

Se você utilizar o dataset UFRB-ESD ou este código em sua pesquisa, por favor, cite o nosso trabalho.
