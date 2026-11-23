# UFRB-ESD — Emergency Sound Dataset

Código e resultados experimentais do artigo **UFRB-ESD: Dataset para Classificação de Sons Emergenciais em Dispositivos Vestíveis para Deficientes Auditivos**, publicado na Revista Eletrônica de Ciências Exatas e Tecnológicas (RECET).

**Autores:** Gabriel Sampaio de Oliveira e Igor Dantas dos Santos Miranda — Universidade Federal do Recôncavo da Bahia (UFRB), Brasil.

- [Artigo — DOI: 10.66983/recet.v7i1.5864](https://doi.org/10.66983/recet.v7i1.5864)
- [Dataset — DOI: 10.5281/zenodo.17517988](https://doi.org/10.5281/zenodo.17517988)

## Conjunto de dados e protocolo experimental

O UFRB-ESD contém 2.849 clipes de um segundo, distribuídos em nove classes: buzina, cachorro, colisão, explosão, freio, grito, outros, sirene e tiro. A classe *outros* reúne sons cotidianos não emergenciais. Os áudios são disponibilizados separadamente no Zenodo.

O benchmark utiliza 2.827 clipes. Os 22 arquivos excluídos por saturação digital acentuada permanecem no conjunto público e estão identificados em `data_exclusions.csv`. O manifesto registra o caminho, a classe, a origem inferida e a assinatura digital de cada arquivo utilizado no experimento.

A avaliação emprega validação cruzada estratificada e agrupada em dez folds, com semente 42. Recortes da mesma gravação-fonte permanecem na mesma partição. A atribuição final contém 868 grupos e é compartilhada por MLP, CNN 2D, ResNet-18 e ResNet-50. A validação interna também respeita os grupos e é utilizada para *early stopping*. As características MFCC são normalizadas com estatísticas calculadas apenas no treinamento de cada rodada.

Parte das relações de origem foi reconstruída pelos nomes dos arquivos. As regras, as correções manuais e a atribuição final dos folds estão incluídas para permitir sua inspeção.

## Resultados de referência

Média ± desvio-padrão nos dez folds externos, em porcentagem:

| Modelo | Acurácia | F1 ponderado | F1 macro | AUC macro |
|---|---:|---:|---:|---:|
| MLP | 77,32 ± 3,36 | 76,23 ± 3,25 | 66,50 ± 4,99 | 94,06 ± 1,35 |
| CNN 2D | 77,18 ± 4,24 | 76,30 ± 4,48 | 67,38 ± 3,85 | 94,90 ± 1,22 |
| ResNet-18 | 75,70 ± 4,08 | 75,66 ± 4,78 | 67,59 ± 5,72 | 93,89 ± 1,32 |
| ResNet-50 | 74,89 ± 4,16 | 75,11 ± 3,86 | 66,24 ± 4,96 | 94,03 ± 1,57 |

A CNN 2D apresentou a maior AUC macro média e o maior F1 ponderado médio; a MLP obteve a maior acurácia média. A inferência estatística utiliza 5.000 reamostragens dos grupos de gravação-fonte, com correção de Bonferroni nas seis comparações entre modelos por métrica. Todos os intervalos simultâneos das diferenças incluem zero, não indicando superioridade consistente de uma arquitetura.

As estimativas OOF (*out-of-fold*) agregadas e as médias dos folds são sínteses distintas. Os testes por fold com correção de Holm, também disponibilizados, têm caráter exploratório e não substituem a análise agrupada adotada no artigo.

## Organização do repositório

| Arquivo ou diretório | Conteúdo |
|---|---|
| `modelos.ipynb` | Protocolo de treinamento e avaliação, com saídas da execução de referência |
| `resultados_artigo.ipynb` | Consulta e geração das tabelas e figuras a partir dos resultados consolidados |
| `grouped_cv_pipeline.py` | Extração de características, particionamento, modelos e métricas |
| `ufrb_esd_grouping.py` | Identificação e agrupamento dos recortes por gravação de origem |
| `group_manifest.csv`, `group_overrides.csv`, `data_exclusions.csv` | Manifesto, correções de agrupamento e exclusões do benchmark |
| `audit_pcm_quality.py` | Auditoria das amostras PCM dos arquivos originais |
| `analise_bootstrap_agrupado.cs` | Reamostragem agrupada e comparações entre modelos |
| `resultados_grouped_cv/full_10fold/` | Previsões OOF, métricas, partições, tabelas, figuras e configuração experimental |

Registros intermediários de cada treinamento, pesos dos modelos e cache de características não integram esta distribuição. As métricas consolidadas das 40 combinações de modelo e fold estão em `fold_metrics.csv`; as probabilidades OOF estão em `oof_predictions.csv`.

## Consulta dos resultados

`resultados_artigo.ipynb` apresenta as tabelas, as curvas ROC, as matrizes de confusão e os resultados do bootstrap. As saídas estão salvas e podem ser consultadas sem executar o notebook. Sua execução requer os arquivos deste repositório, mas não os áudios, TensorFlow ou GPU.

Para uso local, instale as dependências de consulta e abra o notebook na raiz do repositório:

```bash
python -m pip install -r requirements-resultados.txt
```

No Google Colab, mantenha os arquivos em `Meu Drive/UFRB-ESD` e abra o notebook correspondente. O caminho é definido na célula de configuração e pode ser ajustado para outra localização.

## Reprodução do experimento

O ambiente da execução de referência está registrado em `resultados_grouped_cv/full_10fold/environment.json`, e os parâmetros estão em `experiment_config.json`. As versões registradas das bibliotecas constam em `requirements.txt`. A versão de Matplotlib não foi registrada; sua dependência é indicada separadamente como biblioteca de visualização. Diferenças de ambiente e hardware podem produzir variações numéricas.

1. Instale as dependências de `requirements.txt` ou utilize a célula de instalação de `modelos.ipynb`.
2. Obtenha os áudios no Zenodo e organize-os em `data/<classe>/<arquivo>.wav`, na raiz do projeto.
3. Confira `PROJECT_DIR` e defina `RUN_TRAINING=True` em `modelos.ipynb`. Para o experimento completo, mantenha `QUICK_TEST=False` e `TRAIN_FINAL_MODEL=False`.
4. Execute as células na ordem. Novos resultados são gravados em `resultados_reproducao/`, separadamente dos arquivos de referência. Esse diretório passa a conter os registros individuais, permitindo retomar uma execução interrompida com a mesma configuração.

Por padrão, `RUN_TRAINING=False` mantém a consulta dos resultados publicados. A etapa opcional de treinamento em todos os dados produz um artefato de implantação e não compõe as métricas OOF.

Se a cópia dos áudios ainda contiver `data/tiro/15720723.wav`, o nome correspondente no manifesto é `data/tiro/1572073.wav`. Confira o arquivo antes de renomeá-lo, sem sobrescrever outro arquivo existente.

O programa `analise_bootstrap_agrupado.cs` recebe, após compilação, o CSV de previsões, dois caminhos de saída e o número opcional de replicações:

```text
analise_bootstrap_agrupado.exe oof_predictions.csv saida_modelos.csv saida_pares.csv 5000
```

Os CSVs do bootstrap de referência já estão incluídos. Uma nova análise deve utilizar caminhos de saída distintos para preservá-los.

## Citação e licença

Ao utilizar o conjunto ou o protocolo experimental, cite o artigo e o registro do dataset nos respectivos DOIs. Referência do artigo:

OLIVEIRA, Gabriel Sampaio de; MIRANDA, Igor Dantas dos Santos. UFRB-ESD: Dataset para Classificação de Sons Emergenciais em Dispositivos Vestíveis para Deficientes Auditivos. **Revista Eletrônica de Ciências Exatas e Tecnológicas**, v. 7, n. 1, p. 74–79, 2026. DOI: [10.66983/recet.v7i1.5864](https://doi.org/10.66983/recet.v7i1.5864).

O dataset é distribuído sob a licença Creative Commons Attribution 4.0 International (CC BY 4.0), conforme o registro no Zenodo. Essa indicação refere-se aos áudios e não estabelece uma licença adicional para o código.
