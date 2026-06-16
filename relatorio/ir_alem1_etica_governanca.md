# CardioIA — Fase 4 | IR ALÉM 1: Ética e Governança em Visão Computacional

**Equipe:** Alice C. M. Assis (RM566233) · Leonardo S. Souza (RM563928) · Lucas B. Francelino (RM561409) · Pedro L. T. Silva (RM561644) · Vitor A. Bezerra (RM563001)  
**Data:** Junho/2026

## 1. Dataset e Limitações Estruturais

Utilizamos o **Chest X-Ray Images (Pneumonia)** (Kaggle: paultimothymooney/chest-xray-pneumonia), o mesmo da Parte 2, com 5.856 radiografias de tórax rotuladas em NORMAL e PNEUMONIA. A análise identificou sete limitações estruturais que condicionam o alcance e a confiabilidade do modelo:

| # | Limitação | Impacto potencial |
|---|---|---|
| 1 | **Ausência de metadados demográficos** (idade, sexo, etnia) | Impede análise de fairness por subgrupo populacional |
| 2 | **Viés de fonte única:** Guangzhou Women and Children's Medical Center — pacientes pediátricos chineses exclusivamente | Generalização para adultos ou outras etnias é desconhecida |
| 3 | **Não distingue pneumonia bacteriana de viral** | Diferentes prognósticos e tratamentos não são diferenciados |
| 4 | **Desbalanceamento ~3:1** (PNEUMONIA:NORMAL) no treino | Favorece PNEUMONIA sem compensação por `class_weight` |
| 5 | **Conjunto de validação original degenerado** (16 imagens) | Monitoramento de treino pouco confiável sem re-splitting |
| 6 | **Sem informação sobre equipamento de raio-X** | Artefatos de hardware específicos não controlados |
| 7 | **Potencial viés de confirmação** — imagens de pacientes já diagnosticados | Dataset pode superestimar casos positivos claros e excluir casos limítrofes |

Essas limitações comprometem a generalização do modelo para populações fora do contexto de coleta (adultos, outras regiões, outros equipamentos), e tornam mandatória a validação prospectiva antes de qualquer uso clínico.

## 2. Análise de Fairness: Resultados

Como o dataset não possui metadados demográficos, aplicamos **fairness baseada em classe**: tratamos os grupos NORMAL e PNEUMONIA como as duas "populações" e verificamos se as taxas de erro são simétricas entre elas.

### Métricas no conjunto de teste (624 imagens: 234 NORMAL, 390 PNEUMONIA)

| Métrica | NORMAL | PNEUMONIA |
|---|---|---|
| Recall (TPR) | ~72.2% | ~98.5% |
| Precisão | ~94.4% | ~85.5% |
| F1-score | ~81.7% | ~91.5% |
| F1 Macro | — | ~0.87 |
| FPR (Normal→Pneumonia) | ~27.8% | — |
| FNR (Pneumonia→Normal) | — | ~1.5% |
| AUC-ROC | — | ~0.97 |
| ECE (calibração) | — | ~0.05–0.10 |

> Os valores exatos variam por execução (~±2%) devido ao não-determinismo da GPU. Os números acima refletem a execução reportada no notebook 02. O notebook `03_etica_governanca.ipynb` é distribuído sem saídas salvas (para manter o arquivo leve); ao reexecutá-lo em sequência aos notebooks 01 e 02, ele regenera os gráficos de fairness (curvas ROC/PR, análise de calibração e matriz de confusão por limiar).

**Equalized Odds:** O modelo viola o critério — o gap de TPR entre PNEUMONIA (~98.5%) e NORMAL (~72.2%) é de aproximadamente **26 pontos percentuais**. Equal Opportunity também é violada pela mesma razão. Pacientes normais têm proteção significativamente menor do que pacientes com pneumonia.

**Calibração:** O ECE indica calibração moderada; o modelo tende a ser sobreconfiante (distribuição bimodal das probabilidades). Um modelo bem calibrado que exibe 80% de confiança deve acertar em ~80% dos casos — desvios disso tornam as probabilidades menos interpretáveis clinicamente.

## 3. Implicações Éticas

**Falsos Negativos — o risco mais crítico.** Com FNR de ~1.5%, aproximadamente 6 em cada 624 pacientes com pneumonia seriam incorretamente classificados como normais. Em um hospital de alta demanda com centenas de exames diários, isso escala para dezenas de casos perdidos por dia. Um falso negativo em triagem de pneumonia pode resultar em atraso no tratamento, agravamento do quadro e risco de vida — especialmente em populações pediátricas vulneráveis como a deste dataset.

**Falsos Positivos — custo do sobre-diagnóstico.** A FPR de ~28% significa que quase 1 em 3 pacientes saudáveis seria erroneamente encaminhado como suspeito de pneumonia. Cada falso alarme gera ansiedade familiar, exames adicionais desnecessários, uso indevido de antibióticos (com risco de resistência bacteriana) e sobrecarga do sistema de saúde. Em contextos de triagem em larga escala, esse custo agregado é substancial.

**Viés de representatividade — o risco estrutural.** O modelo foi treinado exclusivamente em pacientes pediátricos de um único hospital na China. Seu desempenho em adultos, pacientes de outras etnias, exames em equipamentos diferentes ou com outras comorbidades pulmonares é **completamente desconhecido**. Implantar este modelo em qualquer outro contexto sem validação local constitui uma violação dos princípios de IA responsável em saúde (beneficência, não-maleficência, justiça e autonomia informada do paciente).

## 4. Mitigações Propostas

- **Ajuste de limiar de decisão** *(implementada no notebook)*: reduzir o limiar de 0.5 para um valor que garanta recall de PNEUMONIA ≥ 99%, aceitando o trade-off de mais falsos positivos. Adequado para triagem; o limiar ideal varia conforme o contexto clínico e deve ser definido com especialistas.

- **Coleta de dados diversificados**: expandir o dataset para múltiplos hospitais, países e faixas etárias; incluir metadados demográficos para permitir auditoria de fairness por subgrupo; distinguir pneumonia bacteriana de viral nos rótulos.

- **Calibração pós-treinamento**: aplicar Platt Scaling ou Temperature Scaling para que as probabilidades do modelo reflitam com fidelidade a incerteza real, tornando os escores de confiança clinicamente interpretáveis.

- **Human-in-the-loop obrigatório**: predições com confiança entre 0.3 e 0.7 (zona de incerteza) devem ser obrigatoriamente revisadas por radiologista antes de qualquer comunicação ao paciente. Nenhuma decisão clínica deve ser tomada com base exclusivamente na saída do modelo.

---

> ⚠️ **Este protótipo educacional não deve ser utilizado em decisões clínicas reais sem validação prospectiva, calibração em população-alvo e aprovação regulatória (ANVISA/FDA).**
