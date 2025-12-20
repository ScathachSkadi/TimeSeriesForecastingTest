# 비트코인 딥러닝 트레이딩 모델 & 전략 보고서 📈

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ScathachSkadi/TimeSeriesForecastingTest/blob/main/solution_notebook.ipynb)

본 프로젝트는 비트코인(BTC)의 가격 변화 방향을 예측하는 **딥러닝 모델**을 설계하고, 이를 기반으로 **투자 전략**을 수립하여 벤치마크(Buy and Hold) 대비 초과 수익을 달성하는 것을 목표로 합니다.

---

## 1. 모델 설계 및 발전 과정 (Model Evolution)
본 프로젝트는 단순히 하나의 모델을 적용한 것이 아니라, **총 3단계의 아키텍처 개선**을 통해 최적의 모델을 도출했습니다.

### 🏛️ 1단계: GRU (Gated Recurrent Unit)
*   **초기 접근**: 시계열 처리에 강한 RNN 기반의 GRU를 사용.
*   **한계**: 초기 학습은 빠르나, 긴 시퀀스(Long Sequence)에서 기울기 소실 문제 발생 및 성능 정체.

### 🏛️ 2단계: Transformer
*   **시도**: Attention 메커니즘을 도입하여 장기 의존성(Long-term dependency) 해결 시도.
*   **결과**: 데이터셋 크기가 작아 과적합(Overfitting)이 심하게 발생하여 일반화에 실패.

### 🏛️ 3단계: TCN (최종 선정) & 규제 강화
*   **해결책**: 작은 데이터셋에서도 안정적인 **TCN (Temporal Convolutional Network)** 도입.
*   **추가 개선**: Optuna를 통한 파라미터 튜닝과 **L2 규제(Weight Decay)**, **Dropout**을 강화하여 과적합 문제를 기술적으로 해결.

### 🧠 최종 모델 아키텍처: Regularized TCN
기존 RNN 및 Transformer의 단점(학습 불안정성, 데이터 과적합)을 개선하기 위해 **TCN**을 채택하였습니다.
*   **Dilated Convolutions**: 적은 파라미터로 긴 시계열의 과거 정보를 효율적으로 참조합니다.
*   **Causal Padding**: 미래의 정보가 과거로 유출(Leakage)되는 것을 방지합니다.
*   **Residual Connection**: 층을 깊게 쌓아도 학습이 안정적으로 이루어지도록 돕습니다.
*   **구조**: Input -> Temporal Block (Dilated Conv + ReLU + Dropout) x N -> Global Pooling -> FC Layer -> Sigmoid

### 🔍 하이퍼파라미터 최적화 (Optuna)
**검증 손실(Loss) 최소화**가 아닌, **총 수익률(Total Return) 극대화**를 목표로 변경하여 더 실전적인 최적화를 수행했습니다.
*   **튜닝 대상**: 
*   **튜닝 대상**: 
    *   **과적합(Overfitting) 방지 전략**:
        *   `weight_decay`: **L2 규제** 추가 (Optuna 튜닝 대상)
        *   `dropout`: 0.2 ~ 0.5 (높은 Drop 확률)
        *   `kernel_size`: [2, 3] (복잡도 감소를 위해 5 제외)
        *   `num_channels`: [16, 32] (네트워크 용량 축소)
    *   `learning_rate`: 1e-4 ~ 1e-3
    *   **`threshold`**: 0.4 ~ 0.7
*   **목표 함수**: Validation Set에서의 시뮬레이션 수익률 (Maximize Profit)

---

## 2. 투자 전략 설계 (Trading Strategy)

### 📊 스마트 신뢰도 전략 (Smart Confidence Strategy)
단순히 상승/하락 예측(0/1)만 사용하는 것이 아니라, 모델의 **확신(Confidence)** 정도에 따라 투자 비중을 조절하는 전략을 수립했습니다.

1.  **진입 임계값 (Threshold)**: **Optuna에 의해 최적화됨 (예: 0.5 ~ 0.65)**
    *   고정된 값(0.6)을 쓰는 대신, 현재 시장 상황과 모델 특성에 가장 적합한 임계값을 자동으로 찾습니다.
2.  **동적 포지션 사이징 (Dynamic Position Sizing)**:
    *   확률이 높을수록 투자 비중을 늘립니다.
    *   `투자 비중 = (예측확률 - 0.5) * 2` (최대 100%)
    *   예: 확률 60% → 자본의 20% 투자 / 확률 80% → 자본의 60% 투자

---

## 3. 벤치마크 비교 및 분석 (Analysis)

| 전략 (Strategy) | 특징 | 장점 | 단점 |
| :--- | :--- | :--- | :--- |
| **Buy and Hold** | 시작 시점에 전액 매수 후 보유 | 상승장에서 강력함, 수수료 최소화 | 하락장에서 큰 손실 발생 가능 |
| **Simple Strategy** | 상승 예측 시 전액 매수 | 구현이 간단함 | 잦은 거래로 수수료 과다, 신호 노이즈에 취약 |
| **Smart Confidence** | **확률 기반 비중 조절** | **리스크 관리 우수, 하락장에서 손실 방어** | 급격한 폭등장에서는 수익률이 Buy&Hold보다 낮을 수 있음 |

**분석 결과**:
본 모델은 하락장이나 횡보장에서 현금 비중을 조절함으로써 벤치마크 대비 안정적인 우상향 자산 곡선을 그리는 것을 목표로 합니다. Optuna를 통해 최적화된 모델은 노이즈가 많은 금융 데이터에서도 일반화된 패턴을 잘 포착하였습니다.


---

## 4. 시각화 및 평가 (Visualizations)

`solution_notebook.ipynb`에는 모델의 성능을 입증하기 위한 다양한 시각화 자료가 포함되어 있습니다.
1.  **Training Loss Plot**: 학습 진행 상황 시각화 (Train vs Validation Loss)
2.  **Hyperparameter Importance**: 주요 하이퍼파라미터 영향력 분석 (Optuna Visualization)
3.  **Confusion Matrix**: 0.6 임계값 기준의 예측 정확도 히트맵
4.  **Strategy Performance**: 전략 수익률 vs 벤치마크 수익률 비교 그래프

---

## 4. 실행 방법 (How to Run)

본 레포지토리에는 새로 작성된 **솔루션 노트북**이 포함되어 있습니다.

1.  **환경 설정**:
    ```bash
    pip install -r requirements.txt
    ```
    *(yfinance, torch, optuna 등 필수 라이브러리가 설치됩니다)*

2.  **솔루션 실행**:
    *   `solution_notebook.ipynb` 파일을 실행합니다.

### Google Colab 실행 (Open in Colab)
`Open in Colab` 버튼을 클릭하여 노트북을 열면, 첫 번째 셀에서 다음 작업이 자동으로 수행됩니다:
1.  **필수 라이브러리 설치** (`optuna`, `yfinance` 등)
2.  **`utils.py` 다운로드** (데이터 로딩 및 전처리 함수 포함)

---

## 5. 결론 및 한계점 (Conclusion & Limitations)

**"보수적인 리스크 관리가 핵심입니다."**

본 프로젝트의 데이터셋(약 5년치 일일 데이터)은 딥러닝 모델이 모든 시장 상황을 완벽하게 학습하기에는 다소 **부족한 양(Small Dataset)**입니다. 따라서 본 모델은 공격적으로 큰 수익을 추구하기보다는, **하락장에서의 손실을 방어하고 작지만 확실한 수익을 쌓아가는 것**에 최적화되어 있습니다.

*   **한계점**: 
    *   **데이터 부족 (Data Scarcity)**: 딥러닝 모델은 수만 개 이상의 데이터가 필요하지만, 일일 데이터는 수천 개에 불과합니다. 이로 인해 모델이 복잡한 패턴을 배우기보다 단순한 추세만 학습할 가능성이 높습니다.
    *   **입력 변수의 한계**: 가격(Price)과 거래량(Volume) 정보만으로는 시장의 모든 변수(거시경제, 뉴스 등)를 설명할 수 없습니다.

## 6. 향후 개선 방향 (Future Work)
만약 이 프로젝트를 더 발전시킨다면 다음 접근법을 추천합니다:
1.  **외부 데이터 추가 (Feature Engineering)**: 온체인 데이터, 금리, 뉴스 감성 분석(Sentiment Analysis) 등을 입력 변수로 추가하여 정보량을 늘립니다.
2.  **데이터 증강 (Data Augmentation)**: 시계열 데이터에 노이즈를 섞거나 변형하여 학습 데이터 양을 인위적으로 늘립니다.
3.  **앙상블 (Ensemble)**: TCN뿐만 아니라 XGBoost, Random Forest 등 서로 다른 성격의 모델을 결합하여 예측 안정성을 높입니다.

