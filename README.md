# 비트코인 딥러닝 트레이딩 모델 & 전략 보고서 📈

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ScathachSkadi/TimeSeriesForecastingTest/blob/main/solution_notebook.ipynb)

본 프로젝트는 비트코인(BTC)의 가격 변화 방향을 예측하는 **딥러닝 모델**을 설계하고, 이를 기반으로 **투자 전략**을 수립하여 벤치마크(Buy and Hold) 대비 초과 수익을 달성하는 것을 목표로 합니다.

---

## 1. 모델 설계 및 훈련 (Model Design & Training)

### 🧠 모델 아키텍처: TCN (Temporal Convolutional Network)
기존 RNN 및 Transformer의 단점(학습 불안정성, 데이터 과적합)을 개선하기 위해 **TCN**을 채택하였습니다.
*   **Dilated Convolutions**: 적은 파라미터로 긴 시계열의 과거 정보를 효율적으로 참조합니다.
*   **Causal Padding**: 미래의 정보가 과거로 유출(Leakage)되는 것을 방지합니다.
*   **Residual Connection**: 층을 깊게 쌓아도 학습이 안정적으로 이루어지도록 돕습니다.
*   **구조**: Input -> Temporal Block (Dilated Conv + ReLU + Dropout) x N -> Global Pooling -> FC Layer -> Sigmoid

### 🔍 하이퍼파라미터 최적화 (Optuna)
**검증 손실(Loss) 최소화**가 아닌, **총 수익률(Total Return) 극대화**를 목표로 변경하여 더 실전적인 최적화를 수행했습니다.
*   **튜닝 대상**: 
*   **튜닝 대상**: 
    *   **`kernel_size`**: [2, 3, 5] (커널 크기)
    *   **`n_levels`**: [2, 3, 4] (네트워크 깊이)
    *   **`num_channels`**: [16, 32, 64] (필터 개수)
    *   `dropout`: 0.1 ~ 0.5
    *   `learning_rate`: 1e-4 ~ 1e-2
    *   **`threshold`**: 0.4 ~ 0.7 (진입 임계값 자동 최적화)
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

*   **한계점**: 데이터 부족으로 인해 급격한 추세 전환이나 블랙 스완 이벤트를 예측하는 데에는 한계가 있을 수 있습니다.
*   **의의**: 그럼에도 불구하고, 무작정 보유하는 것(Buy & Hold)보다 **MDD(최대 낙폭)를 줄이고 안정적인 수익 곡선을 만드는 데 효과적임**을 확인했습니다.

