# 🧮 MV: From Formula Images to Structured LaTeX via RCNN + Transformer

> 수식 이미지에서 구조화된 LaTeX 코드(FLC)를 생성하기 위한 Vision-to-Sequence 파이프라인  
> *이 프로젝트는 인공지능이 수학 문제를 자동으로 인식하고, 해석에 도움이 되는 반응형 그래프 생성 프로그램을 만들기 위한 선행 연구 프로젝트입니다.*

---

## 🧠 프로젝트 개요

본 프로젝트는 수식 이미지를 입력받아 **수식 구성 요소를 탐지하고**, 이를 시퀀스로 변환하여 최종적으로 구조화된 LaTeX 코드를 생성하는 파이프라인을 구현합니다.

이 과정은 크게 다음 3단계로 구성됩니다:

1. **이미지 → VLC(Visible LaTeX Characters)**  
2. **VLC → FLC(Full LaTeX Characters)**  
3. **FLC 출력 → 수식 재구성**

---

## 🔍 주요 설계 철학

- 수식은 단순한 문자 나열이 아닌 **공간적 구조가 핵심**
- 따라서 **정확한 박스 위치 정보 + 시퀀스 모델링**이 필수
- 이를 위해 **Faster R-CNN 기반 모델로 박스 정보 추출**
- 추출된 위치 정보와 함께 Transformer에 전달 → 추론 성능 강화

---

## 🧩 전체 파이프라인

```
[수식 이미지]
   ↓
RCNN (Faster R-CNN + CoordConv2d)
   ↓
VLC 시퀀스 (박스 + 토큰)
   ↓
커스텀 Transformer (Seq2Seq)
   ↓
FLC 시퀀스
```

---

## 🏗️ 디렉토리 구조

```
MV/
├── img2vlc/         # 이미지 → VLC 시퀀스 변환 모델
├── preprocessing/   # VLC 시퀀스 정제 및 분석
├── vlc2flc/         # VLC → FLC 변환 (Transformer 기반)
├── utils.py         # 공통 유틸 함수
├── configs.py       # 전역 설정값
├── vlc2tok.json     # VLC 토큰 정의
```

---

## 📌 기술 스택

| 파트 | 모델 | 설명 |
|------|------|------|
| **Detection** | `FasterRCNN (ResNet50 + FPN)` | 수식 객체 탐지 및 위치 추출 |
| **Conv 보강** | `CoordConv2d` | CNN에 위치 정보 강화 |
| **Transformer** | `Custom Seq2Seq Transformer` | VLC → FLC 변환 |
| **시퀀스 정의** | `VLC` = Visible LaTeX Characters | 수식의 문자 단위 표현 |
|                | `FLC` = Full LaTeX Characters | 구조화된 LaTeX 시퀀스 (VLC 조합 결과) |

---

## 📈 성능 회고

### ✅ RCNN (성공)
- 수식 기호의 위치 인식에 매우 효과적
- CoordConv2d를 통해 공간 감지 능력 향상
- 박스 정확도 및 수식 구문 구성까지 만족

### ❌ Transformer (실패)
- **정확도 최악 수준**  
- 숫자와 문자 조합은 맞았지만 **순서 왜곡** 심각
- **위치 정보(boxes)를 전혀 인식하지 못함**
- 원인 추정:
  - 구조적 한계: Transformer에 위치 정보가 충분히 인코딩되지 않음
  - 노트북 환경에서의 **성능 부족**으로 학습 효율 저하
  - sequence alignment 불안정 + masking 미완성

---

## 🧪 실험적 통찰

- 다양한 detection 모델 실험 후, 당시 기준 최고의 성능을 가진 **RCNN 사용**
- 프로젝트 완료 후에는 다음과 같은 결론 도출:
  > "처음부터 Vision Transformer 기반 인코더(ViT)를 사용했더라면  
  > 공간 정보와 시퀀스 인식이 자연스럽게 통합되었을 것."

---

## ❓ 용어 정리

| 용어 | 의미 |
|------|------|
| **VLC (Visible LaTeX Characters)** | 수식 이미지에서 추출된 LaTeX 코드 문자 단위 시퀀스 |
| **FLC (Full LaTeX Characters)** | VLC들을 구조적 의미 단위로 조합한 최종 수식 표현 |

예시:

```python
VLC: ['\\frac', '{', 'x', '}', '{', 'y', '}']
FLC: "\\frac{x}{y}"
```

---

## ✍️ 개발자 노트

> "수식은 시각적이면서도 구조적인 언어입니다.  
> 우리는 RCNN을 통해 그 시각 구조를 인식하고, Transformer를 통해 그것을 언어화했습니다.  
> 성능의 한계와 구조적 교훈을 통해, 더 나은 모델링 방향으로 나아갈 수 있었습니다."
