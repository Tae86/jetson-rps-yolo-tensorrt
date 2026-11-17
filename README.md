# ✊✌️✋ Mini RPS — Jetson 온디바이스 가위바위보 게임

> **YOLO + TensorRT**로 카메라 속 손 모양(가위/바위/보)을 실시간 인식하는 **온디바이스 AI** 가위바위보 게임입니다.
> NVIDIA Jetson에서 모델을 TensorRT 엔진으로 변환해 클라우드 없이 로컬에서 추론합니다.
> *AI 시스템반도체 SW개발자 2기 (대한상공회의소) · On-device AI 과정*

## ✨ 주요 기능

| 모드 | 설명 |
|---|---|
| **SINGLE** (vs CPU) | 손을 비추면 3초 카운트다운 → CPU가 랜덤으로 가위/바위/보를 공개하고 승패 판정 (`YOU WIN!` / `CPU WINS!` / `DRAW`) |
| **MULTI** (2인 대결) | 카메라에 손 2개를 동시에 비추면 P1(시안)/P2(마젠타)로 자동 구분해 판정 |

- 카운트다운 중 손을 0.5초 넘게 떼면 자동 취소 → 다시 비추면 재시작
- 같은 자세를 계속 들고 있어도 중복 채점되지 않음 (손을 떼야 다음 라운드)
- OpenCV로 직접 그린 네온 테마 UI (그라데이션, 글로우, 파티클 효과), 모드별 스코어보드

## 🧠 기술 구성

| 영역 | 내용 |
|---|---|
| 모델 | YOLO (Colab 학습) 3클래스: `scissors` / `rock` / `paper` |
| 추론 최적화 | `best.pt → best.onnx → best.engine` (TensorRT, `trtexec`로 Jetson에서 변환) |
| 실행 | `ultralytics`로 `.engine` 로드, `cv2.VideoCapture` 640×480, 버퍼 1 (지연 최소화) |
| 참고 모듈 | `trt_module.py` — 수업에서 제공된 TensorRT 추론 래퍼 (게임 실행에는 사용하지 않음) |
| UI | OpenCV 전용 (마우스 버튼 + 키보드 입력) |

## 📁 파일 구성

| 파일 | 설명 |
|---|---|
| `Mini_RPS.py` | 메인 게임 스크립트 |
| `best.onnx` | 학습된 YOLO 모델 (ONNX) |
| `rock.png` `paper.png` `scissors.png` | SINGLE 모드에서 CPU가 내는 손 이미지 |
| `trt_module.py` | 수업 제공 TensorRT 추론 래퍼 (참고용) |

> `best.engine`은 Jetson/TensorRT 버전에 종속되므로 저장소에 포함하지 않았습니다. 아래 방법으로 직접 생성하세요.

## 🚀 실행 방법

```bash
# 1) 환경
source venv/bin/activate
pip install opencv-python numpy ultralytics

# 2) Jetson에서 TensorRT 엔진 생성 (최초 1회)
trtexec --onnx=best.onnx --saveEngine=best.engine

# 3) 실행 (모든 파일을 같은 폴더에 두기)
python Mini_RPS.py
```

### 조작키

| 입력 | 동작 |
|---|---|
| `1` / `2` 또는 버튼 클릭 | SINGLE / MULTI 선택 |
| `m` 또는 MENU 버튼 | 메뉴로 돌아가기 |
| `r` | 현재 모드 스코어 초기화 |
| `q` | 종료 |

## 📚 배운 점

- 학습한 모델을 ONNX → TensorRT로 변환해 엣지 디바이스에서 실시간 추론하는 전체 파이프라인
- 카메라 버퍼 지연, 중복 판정 방지 등 실시간 인터랙션 로직 설계
