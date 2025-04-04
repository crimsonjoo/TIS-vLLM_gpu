# TRI-vLLM-GPU: Triton 추론 서버를 이용한 LLM 서빙 시스템

이 프로젝트는 NVIDIA Triton Inference Server와 vLLM을 활용하여 대규모 언어 모델(LLM)을 효율적으로 서빙하고 추론하는 시스템입니다. Docker 컨테이너를 통해 로컬 환경에서 쉽게 배포하고 사용할 수 있으며, 성능 분석 도구를 통해 LLM의 성능을 측정하고 시각화할 수 있습니다.

## 프로젝트 구조

```
/
├── container/                  # 컨테이너 내부에 복사될 파일들
│   ├── logs/                   # 로그 저장 디렉토리
│   └── llm_perf_analyzer/      # LLM 성능 분석 도구
│       ├── run_perf_test.sh    # 성능 테스트 실행 스크립트
│       ├── run_visualization.py # 결과 시각화 실행 스크립트
│       ├── utils/              # 유틸리티 모듈
│       │   ├── __init__.py     # 패키지 초기화 파일
│       │   ├── charts.py       # 차트 생성 함수
│       │   ├── compare.py      # 결과 비교 함수
│       │   └── data.py         # 데이터 처리 함수
│       ├── artifacts/          # 테스트 결과 저장 디렉토리
│       └── README.md           # 성능 분석기 설명서
│
├── local/                      # 로컬 환경 설정 및 코드
│   ├── _docker/                # Docker 관련 파일
│   │   ├── .env                # 환경 변수 설정
│   │   ├── Dockerfile.tri-vllm_gpu   # Dockerfile
│   │   ├── docker-compose.tri-vllm_gpu.yml  # Docker Compose 설정
│   │   ├── entrypoint.sh       # 컨테이너 시작 스크립트
│   │   └── entrypoint_sub.sh   # 보조 시작 스크립트
│   │
│   ├── _kubernetes/            # Kubernetes 관련 파일
│   │
│   └── inference/              # 추론 관련 코드
│       ├── config/             # 설정 파일 디렉토리
│       │   └── inference_config.yaml  # 추론 설정 파일
│       ├── config_manager.py   # 설정 관리 모듈
│       ├── infer.py            # 추론 실행 스크립트
│       └── README.md           # 추론 시스템 설명서
│
└── vllm_backend/              # Triton vLLM 백엔드 코드
    └── README.md              # vLLM 백엔드 설명서
```

## 주요 기능

- **Triton Inference Server 기반 추론**: NVIDIA Triton Inference Server를 활용한 고성능 추론 서비스
- **vLLM 백엔드 지원**: 효율적인 LLM 추론을 위한 vLLM 백엔드 통합
- **Docker 컨테이너화**: 쉬운 배포와 확장을 위한 Docker 컨테이너 지원
- **다중 GPU 지원**: 여러 GPU를 활용한 병렬 처리 지원
- **동적 설정 관리**: YAML, 환경변수, .env 파일을 통한 유연한 설정 관리
- **스트리밍 추론**: 토큰 단위의 스트리밍 추론 지원
- **비동기 처리**: asyncio를 활용한 비동기 추론 처리
- **성능 분석 및 시각화**: GenAI 성능 분석기를 통한 LLM 성능 측정 및 시각화

## 시작하기

### 사전 요구사항

- Docker 및 Docker Compose 설치
- NVIDIA GPU 및 NVIDIA Container Toolkit 설치
- CUDA 11.8 이상 지원

### 설치 및 실행

1. 저장소 클론
   ```bash
   git clone https://github.com/your-username/TRI-vLLM-gpu.git
   cd TRI-vLLM-gpu
   ```

2. 환경 변수 설정
   `local/_docker/.env` 파일을 수정하여 필요한 환경 변수를 설정합니다:
   ```
   # 사용할 GPU 인덱스 설정
   NVIDIA_VISIBLE_DEVICES=0,1  # 0번, 1번 GPU 사용
   
   # 포트 설정
   TRITON_HTTP_PORT=9010
   TRITON_GRPC_PORT=9011
   TRITON_METRICS_PORT=9012
   APP_PORT=5010
   SSH_PORT=5678
   
   # 컨테이너 이름 및 이미지 설정
   CONTAINER_NAME=tri-vllm-gpu
   IMAGE_NAME=your-username/tri-vllm-python-py3
   ```

3. Docker 컨테이너 빌드 및 실행
   ```bash
   cd local/_docker
   docker compose -f docker-compose.tri-vllm_gpu.yml up -d
   ```

4. 컨테이너 접속
   ```bash
   docker exec -it tri-vllm-gpu bash
   ```

### 추론 설정

`local/inference/config/inference_config.yaml` 파일을 수정하여 추론 설정을 지정할 수 있습니다:

```yaml
# Triton 서버 설정
port: 9011

# 모델 설정
model_name: "vllm_model"
version: 1

# 샘플링 파라미터
parameters:
  temperature: 0.7
  top_p: 0.9
  stop:
    - "Human:"
    - "Assistant:"

# 프롬프트 설정
prompts:
  system_prompt: "Please reason step by step. Please answer in korean."
  user_input: "주식 투자를 잘 할 수 있는 방법에 대해서 직장인에게 추천해줘."
  template: "user: {user_input}\n{system_prompt}\n\nassistant: "
```

### 추론 실행

컨테이너 내에서 다음 명령을 실행하여 추론을 수행할 수 있습니다:

```bash
cd /workspace
python3 /local/inference/infer.py
```

또는 환경 변수를 사용하여 설정을 변경할 수 있습니다:

```bash
MODEL_NAME="다른모델" TRITON_GRPC_PORT=9012 python3 /local/inference/infer.py
```

## GenAI 성능 분석기 사용하기

GenAI 성능 분석기는 LLM의 성능을 측정하고 시각화하는 도구입니다. 다양한 지연 시간, 처리량, GPU 활용도 등의 성능 지표를 수집하고 분석할 수 있습니다.

### 성능 테스트 실행

성능 테스트를 실행하려면 다음 명령을 사용합니다:

```bash
cd /workspace/llm_perf_analyzer
./run_perf_test.sh
```

테스트 파라미터는 스크립트 내에서 직접 수정할 수 있습니다:

```bash
# 모델 및 서비스 설정
MODEL_NAME="exaone-deep-32B"    # 테스트할 모델 이름
SERVICE_KIND="triton"           # 서비스 종류 (triton, tgi, vllm 등)
BACKEND="vllm"                  # 백엔드 (vllm, tensorrtllm 등)

# 입력 및 출력 토큰 설정
INPUT_TOKENS_MEAN=200           # 입력 프롬프트의 평균 토큰 수
OUTPUT_TOKENS_MEAN=100          # 생성할 출력 토큰의 평균 수

# 테스트 설정
REQUEST_COUNT=5                 # 성능 측정에 사용할 요청 수
WARMUP_REQUEST_COUNT=2          # 워밍업에 사용할 요청 수
STREAMING=true                  # 토큰 스트리밍 사용 여부
```

### 결과 시각화

테스트 결과를 시각화하려면 다음 명령을 사용합니다:

```bash
# 단일 결과 시각화
python run_visualization.py --result_dir /workspace/llm_perf_analyzer/artifacts/perf_<모델명>_<타임스탬프>

# 여러 결과 비교
python run_visualization.py --compare /path/to/result1 /path/to/result2 --names "테스트1" "테스트2"

# 최신 결과 시각화
python run_visualization.py --latest

# 특정 모델의 최신 결과 시각화
python run_visualization.py --model <모델명>
```

### 주요 성능 지표

성능 분석기는 다음과 같은 주요 지표를 측정합니다:

- **지연 시간 (Latency)**: TTFT(Time To First Token), 토큰 간 지연 시간, 요청 지연 시간
- **처리량 (Throughput)**: 초당 생성 토큰 수, 초당 처리 요청 수
- **GPU 활용도**: GPU 사용률, 메모리 사용량, 전력 소비량

자세한 내용은 `container/llm_perf_analyzer/README.md` 파일을 참조하세요.

## vLLM 백엔드 설정

vLLM 백엔드는 Triton Inference Server에 통합되어 있으며, 다음과 같은 방법으로 설정할 수 있습니다:

1. 모델 저장소 구성
   - 모델 저장소 디렉토리를 생성하고 모델 설정 파일을 추가합니다.
   - `model.json` 파일에 vLLM 엔진 설정을 지정합니다.

2. Triton 서버 시작
   ```bash
   tritonserver --model-repository=/path/to/model_repository
   ```

3. 추론 요청 전송
   ```bash
   curl -X POST localhost:8000/v2/models/vllm_model/generate -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0}}'
   ```

## 다중 GPU 지원

다중 GPU를 활용하려면 다음 설정을 수정하세요:

1. `.env` 파일에서 사용할 GPU 인덱스 지정:
   ```
   NVIDIA_VISIBLE_DEVICES=0,1,2,3  # 0, 1, 2, 3번 GPU 사용
   ```

2. `model.json` 파일에서 텐서 병렬 크기 설정:
   ```json
   {
     "model": "meta-llama/Llama-2-7b-chat-hf",
     "tensor_parallel_size": 4,
     "gpu_memory_utilization": 0.8
   }
   ```

## 문제 해결

- **GPU 메모리 부족**: `model.json`에서 `gpu_memory_utilization` 값을 낮추어 조정하세요.
- **포트 충돌**: `.env` 파일에서 포트 번호를 변경하세요.
- **모델 로딩 실패**: 모델 경로와 접근 권한을 확인하세요.
- **성능 테스트 오류**: `run_perf_test.sh`의 파라미터가 올바르게 설정되었는지 확인하세요.
- **시각화 실패**: 필요한 Python 패키지(pandas, matplotlib, seaborn)가 설치되어 있는지 확인하세요.

## 라이센스

이 프로젝트는 BSD-3-Clause 라이센스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 참고 자료

- [Triton Inference Server 문서](https://github.com/triton-inference-server/server)
- [vLLM 프로젝트](https://github.com/vllm-project/vllm)
- [vLLM 지원 모델 목록](https://vllm.readthedocs.io/en/latest/models/supported_models.html)
- [GenAI-Perf 도구](https://github.com/triton-inference-server/genai-perf)
