# TRI-vLLM-GPU: Triton 추론 서버를 이용한 LLM 서빙 시스템

이 프로젝트는 NVIDIA Triton Inference Server와 vLLM을 활용하여 대규모 언어 모델(LLM)을 효율적으로 서빙하고 추론하는 시스템입니다. Docker 컨테이너를 통해 로컬 환경에서 쉽게 배포하고 사용할 수 있습니다.

## 프로젝트 구조

```
/
├── container/                  # 컨테이너 내부에 복사될 파일들
│   └── logs/                   # 로그 저장 디렉토리
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

## 라이센스

이 프로젝트는 BSD-3-Clause 라이센스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 참고 자료

- [Triton Inference Server 문서](https://github.com/triton-inference-server/server)
- [vLLM 프로젝트](https://github.com/vllm-project/vllm)
- [vLLM 지원 모델 목록](https://vllm.readthedocs.io/en/latest/models/supported_models.html)
