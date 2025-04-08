# TRI-vLLM-GPU 추론 시스템

이 프로젝트는 Triton Inference Server와 vLLM을 사용하여 대규모 언어 모델(LLM)의 추론을 수행하는 시스템입니다.

## 프로젝트 구조

```
/local/inference/
├── config/
│   └── inference_config.yaml  # 추론 설정 파일
├── src/                       # 소스 코드 디렉토리
│   ├── __init__.py            # 패키지 초기화 파일
│   ├── config/                # 설정 관련 모듈
│   │   ├── __init__.py        # 설정 패키지 초기화
│   │   └── manager.py         # 설정 관리 클래스
│   └── inference/             # 추론 관련 모듈
│       ├── __init__.py        # 추론 패키지 초기화
│       ├── client.py          # Triton 클라이언트 함수
│       └── runner.py          # 추론 실행 함수
├── main.py                    # 메인 실행 스크립트
└── README.md                  # 프로젝트 설명서
```

## 주요 기능

- **동적 설정 관리**: 다양한 소스(YAML, 환경변수, .env 파일)에서 설정을 로드하고 우선순위를 적용
- **Triton 서버 연결**: gRPC를 통한 Triton Inference Server 연결 및 통신
- **스트리밍 추론**: 토큰 단위의 스트리밍 추론 지원
- **비동기 처리**: asyncio를 활용한 비동기 추론 처리
- **모듈화된 구조**: 기능별로 분리된 모듈 구조로 유지보수성 향상

## 설정 관리

설정은 다음 우선순위로 적용됩니다:
1. 환경 변수
2. .env 파일 (Docker 컨테이너 설정)
3. YAML 설정 파일 (inference_config.yaml)
4. 기본값

### 주요 설정 항목

- **port**: Triton 서버의 gRPC 포트 (기본값: 9001)
- **model_name**: 사용할 모델 이름 (기본값: "vllm_model")
- **version**: 모델 버전 (기본값: 1)
- **parameters**: 샘플링 파라미터 (temperature, top_p 등)
- **prompts**: 프롬프트 관련 설정 (system_prompt, user_input, template)

## 사용 방법

### 1. 설정 파일 준비

`config/inference_config.yaml` 파일을 수정하여 기본 설정을 지정할 수 있습니다:

```yaml
# Triton 서버 설정
port: 9011

# 모델 설정
model_name: "Deep-7.8B"
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

### 2. 추론 실행

```bash
# 기본 설정으로 실행
python main.py

# 스트리밍 모드 비활성화
python main.py --no-stream

# 환경 변수로 설정 변경
MODEL_NAME="다른모델" TRITON_GRPC_PORT=9012 python main.py
```

### 3. 코드에서 사용

```python
import asyncio
from src.inference import try_request
from src.config import get_config

# 기본 설정으로 추론
asyncio.run(try_request())

# 커스텀 설정으로 추론
custom_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 200
}

custom_prompts = {
    "system_prompt": "당신은 전문 코딩 튜터입니다.",
    "user_input": "파이썬으로 퀵소트를 구현해주세요.",
    "template": "{system_prompt}\n질문: {user_input}\n답변: "
}

asyncio.run(try_request(
    model_name="코딩모델",
    version=2,
    sampling_params=custom_params,
    prompts=custom_prompts,
    stream=True
))
```

## 모듈 설명

### 1. 설정 관리 모듈 (src/config)

설정 관리 모듈은 다양한 소스에서 설정을 로드하고 관리합니다:

```python
from src.config import get_config

# 특정 설정 가져오기
port = get_config("port", 9001)
model_name = get_config("model_name", "기본모델")

# 중첩된 설정 가져오기
temperature = get_config("parameters.temperature", 0.7)

# 모든 설정 가져오기
all_config = get_config()
```

### 2. 추론 모듈 (src/inference)

추론 모듈은 Triton 서버와의 통신 및 추론 실행을 담당합니다:

```python
from src.inference import try_request

# 추론 실행
await try_request(
    model_name="모델명",
    version=1,
    sampling_params={"temperature": 0.7},
    prompts={"user_input": "질문내용"},
    stream=True
)
```

## Docker 환경

Docker 컨테이너에서 실행할 경우, `_docker/.env` 파일의 설정이 자동으로 적용됩니다:

```
TRITON_HTTP_PORT=9010
TRITON_GRPC_PORT=9011
TRITON_METRICS_PORT=9012
```

## 주의사항

- Triton 서버가 실행 중이어야 추론이 가능합니다.
- Docker 컨테이너에서 실행할 경우 포트 매핑이 올바르게 설정되어 있어야 합니다.
- 모델 이름과 버전이 Triton 서버에 로드된 모델과 일치해야 합니다.

# Triton Inference Server와 vLLM을 이용한 추론 시스템

이 디렉토리는 NVIDIA Triton Inference Server와 vLLM 백엔드를 사용하여 대규모 언어 모델(LLM)의 추론을 수행하는 시스템을 포함합니다. 동적 설정 관리, gRPC 통신, 토큰 스트리밍 등의 기능을 제공합니다.

## 디렉토리 구조

```
inference/
├── config/                 # 설정 파일 디렉토리
│   └── inference_config.yaml  # 기본 추론 설정 파일
├── config_manager.py       # 설정 관리 모듈
├── infer.py                # 추론 실행 스크립트
└── README.md               # 이 파일
```

## 주요 기능

- **동적 설정 관리**: YAML 파일, 환경 변수, 명령줄 인수를 통한 설정 관리
- **gRPC 통신**: Triton Inference Server와의 효율적인 gRPC 통신
- **토큰 스트리밍**: 생성된 토큰을 실시간으로 스트리밍하는 기능
- **비동기 처리**: asyncio를 활용한 비동기 추론 처리
- **다양한 모델 지원**: 다양한 LLM 모델과 설정 지원

## 설치 및 요구사항

이 시스템은 TRI-vLLM-GPU 프로젝트의 Docker 컨테이너에 이미 포함되어 있습니다. 별도의 설치가 필요하지 않습니다.

필요한 Python 패키지:
- tritonclient[grpc]
- pyyaml
- numpy
- tqdm
- asyncio

## 사용 방법

### 기본 사용법

추론 시스템을 실행하려면 다음 명령을 사용합니다:

```bash
python infer.py
```

기본적으로 `config/inference_config.yaml` 파일의 설정을 사용합니다.

### 환경 변수를 통한 설정

환경 변수를 사용하여 설정을 변경할 수 있습니다:

```bash
MODEL_NAME="다른모델" TRITON_GRPC_PORT=9012 python infer.py
```

지원되는 환경 변수:
- `TRITON_SERVER`: Triton 서버 주소 (기본값: "localhost")
- `TRITON_GRPC_PORT`: Triton gRPC 포트 (기본값: 9011)
- `MODEL_NAME`: 사용할 모델 이름 (기본값: "vllm_model")
- `MODEL_VERSION`: 모델 버전 (기본값: 1)
- `CONFIG_PATH`: 설정 파일 경로 (기본값: "config/inference_config.yaml")

### 명령줄 인수를 통한 설정

명령줄 인수를 사용하여 설정을 변경할 수 있습니다:

```bash
python infer.py --model_name "다른모델" --triton_port 9012
```

지원되는 명령줄 인수:
- `--triton_server`: Triton 서버 주소
- `--triton_port`: Triton gRPC 포트
- `--model_name`: 사용할 모델 이름
- `--model_version`: 모델 버전
- `--config_path`: 설정 파일 경로

## 설정 파일 구성

`config/inference_config.yaml` 파일을 수정하여 추론 설정을 지정할 수 있습니다:

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

## 주요 모듈 설명

### config_manager.py

설정 관리를 담당하는 모듈입니다. YAML 파일, 환경 변수, 명령줄 인수를 통합하여 설정을 관리합니다.

주요 기능:
- `load_config()`: 설정 파일 로드
- `update_config_from_env()`: 환경 변수에서 설정 업데이트
- `update_config_from_args()`: 명령줄 인수에서 설정 업데이트
- `get_config()`: 통합된 설정 반환

### infer.py

추론을 실행하는 메인 스크립트입니다. Triton Inference Server와 통신하여 LLM 추론을 수행합니다.

주요 기능:
- `create_inference_request()`: 추론 요청 생성
- `stream_tokens()`: 토큰 스트리밍 처리
- `run_inference()`: 추론 실행
- `main()`: 메인 함수

## 고급 사용법

### 커스텀 프롬프트 템플릿

설정 파일의 `prompts.template` 필드를 수정하여 커스텀 프롬프트 템플릿을 사용할 수 있습니다:

```yaml
prompts:
  system_prompt: "다음 질문에 대해 전문가처럼 답변해주세요."
  user_input: "인공지능의 미래는 어떻게 될까요?"
  template: "시스템: {system_prompt}\n\n사용자: {user_input}\n\n어시스턴트: "
```

### 샘플링 파라미터 조정

설정 파일의 `parameters` 섹션을 수정하여 샘플링 파라미터를 조정할 수 있습니다:

```yaml
parameters:
  temperature: 0.5  # 낮을수록 더 결정적인 출력
  top_p: 0.95       # 누적 확률 임계값
  top_k: 50         # 상위 k개 토큰만 고려
  max_tokens: 1024  # 최대 생성 토큰 수
  stop:             # 생성 중단 토큰 목록
    - "END"
    - "STOP"
```

## 문제 해결

일반적인 문제 및 해결 방법:

1. **Triton 서버 연결 실패**:
   - Triton 서버가 실행 중인지 확인
   - 포트 번호가 올바른지 확인
   - 방화벽 설정 확인

2. **모델 로딩 실패**:
   - 모델 이름과 버전이 올바른지 확인
   - 모델 저장소가 올바르게 구성되었는지 확인
   - Triton 서버 로그 확인

3. **메모리 부족 오류**:
   - 입력 또는 출력 토큰 수 제한
   - 모델 설정에서 `gpu_memory_utilization` 값 조정
   - 더 큰 메모리를 가진 GPU 사용

4. **느린 추론 속도**:
   - 텐서 병렬화 활성화
   - 배치 처리 사용
   - 양자화된 모델 사용

## 참고 자료

- [Triton Inference Server 문서](https://github.com/triton-inference-server/server)
- [vLLM 프로젝트](https://github.com/vllm-project/vllm)
- [vLLM 지원 모델 목록](https://vllm.readthedocs.io/en/latest/models/supported_models.html)
