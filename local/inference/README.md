# TRI-vLLM-GPU 추론 시스템

이 프로젝트는 Triton Inference Server와 vLLM을 사용하여 대규모 언어 모델(LLM)의 추론을 수행하는 시스템입니다.

## 프로젝트 구조

```
/local/inference/
├── config/
│   └── inference_config.yaml  # 추론 설정 파일
├── config_manager.py          # 설정 관리 모듈
├── infer.py                   # 추론 실행 스크립트
└── README.md                  # 프로젝트 설명서
```

## 주요 기능

- **동적 설정 관리**: 다양한 소스(YAML, 환경변수, .env 파일)에서 설정을 로드하고 우선순위를 적용
- **Triton 서버 연결**: gRPC를 통한 Triton Inference Server 연결 및 통신
- **스트리밍 추론**: 토큰 단위의 스트리밍 추론 지원
- **비동기 처리**: asyncio를 활용한 비동기 추론 처리

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
python infer.py

# 환경 변수로 설정 변경
MODEL_NAME="다른모델" TRITON_GRPC_PORT=9012 python infer.py
```

### 3. 코드에서 사용

```python
import asyncio
from infer import try_request

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

## 설정 관리자 (config_manager.py)

설정 관리자는 다양한 소스에서 설정을 로드하고 관리하는 모듈입니다:

```python
from config_manager import get_config

# 특정 설정 가져오기
port = get_config("port", 9001)
model_name = get_config("model_name", "기본모델")

# 중첩된 설정 가져오기
temperature = get_config("parameters.temperature", 0.7)

# 모든 설정 가져오기
all_config = get_config()
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
