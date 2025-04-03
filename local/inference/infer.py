import asyncio
import numpy as np
import tritonclient.grpc.aio as grpcclient
import uuid
import json
import os
from pathlib import Path

# 설정 관리자 임포트
from config_manager import get_config

# 구동 원리
"""
# Ⅰ. 전체 코드 실행 흐름
1. 사용자가 특정 모델과 버전, 파라미터를 선택하여 try_request 함수를 호출
   - 예: asyncio.run(try_request("vllm_model", version=1, sampling_params=custom_params, stream=True))

2. try_request 함수 내부에서:
   a. get_triton_url 함수를 호출하여 Triton 서버 URL 생성
   b. InferenceServerClient 객체 생성 (gRPC 통신을 위한 클라이언트)
   c. 프롬프트 생성 (시스템 프롬프트 + 사용자 입력)
   d. requests_gen 비동기 제너레이터 함수 정의 (요청 생성)
   e. stream_infer 메서드를 호출하여 Triton 서버에 요청 전송
   f. 응답을 스트리밍 또는 비스트리밍 방식으로 처리

3. create_request 함수에서:
   a. 텍스트 입력 텐서 생성 및 설정
   b. 스트리밍 설정 텐서 생성 및 설정
   c. 샘플링 파라미터가 있는 경우 해당 텐서 생성 및 설정
   d. 출력 텐서 설정
   e. 최종 요청 데이터 반환

4. Triton 서버에서 추론 수행:
   a. 서버는 요청을 받아 지정된 모델로 전달
   b. 모델은 입력 텍스트와 파라미터를 기반으로 추론 수행
   c. 스트리밍 모드에서는 토큰이 생성될 때마다 응답 전송
   d. 비스트리밍 모드에서도 내부적으로는 스트리밍 방식으로 동작하지만 클라이언트에서 모든 응답을 모아서 한 번에 처리

5. 클라이언트가 서버의 응답을 받아 출력:
   a. 스트리밍 모드: 응답이 오는 대로 즉시 출력
   b. 비스트리밍 모드: 모든 응답을 모아서 한 번에 출력

# Ⅱ. 주요 함수 역할
- get_triton_url: 모델 이름과 버전을 기반으로 Triton 서버 URL 생성
- create_request: 프롬프트와 파라미터를 기반으로 Triton 서버에 보낼 요청 데이터 생성
- try_request: 전체 추론 프로세스를 관리하는 메인 함수
- requests_gen: 비동기 요청 생성기 (제너레이터)

# Ⅲ. Triton 클라이언트-서버 통신 개념
- gRPC 프로토콜: Triton은 gRPC를 사용하여 클라이언트-서버 간 통신
- 텐서 입출력: 모든 입력과 출력은 텐서 형태로 변환되어 전송
- 비동기 통신: asyncio를 사용한 비동기 방식으로 효율적인 통신 구현
- Decoupled Transaction Policy: vLLM과 같은 LLM 모델은 응답 생성 중에도 클라이언트-서버 연결 유지 필요
- 스트리밍 추론: 토큰이 생성될 때마다 점진적으로 응답을 전송하는 방식

# Ⅳ. 알아두면 좋은 Triton 관련 개념
- 모델 리포지토리: Triton 서버는 모델 리포지토리에서 모델을 로드 (config.pbtxt 파일로 설정)
- 모델 버전 관리: 동일 모델의 여러 버전을 동시에 서빙 가능
- 동적 배치 처리: 여러 요청을 배치로 묶어 처리하여 처리량 향상
- 모델 앙상블: 여러 모델을 파이프라인으로 연결하여 복잡한 워크플로우 구현 가능
- 모델 인스턴스 그룹: GPU, CPU 등 다양한 하드웨어에 모델 인스턴스 할당 가능
- 헬스 체크: 서버와 모델의 상태를 모니터링하는 기능 제공
- 메트릭 수집: 추론 성능 및 사용량 통계 수집 기능
- KServe 통합: Kubernetes 환경에서의 모델 서빙 지원
"""

def get_triton_url(model_name: str, version: int = None) -> str:
    """
    Triton 서버의 URL을 생성하는 함수.
    특정 버전이 존재하면 해당 버전의 URL을 반환하고, 없으면 최신 버전의 URL을 반환.
    
    Args:
        model_name (str): 사용할 모델의 이름.
        version (int, optional): 사용할 모델의 버전. 기본값은 None.

    Returns:
        str: Triton 서버에서 해당 모델의 추론을 요청할 URL.
    """
    # 설정 관리자에서 포트 가져오기 (기본값 9001)
    port = get_config("port", 9001)
    
    if version:
        return f"localhost:{port}/v2/models/{model_name}/versions/{version}/infer"
    return f"localhost:{port}/v2/models/{model_name}/infer"

def create_request(prompt: str, request_id: str, model_name: str, sampling_params: dict = None, stream: bool = True) -> dict:
    """
    Triton Inference Server에 보낼 요청을 생성하는 함수.
    
    Args:
        prompt (str): 입력 프롬프트 (질문).
        request_id (str): 요청을 식별하기 위한 고유 ID.
        model_name (str): 사용할 모델의 이름.
        sampling_params (dict, optional): 샘플링 파라미터. 기본값은 None.
        stream (bool, optional): 스트리밍 모드 사용 여부. 기본값은 True.

    Returns:
        dict: Triton에 보낼 요청 데이터.
    """
    input_tensor = grpcclient.InferInput("text_input", [1], "BYTES")
    input_tensor.set_data_from_numpy(np.array([prompt.encode("utf-8")]))

    stream_setting = grpcclient.InferInput("stream", [1], "BOOL")
    stream_setting.set_data_from_numpy(np.array([stream]))

    inputs = [input_tensor, stream_setting]

    if sampling_params:
        sampling_parameters_data = np.array([json.dumps(sampling_params).encode("utf-8")], dtype=np.object_)
        sampling_parameters_input = grpcclient.InferInput("sampling_parameters", [1], "BYTES")
        sampling_parameters_input.set_data_from_numpy(sampling_parameters_data)
        inputs.append(sampling_parameters_input)

    output = grpcclient.InferRequestedOutput("text_output")
    
    return {
        "model_name": model_name,
        "inputs": inputs,
        "outputs": [output],
        "request_id": request_id
    }

async def try_request(model_name: str = None, version: int = None, sampling_params: dict = None, prompts: dict = None, stream: bool = True):
    """
    Triton 서버에 추론 요청을 보내는 함수.
    
    Args:
        model_name (str, optional): 사용할 모델의 이름. 기본값은 None (설정에서 가져옴).
        version (int, optional): 사용할 모델의 버전. 기본값은 None (설정에서 가져옴).
        sampling_params (dict, optional): 샘플링 파라미터. 기본값은 None (설정에서 가져옴).
        prompts (dict, optional): 프롬프트 설정. 기본값은 None (설정에서 가져옴).
        stream (bool): 스트리밍 모드 사용 여부. 기본값은 True.
        
    Returns:
        None
    """
    try:
        # 설정에서 기본값 가져오기
        if model_name is None:
            model_name = get_config("model_name", "vllm_model")
        
        if version is None:
            version = get_config("version", 1)
        
        if sampling_params is None:
            sampling_params = get_config("parameters", {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 100,
                "stop": ["Human:", "Assistant:"]
            })
        
        # 1. Triton 서버 URL 생성
        url = get_triton_url(model_name, version)
        # 2. gRPC 클라이언트 생성 (비동기 통신을 위한 객체)
        client = grpcclient.InferenceServerClient(url=url)
        
        # 3. 프롬프트 설정 로드 및 처리
        # 3-1. prompts가 None이면 설정에서 가져오기
        if prompts is None:
            prompts = get_config("prompts", {
                "system_prompt": "당신은 언제나 반말로 대화를 해야 합니다.",
                "user_input": "너는 누구야?",
                "template": "{system_prompt}\nHuman: {user_input}\nAssistant: "
            })
        
        # 3-2. 프롬프트 구성 요소 추출    
        system_prompt = prompts.get("system_prompt")
        user_input = prompts.get("user_input")
        template = prompts.get("template")
        
        # 3-3. 템플릿에 값 채우기 (최종 프롬프트 생성)
        prompt = template.format(system_prompt=system_prompt, user_input=user_input)
        
        # 4. 요청 ID 생성 (각 요청을 고유하게 식별하기 위한 UUID)
        request_id = str(uuid.uuid4())
        
        # 5. 비동기 요청 생성기 함수 정의
        async def requests_gen():
            # 5-1. 요청 데이터 생성 (텐서 및 메타데이터 포함)
            request = create_request(prompt, request_id, model_name, sampling_params, stream)
            # 5-2. 요청 데이터 yield (비동기 스트림으로 전송)
            yield request
        
        # 6. 스트리밍 추론 요청 전송 (vLLM 모델은 decoupled transaction policy를 사용하므로 항상 stream_infer 사용)
        response_stream = client.stream_infer(requests_gen())
        
        # 7. 응답 처리 (스트리밍 모드에 따라 다르게 처리)
        if stream:
            # 7-1. 스트리밍 모드: 응답이 오는 대로 즉시 출력
            async for response in response_stream:
                result, error = response
                if error:
                    print(f"Error: {error}")
                    break
                
                # 7-2. 응답 텐서 처리 및 출력
                output = result.as_numpy("text_output")
                if output is not None and len(output) > 0:
                    text = output[0].decode('utf-8')
                    print(text, end='', flush=True)
            print()  # 줄바꿈
        else:
            # 7-3. 비스트리밍 모드: 모든 응답을 모아서 한 번에 출력
            full_response = ""
            # 프롬프트 접두사 (이 부분을 응답에서 제거하기 위해 사용)
            prompt_prefix = prompt
            
            async for response in response_stream:
                result, error = response
                if error:
                    print(f"Error: {error}")
                    break
                
                # 7-4. 응답 텐서 처리 및 누적
                output = result.as_numpy("text_output")
                if output is not None and len(output) > 0:
                    text = output[0].decode('utf-8')
                    # 누적된 응답에 추가
                    full_response = text
            
            # 7-5. 프롬프트 부분 제거 후 실제 생성된 텍스트만 출력
            if full_response.startswith(prompt_prefix):
                # 프롬프트 부분을 제거하고 실제 생성된 텍스트만 출력
                generated_text = full_response[len(prompt_prefix):]
                print(generated_text)
            else:
                # 프롬프트가 포함되지 않은 경우 전체 응답 출력
                print(full_response)
            
    except Exception as e:
        # 8. 예외 처리 (서버 예외와 일반 예외를 구분하여 처리)
        if isinstance(e, grpcclient.InferenceServerException):
            print(f"Inference Server Exception: {e}")
        else:
            print(f"Error occurred: {e}")

if __name__ == "__main__":
    # 설정 관리자에서 설정 가져오기
    MODEL_NAME = get_config("model_name", "vllm_model")
    VERSION = get_config("version", 1)
    PARAMETERS = get_config("parameters", {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 100,
        "stop": ["Human:", "Assistant:"]
    })
    PROMPTS = get_config("prompts", {
        "system_prompt": "당신은 언제나 반말로 대화를 해야 합니다.",
        "user_input": "너는 누구야?",
        "template": "{system_prompt}\nHuman: {user_input}\nAssistant: "
    })
    
    # 스트리밍 O
    print("\n[스트리밍 O]")
    asyncio.run(try_request(model_name=MODEL_NAME, version=VERSION, sampling_params=PARAMETERS, prompts=PROMPTS, stream=True))

    # 스트리밍 X
    # print("[스트리밍 X] ----------------")
    # asyncio.run(try_request(model_name=MODEL_NAME, version=VERSION, sampling_params=PARAMETERS, prompts=PROMPTS, stream=False))
