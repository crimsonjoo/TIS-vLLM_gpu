"""
# src/inference/runner.py
# 추론 실행 모듈
#
# 이 모듈은 Triton Inference Server에 추론 요청을 보내고 결과를 처리하는 메인 함수를 제공합니다.
# 설정 관리자와 Triton 클라이언트 모듈을 활용하여 LLM 추론을 수행합니다.
#
# 주요 기능:
# 1. 추론 요청 설정 준비
# 2. Triton 서버 연결 및 요청 전송
# 3. 스트리밍/비스트리밍 응답 처리
"""

import asyncio
import uuid
import tritonclient.grpc.aio as grpcclient
from typing import Dict, Any, Optional

from ..config import get_config
from .client import get_triton_url, create_request


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
            print("[스트리밍 O]")
            print()
            async for response in response_stream:
                result, error = response
                if error:
                    print(f"Error: {error}")
                    break
                
                # 7-2. 응답 텐서 처리 및 출력
                output = result.as_numpy("text_output")
                if output is not None and len(output) > 0:
                    text = output[0].decode('utf-8')
                    print(text, end="", flush=True)
        else:
            # 7-3. 비스트리밍 모드: 모든 응답을 모아서 한 번에 출력
            print("[스트리밍 X]")
            print()
            full_text = ""
            async for response in response_stream:
                result, error = response
                if error:
                    print(f"Error: {error}")
                    break
                
                # 7-4. 응답 텐서 처리 및 누적
                output = result.as_numpy("text_output")
                if output is not None and len(output) > 0:
                    text = output[0].decode('utf-8')
                    full_text += text
            
            # 7-5. 누적된 전체 텍스트 출력
            print(full_text)
            
    except Exception as e:
        print(f"추론 요청 중 오류 발생: {e}")
    finally:
        # 8. 클라이언트 연결 종료 (비동기 컨텍스트에서 안전하게 종료)
        if 'client' in locals():
            await client.close()
