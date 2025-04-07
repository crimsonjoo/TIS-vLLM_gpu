"""
# src/inference/client.py
# Triton 추론 클라이언트 모듈
#
# 이 모듈은 Triton Inference Server와의 통신을 담당합니다.
# gRPC 프로토콜을 사용하여 Triton 서버에 추론 요청을 보내고 응답을 처리합니다.
#
# 주요 기능:
# 1. Triton 서버 URL 생성
# 2. 추론 요청 데이터 생성
# 3. 스트리밍/비스트리밍 추론 요청 처리
# 4. 응답 데이터 처리
"""

import asyncio
import numpy as np
import tritonclient.grpc.aio as grpcclient
import uuid
import json
from typing import Dict, Any, List, Optional, AsyncGenerator

from ..config import get_config


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
    # 설정 관리자에서 호스트와 포트 가져오기
    host = get_config("host", "localhost")
    port = get_config("port", 9001)
    
    if version:
        return f"{host}:{port}/v2/models/{model_name}/versions/{version}/infer"
    return f"{host}:{port}/v2/models/{model_name}/infer"


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


async def process_streaming_response(response_stream) -> AsyncGenerator[str, None]:
    """
    스트리밍 응답을 처리하는 함수.
    
    Args:
        response_stream: Triton 서버의 응답 스트림.
        
    Yields:
        str: 생성된 텍스트 토큰.
    """
    async for response in response_stream:
        result, error = response
        if error:
            print(f"Error: {error}")
            break
        
        output = result.as_numpy("text_output")
        if output is not None and len(output) > 0:
            text = output[0].decode('utf-8')
            yield text


async def process_non_streaming_response(response_stream) -> str:
    """
    비스트리밍 응답을 처리하는 함수.
    
    Args:
        response_stream: Triton 서버의 응답 스트림.
        
    Returns:
        str: 생성된 전체 텍스트.
    """
    full_text = ""
    async for response in response_stream:
        result, error = response
        if error:
            print(f"Error: {error}")
            break
        
        output = result.as_numpy("text_output")
        if output is not None and len(output) > 0:
            text = output[0].decode('utf-8')
            full_text += text
    
    return full_text
