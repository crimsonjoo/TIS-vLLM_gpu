"""
# main.py
# 메인 실행 스크립트
#
# 이 스크립트는 Triton Inference Server를 사용하여 LLM 추론을 실행하는 메인 진입점입니다.
# src 패키지의 모듈들을 활용하여 추론 요청을 처리합니다.
#
# 실행 방법:
# python main.py [--no-stream] [--config CONFIG_PATH]
#
# 옵션:
# --no-stream: 스트리밍 모드를 비활성화합니다. 기본값은 스트리밍 모드 활성화입니다.
# --config: 사용할 설정 파일의 경로를 지정합니다. 기본값은 config/inference_config.yaml입니다.
"""

import asyncio
import argparse
import os
from pathlib import Path
from src.config import get_config
from src.inference import try_request
from src.config.manager import ConfigManager


def parse_args():
    """
    명령줄 인수를 파싱하는 함수
    
    Returns:
        argparse.Namespace: 파싱된 인수
    """
    parser = argparse.ArgumentParser(description='Triton Inference Server를 사용한 LLM 추론')
    parser.add_argument('--no-stream', action='store_true', help='스트리밍 모드 비활성화')
    parser.add_argument('--config', type=str, help='설정 파일 경로')
    return parser.parse_args()


async def main():
    """
    메인 실행 함수
    """
    # 명령줄 인수 파싱
    args = parse_args()
    
    # 설정 파일 경로가 지정된 경우 환경 변수로 설정
    if args.config:
        # ConfigManager 인스턴스 재생성 (설정 파일 경로 지정)
        config_path = args.config
        # 상대 경로인 경우 절대 경로로 변환
        if not os.path.isabs(config_path):
            config_path = str(Path(os.getcwd()) / config_path)
        # 새로운 ConfigManager 인스턴스 생성
        config_manager = ConfigManager(yaml_config_path=config_path)
        # get_config 함수가 새 인스턴스를 사용하도록 모듈 변수 업데이트
        import src.config
        src.config._config_manager = config_manager
    
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
        "system_prompt": "당신은 반드시 한국어로 말해야 합니다.",
        "user_input": "너는 누구야?",
        "template": "{system_prompt}\nHuman: {user_input}\nAssistant: "
    })
    
    # 스트리밍 모드 설정
    stream = not args.no_stream
    
    # 추론 실행
    await try_request(
        model_name=MODEL_NAME,
        version=VERSION,
        sampling_params=PARAMETERS,
        prompts=PROMPTS,
        stream=stream
    )


if __name__ == "__main__":
    # asyncio 이벤트 루프 실행
    asyncio.run(main())
