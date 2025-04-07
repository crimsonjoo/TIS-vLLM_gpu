"""
# config/manager.py
# 설정 관리 모듈
#
# 이 모듈은 다양한 소스(YAML 파일, 환경 변수, .env 파일 등)에서 설정 값을 로드하고 관리합니다.
# 설정 우선순위: 환경 변수 > .env 파일 > YAML 파일 > 기본값
#
# 주요 기능:
# 1. YAML 설정 파일 로드
# 2. 환경 변수 로드
# 3. .env 파일 로드
# 4. 설정 값 병합 및 우선순위 적용
# 5. 설정 값 접근 인터페이스 제공
"""

import os
import yaml
import dotenv
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ConfigManager:
    """
    설정 관리 클래스
    
    다양한 소스에서 설정을 로드하고 관리하는 클래스입니다.
    설정 우선순위: 환경 변수 > .env 파일 > YAML 파일 > 기본값
    
    Attributes:
        config (dict): 모든 설정 값을 담고 있는 딕셔너리
        yaml_config_path (str): YAML 설정 파일 경로
        env_file_path (str): .env 파일 경로
    """
    
    def __init__(self, yaml_config_path: Optional[str] = None, env_file_path: Optional[str] = None):
        """
        ConfigManager 초기화
        
        Args:
            yaml_config_path (str, optional): YAML 설정 파일 경로. 기본값은 None.
            env_file_path (str, optional): .env 파일 경로. 기본값은 None.
        """
        self.config = {}
        
        # 기본 경로 설정
        self.yaml_config_path = yaml_config_path
        if yaml_config_path is None:
            # 현재 스크립트 디렉토리의 config/inference_config.yaml을 기본값으로 사용
            script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            self.yaml_config_path = str(script_dir.parent.parent / "config" / "inference_config.yaml")
        
        self.env_file_path = env_file_path
        if env_file_path is None:
            # 상위 디렉토리의 _docker/.env를 기본값으로 사용
            script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            self.env_file_path = str(script_dir.parent.parent.parent / "_docker" / ".env")
        
        # 설정 로드
        self._load_config()
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """
        YAML 설정 파일 로드
        
        Returns:
            dict: YAML 설정 파일에서 로드한 설정 값
        """
        try:
            with open(self.yaml_config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file) or {}
        except Exception as e:
            print(f"YAML 설정 파일 로드 중 오류 발생: {e}")
            return {}
    
    def _load_env_file(self) -> Dict[str, str]:
        """
        .env 파일 로드
        
        Returns:
            dict: .env 파일에서 로드한 환경 변수
        """
        try:
            # .env 파일이 존재하는지 확인
            if os.path.exists(self.env_file_path):
                # .env 파일 로드
                env_vars = dotenv.dotenv_values(self.env_file_path)
                return dict(env_vars)
            return {}
        except Exception as e:
            print(f".env 파일 로드 중 오류 발생: {e}")
            return {}
    
    def _load_environment_variables(self) -> Dict[str, str]:
        """
        환경 변수 로드
        
        Returns:
            dict: 시스템 환경 변수
        """
        # 필요한 환경 변수만 추출
        env_vars = {}
        
        # Triton 서버 관련 환경 변수
        for key in ['TRITON_HTTP_PORT', 'TRITON_GRPC_PORT', 'TRITON_METRICS_PORT']:
            if key in os.environ:
                env_vars[key] = os.environ[key]
        
        # 기타 설정 관련 환경 변수
        for key in ['MODEL_NAME', 'MODEL_VERSION', 'CONFIG_FILE']:
            if key in os.environ:
                env_vars[key] = os.environ[key]
        
        return env_vars
    
    def _merge_configs(self, yaml_config: Dict[str, Any], env_file: Dict[str, str], env_vars: Dict[str, str]) -> Dict[str, Any]:
        """
        설정 값 병합 및 우선순위 적용
        
        Args:
            yaml_config (dict): YAML 설정 파일에서 로드한 설정 값
            env_file (dict): .env 파일에서 로드한 환경 변수
            env_vars (dict): 시스템 환경 변수
            
        Returns:
            dict: 병합된 설정 값
        """
        # 기본 설정 (가장 낮은 우선순위)
        default_config = {
            "host": "localhost",
            "port": 8001,
            "model_name": "vllm_model",
            "version": 1,
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 100,
                "stop": ["Human:", "Assistant:"]
            },
            "prompts": {
                "system_prompt": "당신은 반드시 한국어로 말해야 합니다.",
                "user_input": "너는 누구야?",
                "template": "{system_prompt}\nHuman: {user_input}\nAssistant: "
            }
        }
        
        # 설정 병합 (우선순위: 기본값 < YAML < .env 파일 < 환경 변수)
        merged_config = default_config.copy()
        
        # YAML 설정 병합
        if yaml_config:
            merged_config.update(yaml_config)
        
        # .env 파일 설정 변환 및 병합
        if env_file:
            # port 설정 (.env 파일의 TRITON_GRPC_PORT)
            if 'TRITON_GRPC_PORT' in env_file:
                merged_config['port'] = int(env_file['TRITON_GRPC_PORT'])
        
        # 환경 변수 설정 변환 및 병합
        if env_vars:
            # port 설정 (환경 변수의 TRITON_GRPC_PORT)
            if 'TRITON_GRPC_PORT' in env_vars:
                merged_config['port'] = int(env_vars['TRITON_GRPC_PORT'])
            
            # 모델 이름 설정 (환경 변수의 MODEL_NAME)
            if 'MODEL_NAME' in env_vars:
                merged_config['model_name'] = env_vars['MODEL_NAME']
            
            # 모델 버전 설정 (환경 변수의 MODEL_VERSION)
            if 'MODEL_VERSION' in env_vars:
                merged_config['version'] = int(env_vars['MODEL_VERSION'])
        
        return merged_config
    
    def _load_config(self) -> None:
        """
        모든 설정 소스에서 설정 로드 및 병합
        """
        # 각 소스에서 설정 로드
        yaml_config = self._load_yaml_config()
        env_file = self._load_env_file()
        env_vars = self._load_environment_variables()
        
        # 설정 병합
        self.config = self._merge_configs(yaml_config, env_file, env_vars)
    
    def get(self, key: str = None, default: Any = None) -> Any:
        """
        설정 값 가져오기
        
        Args:
            key (str, optional): 설정 키. None이면 모든 설정 반환. 기본값은 None.
            default (Any, optional): 기본값. 키가 없을 경우 반환됨. 기본값은 None.
            
        Returns:
            Any: 설정 값 또는 모든 설정
        """
        # 키가 None이면 모든 설정 반환
        if key is None:
            return self.config
        
        # 중첩된 키 처리 (예: "parameters.temperature")
        if '.' in key:
            parts = key.split('.')
            current = self.config
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current
        
        # 일반 키 처리
        return self.config.get(key, default)


# 싱글톤 인스턴스 생성
_config_manager = ConfigManager()

def get_config(key: str = None, default: Any = None) -> Union[Any, Dict[str, Any]]:
    """
    설정 값 가져오기 (편의 함수)
    
    Args:
        key (str, optional): 설정 키. None이면 모든 설정 반환. 기본값은 None.
        default (Any, optional): 기본값. 키가 없을 경우 반환됨. 기본값은 None.
        
    Returns:
        Union[Any, Dict[str, Any]]: 설정 값 또는 모든 설정
    """
    return _config_manager.get(key, default)


if __name__ == "__main__":
    # 테스트 코드
    print("전체 설정:")
    print(get_config())
    
    print("\n호스트 설정:")
    print(get_config("host"))
    
    print("\n온도 설정:")
    print(get_config("parameters.temperature"))
    
    print("\n존재하지 않는 설정 (기본값 사용):")
    print(get_config("존재하지_않는_키", "기본값"))
