# GenAI 성능 분석기 (LLM Performance Analyzer)

이 도구는 대규모 언어 모델(LLM)의 성능을 측정하고 시각화하는 분석 도구입니다. Triton Inference Server와 vLLM 백엔드를 사용하여 LLM의 다양한 성능 지표를 수집하고 분석할 수 있습니다.

## 프로젝트 구조

```
llm_perf_analyzer/
├── run_perf_test.sh        # 성능 테스트 실행 스크립트
├── run_visualization.py    # 결과 시각화 실행 스크립트
├── utils/                  # 유틸리티 모듈
│   ├── __init__.py         # 패키지 초기화 파일
│   ├── charts.py           # 차트 생성 함수
│   ├── compare.py          # 결과 비교 함수
│   └── data.py             # 데이터 처리 함수
├── artifacts/              # 테스트 결과 저장 디렉토리 (gitignore에 포함)
└── README.md               # 이 파일
```

## 주요 기능

- **다양한 성능 지표 측정**: 지연 시간(Latency), 처리량(Throughput), GPU 활용도 등
- **결과 시각화**: 다양한 차트와 그래프를 통한 결과 시각화
- **결과 비교**: 여러 테스트 결과를 비교하여 성능 변화 분석
- **다양한 모델 및 설정 지원**: 다양한 LLM 모델과 설정에 대한 성능 테스트 지원
- **스트리밍 모드 지원**: 토큰 스트리밍 모드에서의 성능 측정 지원

## 설치 및 요구사항

이 도구는 TRI-vLLM-GPU 프로젝트의 Docker 컨테이너에 이미 포함되어 있습니다. 별도의 설치가 필요하지 않습니다.

필요한 Python 패키지:
- pandas
- matplotlib
- seaborn
- numpy
- pyyaml
- tqdm

## 사용 방법

### 성능 테스트 실행

성능 테스트를 실행하려면 다음 명령을 사용합니다:

```bash
cd /workspace/llm_perf_analyzer
./run_perf_test.sh
```

테스트 파라미터는 `run_perf_test.sh` 스크립트 내에서 직접 수정할 수 있습니다:

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

### 시각화 옵션

`run_visualization.py` 스크립트는 다음과 같은 명령줄 옵션을 지원합니다:

```
--result_dir PATH       결과 디렉토리 경로
--latest                최신 결과 사용
--model MODEL           특정 모델의 최신 결과 사용
--compare PATH [PATH]   여러 결과 비교
--names NAME [NAME]     비교 결과의 이름 지정
--output PATH           출력 이미지 저장 경로
--format FORMAT         출력 이미지 형식 (png, pdf, svg)
--dpi DPI               출력 이미지 해상도
--no-show               결과 표시하지 않고 파일로만 저장
```

## 측정 지표

성능 분석기는 다음과 같은 주요 지표를 측정합니다:

### 지연 시간 (Latency)
- **TTFT (Time To First Token)**: 첫 번째 토큰이 생성되기까지의 시간
- **토큰 간 지연 시간 (Inter-Token Latency)**: 연속된 토큰 간의 지연 시간
- **요청 지연 시간 (Request Latency)**: 전체 요청의 처리 시간

### 처리량 (Throughput)
- **초당 생성 토큰 수 (Tokens/sec)**: 초당 생성되는 토큰의 수
- **초당 처리 요청 수 (Requests/sec)**: 초당 처리되는 요청의 수

### GPU 활용도
- **GPU 사용률 (GPU Utilization)**: GPU 코어 사용률 (%)
- **GPU 메모리 사용량 (GPU Memory Usage)**: 사용된 GPU 메모리 (MB/GB)
- **GPU 전력 소비량 (GPU Power Consumption)**: GPU 전력 소비 (W)

## 결과 해석

### 성능 지표 해석

- **높은 처리량, 낮은 지연 시간**: 이상적인 성능 상태
- **높은 TTFT, 낮은 토큰 간 지연 시간**: 모델 로딩 또는 초기화에 시간이 소요됨
- **낮은 GPU 사용률, 높은 지연 시간**: CPU 병목 현상 가능성
- **높은 GPU 사용률, 높은 지연 시간**: GPU 메모리 부족 또는 모델 크기 문제

### 최적화 방법

- **텐서 병렬화 (Tensor Parallelism)**: 여러 GPU에 걸쳐 모델 분산
- **배치 처리 (Batch Processing)**: 여러 요청을 배치로 처리
- **KV 캐시 최적화 (KV Cache Optimization)**: 키-값 캐시 최적화
- **양자화 (Quantization)**: 모델 가중치 정밀도 감소

## 문제 해결

- **테스트 실패**: 모델 이름과 서비스 설정이 올바른지 확인
- **시각화 오류**: 필요한 Python 패키지가 설치되어 있는지 확인
- **낮은 성능**: GPU 메모리 사용률과 모델 설정 확인
- **결과 저장 실패**: artifacts 디렉토리의 권한 확인

## 참고 자료

- [GenAI-Perf 도구](https://github.com/triton-inference-server/genai-perf)
- [vLLM 프로젝트](https://github.com/vllm-project/vllm)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
