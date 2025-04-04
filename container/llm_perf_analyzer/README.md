# GenAI 성능 분석기 (GenAI Performance Analyzer)

GenAI 성능 분석기는 대규모 언어 모델(LLM)의 성능을 측정, 분석 및 시각화하기 위한 종합적인 도구입니다. 이 도구는 다양한 모델 서빙 플랫폼(Triton, vLLM 등)에서 LLM의 지연 시간, 처리량, 자원 활용도 등을 측정하고 결과를 시각적으로 표현합니다.

## 주요 기능

- **성능 테스트**: 다양한 조건에서 LLM의 성능을 측정
- **결과 시각화**: 다양한 차트와 그래프를 통한 성능 데이터 시각화
- **결과 비교**: 여러 모델 또는 설정 간의 성능 비교
- **GPU 메트릭 분석**: GPU 사용률, 메모리 사용량, 전력 소비 등의 하드웨어 메트릭 분석

## 프로젝트 구조

```
llm_perf_analyzer/
├── run_perf_test.sh         # 성능 테스트 실행 스크립트
├── run_visualization.py     # 결과 시각화 실행 스크립트
├── utils/                   # 유틸리티 모듈
│   ├── __init__.py          # 패키지 초기화 파일
│   ├── charts.py            # 차트 생성 함수
│   ├── compare.py           # 결과 비교 함수
│   └── data.py              # 데이터 처리 함수
└── artifacts/               # 테스트 결과 및 아티팩트 저장 디렉토리
```

## 설치 및 설정

GenAI 성능 분석기는 Docker 컨테이너 환경에서 실행되도록 설계되었습니다.

### 필수 요구 사항

- Docker 및 Docker Compose
- NVIDIA GPU 및 NVIDIA Docker
- Python 3.8 이상

### 설치 방법

1. 저장소 클론:
   ```bash
   git clone <repository-url>
   cd TRI-vLLM-gpu
   ```

2. Docker 이미지 빌드:
   ```bash
   cd local/_docker
   docker-compose -f docker-compose.tri-vllm_gpu.yml build
   ```

3. 컨테이너 실행:
   ```bash
   docker-compose -f docker-compose.tri-vllm_gpu.yml up -d
   ```

## 사용 방법

### 성능 테스트 실행

성능 테스트를 실행하려면 `run_perf_test.sh` 스크립트를 사용합니다:

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

테스트 결과를 시각화하려면 `run_visualization.py` 스크립트를 사용합니다:

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

#### 시각화 옵션

```bash
usage: run_visualization.py [-h] (--result_dir RESULT_DIR | --compare COMPARE [COMPARE ...] | --latest | --model MODEL)
                           [--output_dir OUTPUT_DIR]
                           [--chart_types {latency,throughput,percentile,gpu,performance} [{latency,throughput,percentile,gpu,performance} ...]]
                           [--names NAMES [NAMES ...]] [--artifacts_dir ARTIFACTS_DIR]

옵션:
  -h, --help            도움말 표시
  --result_dir RESULT_DIR
                        단일 결과 디렉토리 경로
  --compare COMPARE [COMPARE ...]
                        비교할 여러 결과 디렉토리 경로
  --latest              최신 결과 시각화
  --model MODEL         특정 모델의 최신 결과 시각화
  --output_dir OUTPUT_DIR
                        출력 디렉토리 경로
  --chart_types {latency,throughput,percentile,gpu,performance} [{latency,throughput,percentile,gpu,performance} ...]
                        생성할 차트 유형
  --names NAMES [NAMES ...]
                        비교 시 사용할 이름 (--compare와 함께 사용)
  --artifacts_dir ARTIFACTS_DIR
                        아티팩트 디렉토리 경로
```

## 결과 해석

테스트 결과는 다음과 같은 디렉토리 구조로 저장됩니다:

```
artifacts/perf_<모델명>_<타임스탬프>/
├── parameters.txt      # 테스트 파라미터 정보
├── README.md           # 결과 해석 가이드
├── console_output.txt  # 테스트 실행 중 콘솔 출력
├── raw_data/           # GenAI-Perf에서 생성한 원시 데이터 파일
└── charts/             # 성능 지표 시각화 그래프
```

### 주요 성능 지표

#### 지연 시간 (Latency) 관련 지표
- **Time To First Token (TTFT)**: 요청 후 첫 번째 토큰이 생성되기까지의 시간 (ms)
- **Inter Token Latency**: 토큰 간 평균 생성 시간 (ms)
- **Request Latency**: 전체 요청 처리 시간 (ms)

#### 처리량 (Throughput) 관련 지표
- **Output Token Throughput**: 초당 생성되는 토큰 수 (tokens/sec)
- **Request Throughput**: 초당 처리할 수 있는 요청 수 (req/sec)

#### 하드웨어 활용 지표
- **GPU Utilization**: GPU 사용률 (%)
- **GPU Memory Used**: 사용된 GPU 메모리 (GB)
- **GPU Power Usage**: GPU 전력 사용량 (W)

## 유틸리티 모듈 설명

### charts.py

다양한 성능 지표를 시각화하는 차트 생성 함수들을 포함합니다:

- `create_latency_insights_chart()`: 지연 시간 인사이트 차트 생성
- `create_throughput_insights_chart()`: 처리량 인사이트 차트 생성
- `create_bar_chart()`: 일반 막대 그래프 생성
- `create_percentile_chart()`: 백분위수 차트 생성
- `create_gpu_metrics_chart()`: GPU 메트릭 차트 생성

### compare.py

여러 테스트 결과를 비교하는 함수들을 포함합니다:

- `compare_results()`: 성능 지표 비교 차트 생성
- `compare_gpu_metrics()`: GPU 메트릭 비교 차트 생성

### data.py

결과 데이터를 처리하는 함수들을 포함합니다:

- `parse_csv_file()`: CSV 파일 파싱
- `find_result_files()`: 결과 파일 찾기
- `load_result_data()`: 결과 데이터 로드
- `preprocess_metrics_data()`: 메트릭 데이터 전처리

## 문제 해결

일반적인 문제 및 해결 방법:

1. **결과 파일을 찾을 수 없음**:
   - 테스트가 성공적으로 완료되었는지 확인
   - 결과 디렉토리 경로가 올바른지 확인

2. **시각화 오류**:
   - 필요한 Python 패키지가 설치되어 있는지 확인 (pandas, matplotlib, seaborn)
   - 결과 파일 형식이 올바른지 확인

3. **GPU 메트릭 수집 실패**:
   - NVIDIA-SMI가 컨테이너 내에서 작동하는지 확인
   - 적절한 권한으로 컨테이너가 실행되고 있는지 확인

## 라이선스

이 프로젝트는 [라이선스 정보]에 따라 라이선스가 부여됩니다.

## 기여

기여는 언제나 환영합니다! 버그 리포트, 기능 요청 또는 풀 리퀘스트를 통해 프로젝트에 기여할 수 있습니다.
