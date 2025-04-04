#!/bin/bash
# =========================================================================
# GenAI 성능 테스트 실행 스크립트 (run_perf_test.sh)
# =========================================================================
# 
# 이 스크립트는 GenAI-Perf 도구를 사용하여 LLM 모델의 성능을 테스트하고
# 결과를 저장 및 시각화하는 전체 프로세스를 자동화합니다.
#
# 실행 방법:
#   ./run_perf_test.sh
#
# 실행 흐름:
# 1. 테스트 파라미터 설정
# 2. 결과 저장을 위한 디렉토리 생성
# 3. 테스트 파라미터 및 README 파일 생성
# 4. GenAI-Perf 도구를 사용하여 성능 테스트 실행
# 5. 결과 파일 수집 및 정리
# 6. 결과 시각화 실행
#
# 주요 출력:
# - 성능 테스트 결과 (CSV 파일)
# - 시각화 그래프 (PNG 파일)
# - 테스트 파라미터 및 설명 문서
# =========================================================================

# =========================================================================
# 1. 테스트 파라미터 설정
# =========================================================================
# 아래 변수들을 수정하여 테스트 조건을 조정할 수 있습니다.

# 모델 및 서비스 설정
MODEL_NAME="exaone-deep-32B"    # 테스트할 모델 이름
SERVICE_KIND="triton"           # 서비스 종류 (triton, tgi, vllm 등)
BACKEND="vllm"                  # 백엔드 (vllm, tensorrtllm 등)

# 입력 및 출력 토큰 설정
INPUT_TOKENS_MEAN=200           # 입력 프롬프트의 평균 토큰 수
INPUT_TOKENS_STDDEV=0           # 입력 토큰 수의 표준 편차 (0=고정 크기)
OUTPUT_TOKENS_MEAN=100          # 생성할 출력 토큰의 평균 수
OUTPUT_TOKENS_STDDEV=0          # 출력 토큰 수의 표준 편차 (0=고정 크기)

# 테스트 설정
REQUEST_COUNT=5                 # 성능 측정에 사용할 요청 수
WARMUP_REQUEST_COUNT=2          # 워밍업에 사용할 요청 수
STREAMING=true                  # 토큰 스트리밍 사용 여부 (true/false)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)  # 테스트 실행 시간 (자동 생성)

# =========================================================================
# 2. 결과 저장 디렉토리 설정
# =========================================================================
# 결과 파일이 저장될 디렉토리 구조를 설정합니다.

BASE_DIR="/workspace/llm_perf_analyzer"  # 기본 작업 디렉토리
ARTIFACTS_DIR="${BASE_DIR}/artifacts"    # 아티팩트 저장 디렉토리
RESULT_DIR="${ARTIFACTS_DIR}/perf_${MODEL_NAME}_${TIMESTAMP}"  # 결과 저장 디렉토리

# 디렉토리 생성
mkdir -p "${RESULT_DIR}"
mkdir -p "${RESULT_DIR}/raw_data"

# =========================================================================
# 3. 테스트 파라미터 및 README 파일 생성
# =========================================================================
# 테스트 조건과 결과 해석 방법을 문서화합니다.

# 파라미터 저장
cat > "${RESULT_DIR}/parameters.txt" << EOF
# 성능 테스트 파라미터
모델 이름: ${MODEL_NAME}
서비스 종류: ${SERVICE_KIND}
백엔드: ${BACKEND}
평균 입력 토큰 수: ${INPUT_TOKENS_MEAN}
입력 토큰 표준 편차: ${INPUT_TOKENS_STDDEV}
평균 출력 토큰 수: ${OUTPUT_TOKENS_MEAN}
출력 토큰 표준 편차: ${OUTPUT_TOKENS_STDDEV}
요청 수: ${REQUEST_COUNT}
워밍업 요청 수: ${WARMUP_REQUEST_COUNT}
스트리밍 모드: ${STREAMING}
테스트 시간: ${TIMESTAMP}
EOF

# README.md 파일 생성
cat > "${RESULT_DIR}/README.md" << EOF
# GenAI-Perf 성능 테스트 결과

## 테스트 정보
- **모델**: ${MODEL_NAME}
- **서비스**: ${SERVICE_KIND}
- **백엔드**: ${BACKEND}
- **테스트 시간**: ${TIMESTAMP}

## 파라미터 설명

| 파라미터 | 값 | 설명 |
|----------|------|------|
| 모델 이름 | ${MODEL_NAME} | 테스트한 LLM 모델 |
| 서비스 종류 | ${SERVICE_KIND} | 모델 서빙에 사용된 서비스 (Triton, TGI 등) |
| 백엔드 | ${BACKEND} | 모델 서빙에 사용된 백엔드 (vLLM, TensorRT-LLM 등) |
| 평균 입력 토큰 수 | ${INPUT_TOKENS_MEAN} | 테스트에 사용된 입력 프롬프트의 평균 토큰 수 |
| 입력 토큰 표준 편차 | ${INPUT_TOKENS_STDDEV} | 입력 토큰 수의 표준 편차 (0이면 모든 입력이 동일한 토큰 수) |
| 평균 출력 토큰 수 | ${OUTPUT_TOKENS_MEAN} | 모델이 생성할 출력 토큰의 평균 수 |
| 출력 토큰 표준 편차 | ${OUTPUT_TOKENS_STDDEV} | 출력 토큰 수의 표준 편차 (0이면 모든 출력이 동일한 토큰 수) |
| 요청 수 | ${REQUEST_COUNT} | 성능 측정에 사용된 요청 수 |
| 워밍업 요청 수 | ${WARMUP_REQUEST_COUNT} | 성능 측정 전 워밍업에 사용된 요청 수 |
| 스트리밍 모드 | ${STREAMING} | 토큰 스트리밍 사용 여부 |

## 주요 성능 지표 설명

### 지연 시간 (Latency) 관련 지표
- **Time To First Token (TTFT)**: 요청 후 첫 번째 토큰이 생성되기까지의 시간 (ms). 사용자 체감 응답성에 직접적인 영향을 미치는 중요 지표입니다.
- **Inter Token Latency**: 토큰 간 평균 생성 시간 (ms). 낮을수록 더 부드러운 텍스트 생성 경험을 제공합니다.
- **Request Latency**: 전체 요청 처리 시간 (ms). 입력부터 모든 출력 토큰 생성까지의 총 시간입니다.

### 처리량 (Throughput) 관련 지표
- **Output Token Throughput**: 초당 생성되는 토큰 수 (tokens/sec). 높을수록 더 빠른 생성 속도를 의미합니다.
- **Request Throughput**: 초당 처리할 수 있는 요청 수 (req/sec). 시스템의 전체 처리 용량을 나타냅니다.

### 하드웨어 활용 지표
- **GPU Utilization**: GPU 사용률 (%). 모델이 GPU 리소스를 얼마나 효율적으로 활용하는지 보여줍니다.
- **GPU Memory Used**: 사용된 GPU 메모리 (GB). 모델의 메모리 요구사항을 나타냅니다.
- **GPU Power Usage**: GPU 전력 사용량 (W). 모델 실행 시 전력 효율성을 나타냅니다.

## 결과 디렉토리 구조
- **parameters.txt**: 테스트 파라미터 정보
- **console_output.txt**: 테스트 실행 중 콘솔 출력
- **raw_data/**: GenAI-Perf에서 생성한 원시 데이터 파일
- **charts/**: 성능 지표 시각화 그래프

## 시각화 결과 해석 방법
시각화 결과는 \`charts/\` 디렉토리에서 확인할 수 있습니다:

1. **지연 시간 인사이트 차트**: 다양한 지연 시간 지표(TTFT, 토큰 간 지연 등)의 분포와 관계를 보여줍니다. 
   이를 통해 어떤 지연 시간 요소가 전체 성능에 가장 큰 영향을 미치는지 파악할 수 있습니다.

2. **처리량 인사이트 차트**: 토큰 및 요청 처리량과 시퀀스 길이 간의 관계를 보여줍니다. 
   시스템의 전체 처리 용량과 효율성을 평가하는 데 유용합니다.

3. **백분위수 차트**: 지연 시간 지표의 백분위수 분포를 보여줍니다. 
   p50(중앙값), p90, p99 값을 통해 성능의 일관성과 최악의 경우 시나리오를 평가할 수 있습니다.

4. **GPU 메트릭 차트**: GPU 사용률, 메모리 사용량, 전력 소비를 보여줍니다. 
   하드웨어 리소스가 효율적으로 활용되고 있는지 평가하는 데 유용합니다.

## 결과 해석 및 최적화 방향
- **TTFT가 높은 경우**: 모델 초기화 또는 컨텍스트 처리 최적화가 필요할 수 있습니다.
  - KV 캐시 최적화, 모델 웜업 개선, 컨텍스트 처리 알고리즘 최적화 등을 고려하세요.

- **Inter Token Latency가 높은 경우**: 토큰 생성 로직 또는 하드웨어 활용 최적화가 필요할 수 있습니다.
  - 배치 크기 조정, 계산 정밀도 최적화(FP16/BF16), 커널 최적화 등을 고려하세요.

- **GPU 사용률이 낮은 경우**: 배치 크기 조정 또는 모델 병렬화 전략 검토가 필요할 수 있습니다.
  - 더 큰 배치 크기, 더 효율적인 텐서 병렬화, 파이프라인 병렬화 등을 고려하세요.

- **메모리 사용량이 높은 경우**: 모델 양자화 또는 KV 캐시 최적화를 고려할 수 있습니다.
  - INT8/INT4 양자화, 주의력 메커니즘 최적화, KV 캐시 압축 등을 고려하세요.

- **처리량이 낮은 경우**: 병렬 처리 전략 또는 하드웨어 구성 최적화가 필요할 수 있습니다.
  - 더 많은 GPU 사용, 더 효율적인 GPU 간 통신, 연속 배치 처리 등을 고려하세요.
EOF

# =========================================================================
# 4. GenAI-Perf 도구를 사용하여 성능 테스트 실행
# =========================================================================
# GenAI-Perf 명령을 실행하여 성능 테스트를 수행합니다.

echo "성능 테스트 실행 중..."
{
  if [ "$STREAMING" = true ]; then
    # 스트리밍 모드 활성화
    genai-perf profile -m ${MODEL_NAME} \
      --service-kind ${SERVICE_KIND} \
      --backend ${BACKEND} \
      --synthetic-input-tokens-mean ${INPUT_TOKENS_MEAN} \
      --synthetic-input-tokens-stddev ${INPUT_TOKENS_STDDEV} \
      --output-tokens-mean ${OUTPUT_TOKENS_MEAN} \
      --output-tokens-stddev ${OUTPUT_TOKENS_STDDEV} \
      --output-tokens-mean-deterministic \
      --streaming \
      --request-count ${REQUEST_COUNT} \
      --warmup-request-count ${WARMUP_REQUEST_COUNT}
  else
    # 스트리밍 모드 비활성화
    genai-perf profile -m ${MODEL_NAME} \
      --service-kind ${SERVICE_KIND} \
      --backend ${BACKEND} \
      --synthetic-input-tokens-mean ${INPUT_TOKENS_MEAN} \
      --synthetic-input-tokens-stddev ${INPUT_TOKENS_STDDEV} \
      --output-tokens-mean ${OUTPUT_TOKENS_MEAN} \
      --output-tokens-stddev ${OUTPUT_TOKENS_STDDEV} \
      --output-tokens-mean-deterministic \
      --request-count ${REQUEST_COUNT} \
      --warmup-request-count ${WARMUP_REQUEST_COUNT}
  fi
} > "${RESULT_DIR}/console_output.txt" 2>&1

# =========================================================================
# 5. 결과 파일 수집 및 정리
# =========================================================================
# GenAI-Perf가 생성한 결과 파일을 정리된 디렉토리로 복사합니다.

echo "결과 파일 복사 중..."
cp -r ${BASE_DIR}/artifacts/${MODEL_NAME}-${SERVICE_KIND}-${BACKEND}-concurrency1/* "${RESULT_DIR}/raw_data/"

# =========================================================================
# 6. 결과 시각화 실행
# =========================================================================
# 결과 데이터를 시각화하여 차트를 생성합니다.

echo "결과 시각화 중..."
# 새로운 시각화 스크립트 사용
python3 ${BASE_DIR}/run_visualization.py --result_dir "${RESULT_DIR}"

echo "성능 테스트 완료. 결과는 ${RESULT_DIR} 디렉토리에 저장되었습니다."
echo "시각화 결과는 ${RESULT_DIR}/charts 디렉토리에서 확인할 수 있습니다."
echo ""
echo "주요 파일:"
echo "- 테스트 파라미터: ${RESULT_DIR}/parameters.txt"
echo "- 결과 설명: ${RESULT_DIR}/README.md"
echo "- 콘솔 출력: ${RESULT_DIR}/console_output.txt"
echo "- 원시 데이터: ${RESULT_DIR}/raw_data/"
echo "- 시각화 차트: ${RESULT_DIR}/charts/"
