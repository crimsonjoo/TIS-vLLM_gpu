#!/bin/bash

set -e

# 환경 변수 설정
export PYTHONPATH=$PYTHONPATH:/workspace/app
# Triton 서버 바이너리 경로를 PATH에 추가
export PATH=$PATH:/opt/tritonserver/bin

# 필요한 디렉토리 권한 설정
chmod -R 755 /workspace
mkdir -p /workspace/logs

# 메시지 출력
echo "=============================================="
echo "TRI-vLLM 컨테이너가 성공적으로 시작되었습니다."
echo "SSH 포트: 22 (외부 포트: ${SSH_PORT:-5678})"
echo "Triton HTTP 포트: 8000 (외부 포트: ${TRITON_HTTP_PORT:-9010})"
echo "Triton gRPC 포트: 8001 (외부 포트: ${TRITON_GRPC_PORT:-9011})"
echo "Triton Metrics 포트: 8002 (외부 포트: ${TRITON_METRICS_PORT:-9012})"
echo "애플리케이션 포트: 5000 (외부 포트: ${APP_PORT:-5010})"
echo "=============================================="

# SSH 데몬을 포그라운드로 실행 (PID 1로)
exec /usr/sbin/sshd -D
