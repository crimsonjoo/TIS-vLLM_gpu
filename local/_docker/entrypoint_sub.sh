#!/bin/bash

# 스크립트 오류 발생 시 중단
set -e

# 시그널 핸들러 설정 - 좀비 프로세스 방지
handle_sigchld() {
  # 좀비 프로세스 정리를 위해 wait 명령 사용
  while true; do
    wait -n > /dev/null 2>&1 || break
  done
}

# SIGCHLD 시그널 핸들러 등록
trap handle_sigchld SIGCHLD

# SSH 서비스 시작
service ssh start

# 환경 변수 설정
export PYTHONPATH=$PYTHONPATH:/workspace/app

# 필요한 디렉토리 권한 설정
chmod -R 755 /workspace

# 로그 디렉토리 생성
mkdir -p /workspace/logs

echo "=============================================="
echo "TRI-vLLM 컨테이너가 성공적으로 시작되었습니다."
echo "SSH 포트: 22 (외부 포트: ${SSH_PORT:-5678})"
echo "Triton HTTP 포트: 8000 (외부 포트: ${TRITON_HTTP_PORT:-9010})"
echo "Triton gRPC 포트: 8001 (외부 포트: ${TRITON_GRPC_PORT:-9011})"
echo "Triton Metrics 포트: 8002 (외부 포트: ${TRITON_METRICS_PORT:-9012})"
echo "애플리케이션 포트: 5000 (외부 포트: ${APP_PORT:-5010})"
echo "=============================================="

# 컨테이너가 계속 실행되도록 유지하면서 시그널을 적절히 처리
# tail -f /dev/null 대신 더 나은 방법 사용
exec /bin/bash -c "while true; do sleep 30 & wait $!; done"