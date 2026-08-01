# 이 프로젝트의 개발 셸을 준비한다. 반드시 dot-source 로 실행한다.
#
#   . .\scripts\activate.ps1
#
# venv 를 저장소 밖에 두는 이유
# ------------------------------
# 저장소가 OneDrive 동기화 폴더 안에 있다. torch cu130 은 CUDA 런타임 DLL 까지
# 합쳐 설치 후 5~8 GB, 파일 수만 수만 개다. 이를 OneDrive 안에 두면 동기화가
# 폭주하고, 설치 중 파일 락 충돌로 uv sync 가 실패할 수 있다.
# 코드와 pyproject 는 OneDrive 에 두고 venv 만 밖으로 뺀다.

$ErrorActionPreference = 'Stop'

$venvPath = Join-Path $HOME '.venvs\opt-with-rl'
$env:UV_PROJECT_ENVIRONMENT = $venvPath

if (-not (Test-Path $venvPath)) {
    Write-Host "venv 가 없다: $venvPath" -ForegroundColor Yellow
    Write-Host "먼저 .\scripts\bootstrap.ps1 을 실행하라." -ForegroundColor Yellow
    return
}

$activate = Join-Path $venvPath 'Scripts\Activate.ps1'
& $activate

Write-Host "UV_PROJECT_ENVIRONMENT = $env:UV_PROJECT_ENVIRONMENT" -ForegroundColor DarkGray
Write-Host "uv run pytest -q            # Stage 0 게이트" -ForegroundColor DarkGray
