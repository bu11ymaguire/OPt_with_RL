# 클린 클론에서 개발 환경을 구축한다.
#
#   .\scripts\bootstrap.ps1
#
# 필요한 것: uv (https://docs.astral.sh/uv/getting-started/installation/), git
# 의존성은 uv.lock 에 해시까지 고정되어 있으므로 재현 가능하다.

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $HOME '.venvs\opt-with-rl'

Write-Host '=== rl-newton bootstrap ===' -ForegroundColor Cyan
Write-Host "repo : $repoRoot"
Write-Host "venv : $venvPath   (OneDrive 동기화 회피를 위해 저장소 밖)"
Write-Host ''

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv 를 찾을 수 없다. https://docs.astral.sh/uv/getting-started/installation/ 참조.'
}

$env:UV_PROJECT_ENVIRONMENT = $venvPath

Write-Host '[1/4] venv 생성 (Python 3.12)' -ForegroundColor Cyan
uv venv --python 3.12
if ($LASTEXITCODE -ne 0) { throw 'uv venv 실패' }

Write-Host '[2/4] 의존성 설치 (torch cu130 포함, 수 GB 다운로드)' -ForegroundColor Cyan
Push-Location $repoRoot
try {
    uv sync --all-groups --frozen
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'uv.lock 이 pyproject 와 맞지 않다. 재해결을 시도한다.' -ForegroundColor Yellow
        uv sync --all-groups
        if ($LASTEXITCODE -ne 0) { throw 'uv sync 실패' }
    }

    Write-Host '[3/4] CUDA 확인' -ForegroundColor Cyan
    uv run --no-sync python -c @'
import torch
print(f"  torch          {torch.__version__}")
print(f"  cuda available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device         {torch.cuda.get_device_name(0)}")
    print(f"  capability     {torch.cuda.get_device_capability(0)}")
else:
    print("  경고: CUDA 를 쓸 수 없다. Phase 1 이후 실험은 GPU 가 필요하다.")
'@
    if ($LASTEXITCODE -ne 0) { throw 'CUDA 확인 실패' }

    Write-Host '[4/4] Stage 0 게이트: 테스트' -ForegroundColor Cyan
    uv run --no-sync pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Stage 0 게이트 실패: 테스트가 통과하지 않았다' }
}
finally {
    Pop-Location
}

Write-Host ''
Write-Host '=== 완료 ===' -ForegroundColor Green
Write-Host '다음: . .\scripts\activate.ps1' -ForegroundColor DarkGray
Write-Host '계획: docs\experiment_protocol.md' -ForegroundColor DarkGray
