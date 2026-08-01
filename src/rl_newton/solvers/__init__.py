"""선형계 solver — truncated Conjugate Gradient.

구현 예정 (Stage 1)
-------------------
``conjugate_gradient.py``
    ``(H + lambda*I) p = -g`` 를 근사적으로 푼다. ``types.CGResult`` 를 반환한다.

    필수 기능:
      - residual 기반 early stopping (``||r_k|| <= tol * ||r_0||``)
      - maximum iteration budget (RL action이 지정)
      - negative curvature 탐지 (``p^T A p <= pap_eps``)
      - NaN/Inf 탐지
      - HVP 횟수, 초기/최종 residual 반환
      - optional preconditioner 인터페이스

    residual 누적은 FP32로 한다 (README §15).

구현 예정 (Stage 3)
-------------------
``line_search.py``
    step acceptance와 backtracking. loss가 비유한값이거나 허용 배수 이상
    증가하면 step을 거절하고 damping을 올린다. RL agent가 fallback을
    악용하지 못하도록 실패 penalty가 보상에 반영된다 (프로토콜 D3).

구현 예정 (Stage 5, 선택)
-------------------------
``preconditioners.py``
    identity / diagonal Adam state / Hessian diagonal / low-rank Lanczos.
"""
