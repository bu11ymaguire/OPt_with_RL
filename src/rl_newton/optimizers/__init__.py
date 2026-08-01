"""Newton-CG optimizer 변종들.

모든 변종은 동일한 ``types.ControllerAction`` 을 소비한다. 즉 action space가
같고, 차이는 **누가 action을 고르는가** 뿐이다. 이 구조 덕분에 비교가 공정하다는
점이 코드 수준에서 보장된다 (프로토콜 D4).

```text
fixed        : 항상 같은 action
heuristic    : trust ratio 규칙으로 action 선택
open_loop    : progress(step/총 step) 만 보고 action 선택   <- 프로토콜 D4 추가 baseline
controlled   : RL 정책이 state를 보고 action 선택
```

구현 예정 (Stage 3)
-------------------
``fixed_newton_cg.py``
    damping / cg_budget / step_size 고정. step acceptance와 safe fallback 포함.
    ``best_static`` baseline은 이 클래스로 36개 조합을 돌려 최고를 고른 것이다.

``heuristic_newton_cg.py``
    trust ratio ``rho = actual_reduction / predicted_reduction`` 기반 damping 제어.
    ``rho < 0.25`` -> damping x3 / ``rho >= 0.75`` -> damping x0.5 (README §4.3).

``open_loop_newton_cg.py``
    상태를 보지 않는 학습된 스케줄. RL의 이득이 "적응 제어" 때문인지
    "튜닝된 스케줄" 때문인지 구분하는 결정적 baseline.

구현 예정 (Stage 4)
-------------------
``controlled_newton_cg.py``
    외부에서 주입된 정책(callable)로부터 action을 받는다. RL 환경과
    평가 스크립트가 같은 클래스를 공유한다.
"""
