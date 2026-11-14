"""Default historical CWSA-style five-minute risk-budget backtest entrypoint.

Runs [2026-07-01, 2026-11-01), training before 2026-10-01.
Historical fill reconstruction is SCENARIO_ONLY, not actual CWSA first-known.
"""

from historical_5m_wallet_risk import main

if __name__ == "__main__":
    main()
