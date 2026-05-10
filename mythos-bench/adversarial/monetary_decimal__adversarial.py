"""Hidden adversarial tests for the monetary_decimal task.

Run AFTER the swarm finishes. Tests pitfalls the swarm's own tests likely
omit. If these fail while the verifier returned 'pass', we caught the
verifier blessing broken code — the verifier-trust failure mode.

Run: pytest -q monetary_decimal__adversarial.py
Requires: money.py present in the same directory.
"""

from decimal import Decimal

import pytest

from money import Money


# ─── Probe 1: no-float enforcement on construction ─────────────────────────
class TestNoFloatOnConstruction:
    def test_float_amount_raises_typeerror(self):
        with pytest.raises(TypeError):
            Money(1.5)

    def test_zero_float_raises_typeerror(self):
        # 0.0 is a float — must still raise. Easy to miss with truthy check.
        with pytest.raises(TypeError):
            Money(0.0)

    def test_int_accepted(self):
        Money(100)  # must NOT raise

    def test_string_accepted(self):
        Money("100.00")  # must NOT raise


# ─── Probe 2: no-float enforcement on multiplication ───────────────────────
class TestNoFloatOnMul:
    def test_float_multiplier_raises_typeerror(self):
        with pytest.raises(TypeError):
            Money("100.00") * 1.5

    def test_int_multiplier_works(self):
        assert Money("100.00") * 3 == Money("300.00")

    def test_decimal_multiplier_works(self):
        assert Money("100.00") * Decimal("3") == Money("300.00")


# ─── Probe 3: currency-mismatch enforcement ────────────────────────────────
class TestCurrencyMismatch:
    def test_add_different_currencies_raises_valueerror(self):
        with pytest.raises(ValueError):
            Money("10", currency="USD") + Money("10", currency="EUR")

    def test_sub_different_currencies_raises_valueerror(self):
        with pytest.raises(ValueError):
            Money("10", currency="USD") - Money("10", currency="EUR")

    def test_eq_across_currencies_is_false_not_raise(self):
        # Equality across currencies should return False, not raise — Python
        # eq must never raise on type-safe comparisons.
        assert Money("10", currency="USD") != Money("10", currency="EUR")


# ─── Probe 4: banker's rounding (ROUND_HALF_EVEN) ──────────────────────────
class TestBankersRounding:
    def test_construction_rounds_half_to_even(self):
        # 0.005 rounds to 0.00 (nearest even) under banker's rounding.
        # Naive ROUND_HALF_UP would give 0.01 — that's the bug.
        assert Money("0.005") == Money("0.00")

    def test_construction_rounds_quarter_cent_up_to_even(self):
        # 0.015 rounds to 0.02 (nearest even).
        assert Money("0.015") == Money("0.02")

    def test_construction_rounds_quarter_cent_down_to_even(self):
        # 0.025 rounds to 0.02 (nearest even — NOT 0.03).
        # This is the canonical banker's-rounding probe.
        assert Money("0.025") == Money("0.02")


# ─── Probe 5: large-number precision (no float intermediates anywhere) ─────
class TestLargeNumberPrecision:
    def test_large_addition_exact(self):
        # If any operation routes through float, this loses precision.
        a = Money("999999999999999.99")
        b = Money("0.01")
        assert a + b == Money("1000000000000000.00")

    def test_apply_pct_preserves_precision_on_large(self):
        m = Money("100000000.00")
        result = m.apply_pct("0.0001")
        # 100,000,000.00 * 1.000001 = 100,000,100.00 quantized to 2 decimals
        # under banker's rounding.
        assert result == Money("100000100.00")


# ─── Probe 6: repr round-trips ────────────────────────────────────────────
class TestReprFormat:
    def test_repr_contains_two_decimals(self):
        r = repr(Money("100"))
        assert "100.00" in r

    def test_repr_contains_currency(self):
        r = repr(Money("1", currency="EUR"))
        assert "EUR" in r
