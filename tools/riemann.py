"""
compute_riemann — real midpoint Riemann sum tool.

Evaluates the definite integral of a polynomial f over [a, b].
coefficients[i] is the coefficient of x^i:
  f(x) = coefficients[0] + coefficients[1]*x + coefficients[2]*x**2 + ...
"""

from typing import List


def compute_riemann(
    coefficients,   # type: List[float]
    a,              # type: float
    b,              # type: float
    n,              # type: int
):
    # type: (...) -> dict
    """
    Midpoint Riemann sum approximation of the definite integral of p(x) over [a, b].

    Returns a dict with:
      result   : float approximation
      n        : number of rectangles used
      interval : [a, b]
      exact    : analytically exact value (for reference)
    """
    if n <= 0:
        raise ValueError("n must be positive, got {}".format(n))
    if a >= b:
        raise ValueError("Interval must have a < b, got [{}, {}]".format(a, b))

    dx = (b - a) / n
    total = 0.0
    for i in range(n):
        mid = a + (i + 0.5) * dx
        total += _poly(coefficients, mid)
    total *= dx

    exact = _poly_integral(coefficients, a, b)

    return {
        "result": round(total, 10),
        "n": n,
        "interval": [a, b],
        "exact": round(exact, 10),
    }


def _poly(coeffs, x):
    # type: (List[float], float) -> float
    result = 0.0
    for i, c in enumerate(coeffs):
        result += c * (x ** i)
    return result


def _poly_integral(coeffs, a, b):
    # type: (List[float], float, float) -> float
    """Analytically integrates the polynomial from a to b."""
    result = 0.0
    for i, c in enumerate(coeffs):
        exp = i + 1
        result += c * (b ** exp - a ** exp) / exp
    return result


# JSON schema for Ollama tool-call declarations
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compute_riemann",
        "description": (
            "Compute a midpoint Riemann sum approximation of the definite integral "
            "of a polynomial f(x) over the interval [a, b] using n rectangles."
        ),
        "parameters": {
            "type": "object",
            "required": ["coefficients", "a", "b", "n"],
            "properties": {
                "coefficients": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Polynomial coefficients [a0, a1, a2, ...] such that "
                        "f(x) = a0 + a1*x + a2*x^2 + ..."
                    ),
                },
                "a": {
                    "type": "number",
                    "description": "Left endpoint of the integration interval.",
                },
                "b": {
                    "type": "number",
                    "description": "Right endpoint of the integration interval.",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of rectangles (subintervals) to use.",
                },
            },
        },
    },
}
