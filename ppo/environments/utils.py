def piecewise_function(x, a, b):
    """
    Computes the piecewise function f(x):
    
    For 0 <= x <= a:
        f(x) = x^2 / n
    For a <= x <= b:
        f(x) = a^2/n + (2a/n)(x - a)
    
    where n = a(2b - a) ensures:
      - Continuity at x = a,
      - f(b) = 1, and
      - Matching slopes (f'(a) = 2a/n).
    
    Parameters:
        x : float
            The input value (must satisfy 0 <= x <= b).
        a : float
            The breakpoint between quadratic and linear segments.
        b : float
            The endpoint where the function reaches 1.
    
    Returns:
        float: the function value at x.
    
    Raises:
        ValueError: if x is not in the range [0, b].
    """
    if not (0 <= x):
        raise ValueError("x should be in the range [0, b]")
    
    # Calculate n from the condition f(b)=1.
    n = a * (2 * b - a)
    
    if x <= a:
        return x**2 / n
    elif x <= b:
        # Linear segment with slope 2a/n starting at x = a.
        return (a**2 / n) + (2 * a / n) * (x - a)
    else:
        return 1.0

if __name__ == "__main__":
    # Example usage:
    a = 6
    b = 10
    for x in [0, a, (a+b)/2, b, b+1]:
        print(f"f({x}) = {piecewise_function(x, a, b)}")
