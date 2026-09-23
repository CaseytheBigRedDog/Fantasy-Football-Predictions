"""
adopt_residual_ranges.py

Switches predict_week.py to a different way of building floor / median / ceiling.
Only run this AFTER compare_range_methods.py recommends it.

    python adopt_residual_ranges.py hybrid      # residual ranges for the top ~3%, current method for the rest
    python adopt_residual_ranges.py residual    # residual ranges for everyone
    python adopt_residual_ranges.py quantile    # back to the original method (undo)

Run it from your Fantasy-Football-Predictions folder, then:  python predict_week.py
"""
import re
import sys

method = sys.argv[1] if len(sys.argv) > 1 else "residual"
if method not in ("quantile", "residual", "hybrid"):
    print('Choose one of: quantile, residual, hybrid   (example: python adopt_residual_ranges.py hybrid)')
    sys.exit(1)

name = "range_calibration.py"
with open(name, encoding="utf-8", newline="") as f:
    s = f.read()
new_s, n = re.subn(r'RANGE_METHOD = "[a-z]+"', f'RANGE_METHOD = "{method}"', s, count=1)
if n != 1:
    print("Could not find RANGE_METHOD in range_calibration.py; check it by hand.")
    sys.exit(1)
if new_s == s:
    print(f"range_calibration.py already uses the {method} method. Nothing changed.")
else:
    with open(name, "w", encoding="utf-8", newline="") as f:
        f.write(new_s)
    print(f"range_calibration.py: ranges now use the {method} method.")
print("\nNext: python predict_week.py")
