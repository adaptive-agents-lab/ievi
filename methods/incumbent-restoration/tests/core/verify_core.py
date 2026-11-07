"""Run bounded incumbent-restoration regression tests and save their outcomes."""
from pathlib import Path
import io
import json
import sys
import unittest

import jax
import test_incumbent

here = Path(__file__).resolve().parent
suite = unittest.defaultTestLoader.loadTestsFromModule(test_incumbent)
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
output = stream.getvalue()
print(output)
(here / 'test-output.txt').write_text(output, encoding='utf-8')
record = dict(
    python=sys.version,
    jax_version=jax.__version__,
    jax_x64_enabled=jax.config.x64_enabled,
    tests_run=result.testsRun,
    failures=len(result.failures),
    errors=len(result.errors),
    success=result.wasSuccessful(),
)
(here / 'verification.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
sys.exit(0 if result.wasSuccessful() else 1)
