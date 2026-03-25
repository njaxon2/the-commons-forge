"""Builtin function registries for Forge engine.

Aggregates all toolbox registries into a single BUILTIN_REGISTRY dict
that the evaluator can merge into its function table.
"""

# Core toolboxes
from forge.engine.builtins.elfun import ELFUN_REGISTRY
from forge.engine.builtins.general import GENERAL_REGISTRY
from forge.engine.builtins.specfun import SPECFUN_REGISTRY
from forge.engine.builtins.linalg import LINALG_REGISTRY
from forge.engine.builtins.polynomial import POLYNOMIAL_REGISTRY
from forge.engine.builtins.sets import SETS_REGISTRY
from forge.engine.builtins.special_matrix import SPECIAL_MATRIX_REGISTRY
from forge.engine.builtins.strings import STRINGS_REGISTRY
from forge.engine.builtins.time_funcs import TIME_REGISTRY
from forge.engine.builtins.ode import ODE_REGISTRY
from forge.engine.builtins.optimization import OPTIMIZATION_REGISTRY
from forge.engine.builtins.geometry import GEOMETRY_REGISTRY
from forge.engine.builtins.fileio import FILEIO_REGISTRY
from forge.engine.builtins.sparse import SPARSE_REGISTRY
from forge.engine.builtins.plotting import PLOTTING_REGISTRY

# Carry-forward toolboxes
from forge.engine.builtins.signal import SIGNAL_REGISTRY
from forge.engine.builtins.image import IMAGE_REGISTRY
from forge.engine.builtins.statistics import STATISTICS_REGISTRY
from forge.engine.builtins.audio import AUDIO_REGISTRY
from forge.engine.builtins.web import WEB_REGISTRY

# Extended toolboxes
from forge.engine.builtins.control import CONTROL_REGISTRY
from forge.engine.builtins.financial import FINANCIAL_REGISTRY
from forge.engine.builtins.comms import COMMS_REGISTRY
from forge.engine.builtins.database import DATABASE_REGISTRY
from forge.engine.builtins.parallel import PARALLEL_REGISTRY
from forge.engine.builtins.fuzzy import FUZZY_REGISTRY
from forge.engine.builtins.neural import NEURAL_REGISTRY
from forge.engine.builtins.instrument import INSTRUMENT_REGISTRY

# Optional: symbolic (requires sympy)
try:
    from forge.engine.builtins.symbolic import SYMBOLIC_REGISTRY
except ImportError:
    SYMBOLIC_REGISTRY = {}

BUILTIN_REGISTRY = {}

# Core
BUILTIN_REGISTRY.update(ELFUN_REGISTRY)
BUILTIN_REGISTRY.update(GENERAL_REGISTRY)
BUILTIN_REGISTRY.update(SPECFUN_REGISTRY)
BUILTIN_REGISTRY.update(LINALG_REGISTRY)
BUILTIN_REGISTRY.update(POLYNOMIAL_REGISTRY)
BUILTIN_REGISTRY.update(SETS_REGISTRY)
BUILTIN_REGISTRY.update(SPECIAL_MATRIX_REGISTRY)
BUILTIN_REGISTRY.update(STRINGS_REGISTRY)
BUILTIN_REGISTRY.update(TIME_REGISTRY)
BUILTIN_REGISTRY.update(ODE_REGISTRY)
BUILTIN_REGISTRY.update(OPTIMIZATION_REGISTRY)
BUILTIN_REGISTRY.update(GEOMETRY_REGISTRY)
BUILTIN_REGISTRY.update(FILEIO_REGISTRY)
BUILTIN_REGISTRY.update(SPARSE_REGISTRY)
BUILTIN_REGISTRY.update(PLOTTING_REGISTRY)

# Carry-forward
BUILTIN_REGISTRY.update(SIGNAL_REGISTRY)
BUILTIN_REGISTRY.update(IMAGE_REGISTRY)
BUILTIN_REGISTRY.update(STATISTICS_REGISTRY)
BUILTIN_REGISTRY.update(AUDIO_REGISTRY)
BUILTIN_REGISTRY.update(WEB_REGISTRY)

# Extended
BUILTIN_REGISTRY.update(CONTROL_REGISTRY)
BUILTIN_REGISTRY.update(FINANCIAL_REGISTRY)
BUILTIN_REGISTRY.update(SYMBOLIC_REGISTRY)
BUILTIN_REGISTRY.update(COMMS_REGISTRY)
BUILTIN_REGISTRY.update(DATABASE_REGISTRY)
BUILTIN_REGISTRY.update(PARALLEL_REGISTRY)
BUILTIN_REGISTRY.update(FUZZY_REGISTRY)
BUILTIN_REGISTRY.update(NEURAL_REGISTRY)
BUILTIN_REGISTRY.update(INSTRUMENT_REGISTRY)
