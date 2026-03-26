"""String Toolbox for Forge.

Provides 34 string manipulation functions compatible with Octave/MATLAB
string operations. All string functions operate on ForgeChar arrays.

Backend: Python builtins + re module.
"""

from __future__ import annotations

import re
import numpy as np

from forge.engine.types import ForgeArray, _unwrap
from forge.engine.containers import ForgeChar


# ── Helpers ──────────────────────────────────────────────────────

def _wrap(value):
    """Wrap a numpy array as a ForgeArray."""
    if isinstance(value, np.ndarray):
        return ForgeArray(value)
    return value


def _scalar(value):
    """Extract scalar from 0-d or single-element array."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.size == 1:
            return value.item()
    return value


# ── Toolbox function registry ───────────────────────────────────
STRINGS_REGISTRY: dict[str, callable] = {}


def _tb(name: str | None = None):
    """Local decorator to register a toolbox function."""
    def decorator(func):
        fn_name = name or func.__name__
        STRINGS_REGISTRY[fn_name] = func
        return func
    return decorator


# =====================================================================
# Base Conversion (String <-> Number)
# =====================================================================

@_tb("base2dec")
def forge_base2dec(s, base):
    """Convert string representation of number in given base to decimal.

    base2dec('101', 2)  -> 5
    base2dec('FF', 16)  -> 255
    """
    return int(str(s), int(base))


@_tb("bin2dec")
def forge_bin2dec(s):
    """Convert binary string to decimal number.

    bin2dec('101') -> 5
    """
    return int(str(s), 2)


@_tb("dec2base")
def forge_dec2base(d, base, ndigits=None):
    """Convert decimal integer to string in given base.

    dec2base(255, 16) -> 'FF'
    """
    d = int(d)
    base = int(base)
    if base < 2 or base > 36:
        raise ValueError("BASE must be between 2 and 36")
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if d == 0:
        result = "0"
    else:
        negative = d < 0
        d = abs(d)
        result = ""
        while d > 0:
            result = digits[d % base] + result
            d //= base
        if negative:
            result = "-" + result
    if ndigits is not None:
        ndigits = int(ndigits)
        if len(result) < ndigits:
            result = "0" * (ndigits - len(result)) + result
    return result


@_tb("dec2bin")
def forge_dec2bin(d, ndigits=None):
    """Convert decimal to binary string.

    dec2bin(5) -> '101'
    """
    d = int(d)
    result = bin(d)[2:] if d >= 0 else "-" + bin(d)[3:]
    if ndigits is not None:
        ndigits = int(ndigits)
        if len(result) < ndigits:
            result = "0" * (ndigits - len(result)) + result
    return result


@_tb("dec2hex")
def forge_dec2hex(d, ndigits=None):
    """Convert decimal to hexadecimal string.

    dec2hex(255) -> 'FF'
    """
    d = int(d)
    result = format(d, "X")
    if ndigits is not None:
        ndigits = int(ndigits)
        if len(result) < ndigits:
            result = "0" * (ndigits - len(result)) + result
    return result


@_tb("hex2dec")
def forge_hex2dec(s):
    """Convert hexadecimal string to decimal number.

    hex2dec('FF') -> 255
    """
    return int(str(s), 16)


# =====================================================================
# String Construction / Whitespace
# =====================================================================

@_tb("blanks")
def forge_blanks(n):
    """Return a string of N blank characters.

    blanks(5) -> '     '
    """
    return " " * int(n)


@_tb("cstrcat")
def forge_cstrcat(*args):
    """Concatenate strings without trimming trailing blanks.

    Unlike strcat, cstrcat preserves trailing whitespace.
    """
    return "".join(str(a) for a in args)


@_tb("deblank")
def forge_deblank(s):
    """Remove trailing blanks (spaces and tabs) from string.

    deblank('hello   ') -> 'hello'
    """
    if isinstance(s, list):
        return [str(x).rstrip(" \t") for x in s]
    return str(s).rstrip(" \t")


@_tb("strcat")
def forge_strcat(*args):
    """Concatenate strings, trimming trailing blanks from char inputs.

    strcat('hello ', 'world') -> 'helloworld'
    """
    return "".join(str(a).rstrip() for a in args)


@_tb("strjoin")
def forge_strjoin(strings, delimiter=" "):
    """Join cell array of strings with delimiter.

    strjoin({'a', 'b', 'c'}, ', ') -> 'a, b, c'
    """
    return str(delimiter).join(str(s) for s in strings)


# =====================================================================
# String Search / Test
# =====================================================================

@_tb("endsWith")
def forge_endsWith(s, suffix):
    """Check if string ends with suffix.

    Returns logical 1 or 0.
    """
    return str(s).endswith(str(suffix))


@_tb("startsWith")
def forge_startsWith(s, prefix):
    """Check if string starts with prefix.

    Returns logical 1 or 0.
    """
    return str(s).startswith(str(prefix))


@_tb("index")
def forge_index_str(s, t, direction=None):
    """Find first occurrence of string T in S (1-based).

    Returns 0 if not found.
    index('hello world', 'world')    -> 7
    index('hello world', 'world', 'last') -> 7
    """
    s_str = str(s)
    t_str = str(t)
    if direction is not None and str(direction).lower() in ("last", "l"):
        idx = s_str.rfind(t_str)
    else:
        idx = s_str.find(t_str)
    return idx + 1 if idx >= 0 else 0


@_tb("rindex")
def forge_rindex(s, t):
    """Find last occurrence of string T in S (1-based).

    Returns 0 if not found.
    """
    idx = str(s).rfind(str(t))
    return idx + 1 if idx >= 0 else 0


@_tb("strchr")
def forge_strchr(s, chars):
    """Find characters in string, return indices (1-based).

    strchr('hello world', 'lo') -> [3, 4, 5, 8]
    """
    s_str = str(s)
    char_set = set(str(chars))
    indices = [i + 1 for i, c in enumerate(s_str) if c in char_set]
    return np.array(indices, dtype=np.float64) if indices else np.array([], dtype=np.float64)


@_tb("isletter")
def forge_isletter(s):
    """Return logical array indicating which characters are letters.

    isletter('Hello 123') -> [1 1 1 1 1 0 0 0 0]
    """
    s_str = str(s)
    return np.array([c.isalpha() for c in s_str], dtype=bool)


@_tb("isstrprop")
def forge_isstrprop(s, prop):
    """Test character string properties.

    isstrprop('Hello 123', 'alpha')  -> [1 1 1 1 1 0 0 0 0]
    isstrprop('Hello 123', 'digit')  -> [0 0 0 0 0 0 1 1 1]
    isstrprop('Hello 123', 'upper')  -> [1 0 0 0 0 0 0 0 0]
    isstrprop('Hello 123', 'lower')  -> [0 1 1 1 1 0 0 0 0]
    isstrprop('Hello 123', 'wspace') -> [0 0 0 0 0 1 0 0 0]
    isstrprop('Hello 123', 'alphanum') -> [1 1 1 1 1 0 1 1 1]
    """
    s_str = str(s)
    prop = str(prop).lower()
    if prop in ("alpha", "letter"):
        return np.array([c.isalpha() for c in s_str], dtype=bool)
    elif prop == "digit":
        return np.array([c.isdigit() for c in s_str], dtype=bool)
    elif prop == "upper":
        return np.array([c.isupper() for c in s_str], dtype=bool)
    elif prop == "lower":
        return np.array([c.islower() for c in s_str], dtype=bool)
    elif prop in ("wspace", "whitespace"):
        return np.array([c.isspace() for c in s_str], dtype=bool)
    elif prop in ("alphanum", "alnum"):
        return np.array([c.isalnum() for c in s_str], dtype=bool)
    elif prop in ("punct", "punctuation"):
        import string
        pset = set(string.punctuation)
        return np.array([c in pset for c in s_str], dtype=bool)
    elif prop in ("print", "printable"):
        return np.array([c.isprintable() for c in s_str], dtype=bool)
    elif prop == "ascii":
        return np.array([ord(c) < 128 for c in s_str], dtype=bool)
    else:
        raise ValueError(f"Unknown property '{prop}'")


# =====================================================================
# String Modification / Extraction
# =====================================================================

@_tb("erase")
def forge_erase(s, match):
    """Erase all occurrences of MATCH from S.

    erase('Hello World', 'World') -> 'Hello '
    """
    return str(s).replace(str(match), "")


@_tb("strtrim")
def forge_strtrim(s):
    """Remove leading and trailing whitespace.

    strtrim('  hello  ') -> 'hello'
    """
    if isinstance(s, list):
        return [str(x).strip() for x in s]
    return str(s).strip()


@_tb("strjust")
def forge_strjust(s, pos="right"):
    """Justify string.

    strjust(s)          -> right-justified
    strjust(s, 'left')  -> left-justified
    strjust(s, 'center') -> centered
    """
    s_str = str(s)
    n = len(s_str)
    pos = str(pos).lower()
    # Determine the width from the original string length
    stripped = s_str.strip()
    if pos == "left":
        return stripped.ljust(n)
    elif pos == "center":
        return stripped.center(n)
    else:  # right
        return stripped.rjust(n)


@_tb("strtrunc")
def forge_strtrunc(s, n):
    """Truncate string to at most N characters.

    strtrunc('hello world', 5) -> 'hello'
    """
    return str(s)[:int(n)]


@_tb("substr")
def forge_substr(s, offset, length=None):
    """Extract substring (1-based offset).

    substr('hello world', 7)    -> 'world'
    substr('hello world', 7, 3) -> 'wor'
    """
    s_str = str(s)
    start = int(offset) - 1  # convert to 0-based
    if start < 0:
        start = 0
    if length is not None:
        return s_str[start:start + int(length)]
    return s_str[start:]


@_tb("untabify")
def forge_untabify(s, tabstop=8):
    """Replace tab characters with spaces.

    untabify('hello\\tworld')      -> 'hello   world' (default 8 spaces)
    untabify('hello\\tworld', 4)   -> 'hello   world' (4-space tabs)
    """
    return str(s).expandtabs(int(tabstop))


# =====================================================================
# String Splitting / Tokenizing
# =====================================================================

@_tb("strsplit")
def forge_strsplit(s, delimiter=None, collapse_delimiters=True):
    """Split string at delimiter.

    strsplit('hello world')        -> ['hello', 'world']
    strsplit('a,b,c', ',')         -> ['a', 'b', 'c']
    strsplit('a,,b', ',', false)   -> ['a', '', 'b']
    """
    s_str = str(s)
    if delimiter is None:
        return re.split(r'\s+', s_str.strip()) if s_str.strip() else []
    delim = str(delimiter)
    parts = s_str.split(delim)
    if collapse_delimiters and collapse_delimiters is not False:
        # Filter out empty strings from consecutive delimiters
        # Only collapse if explicitly requested (Octave default is true)
        pass  # split already gives correct result for single delimiters
    return parts


@_tb("strtok")
def forge_strtok(s, delimiters=None):
    """Split string at first delimiter token.

    [tok, rem] = strtok('hello world')
    tok = 'hello', rem = ' world'

    Returns (token, remainder).
    """
    s_str = str(s)
    if delimiters is None:
        delimiters = " \t\n\r\f\v"
    else:
        delimiters = str(delimiters)

    # Skip leading delimiters
    start = 0
    while start < len(s_str) and s_str[start] in delimiters:
        start += 1

    if start >= len(s_str):
        return ("", s_str)

    # Find end of token
    end = start
    while end < len(s_str) and s_str[end] not in delimiters:
        end += 1

    token = s_str[start:end]
    remainder = s_str[end:]
    return (token, remainder)


# =====================================================================
# String Conversion
# =====================================================================

@_tb("str2num")
def forge_str2num(s):
    """Convert string to number.

    str2num('3.14')   -> 3.14
    str2num('[1 2 3]') -> array([1, 2, 3])

    Returns (value, status) where status is True on success.
    """
    s_str = str(s).strip()
    try:
        # Try simple scalar
        val = float(s_str)
        if val == int(val) and '.' not in s_str and 'e' not in s_str.lower():
            return int(val)
        return val
    except ValueError:
        pass
    # Try vector notation [1 2 3] or [1, 2, 3]
    try:
        if s_str.startswith("[") and s_str.endswith("]"):
            inner = s_str[1:-1].strip()
            if "," in inner:
                parts = [float(x.strip()) for x in inner.split(",")]
            else:
                parts = [float(x) for x in inner.split()]
            return np.array(parts, dtype=np.float64)
    except (ValueError, TypeError):
        pass
    return np.array([], dtype=np.float64)


@_tb("mat2str")
def forge_mat2str(A, precision=None):
    """Convert matrix to string representation.

    mat2str([1 2; 3 4])     -> '[1 2; 3 4]'
    mat2str(pi, 5)          -> '3.1416'
    """
    A = np.asarray(A, dtype=np.float64)
    if precision is not None:
        fmt = f"%.{int(precision)}g"
    else:
        fmt = "%g"

    if A.ndim == 0 or A.size == 1:
        return fmt % A.item()

    if A.ndim == 1:
        elements = " ".join(fmt % v for v in A)
        return f"[{elements}]"

    # 2D matrix
    rows = []
    for i in range(A.shape[0]):
        rows.append(" ".join(fmt % v for v in A[i]))
    return "[" + "; ".join(rows) + "]"


@_tb("native2unicode")
def forge_native2unicode(bytes_val, encoding="UTF-8"):
    """Convert native byte values to Unicode string.

    native2unicode([72 101 108 108 111]) -> 'Hello'
    """
    if isinstance(bytes_val, (list, np.ndarray)):
        byte_arr = bytes(int(b) for b in np.asarray(bytes_val).ravel())
    elif isinstance(bytes_val, (bytes, bytearray)):
        byte_arr = bytes(bytes_val)
    else:
        byte_arr = bytes([int(bytes_val)])
    return byte_arr.decode(str(encoding))


@_tb("unicode2native")
def forge_unicode2native(s, encoding="UTF-8"):
    """Convert Unicode string to native byte values.

    unicode2native('Hello') -> [72 101 108 108 111]
    """
    encoded = str(s).encode(str(encoding))
    return np.array(list(encoded), dtype=np.float64)


@_tb("regexptranslate")
def forge_regexptranslate(mode, s):
    """Translate string for use in regular expressions.

    regexptranslate('escape', 'a.b')    -> 'a\\.b'
    regexptranslate('wildcard', '*.txt') -> '.*\\.txt'
    """
    s_str = str(s)
    mode = str(mode).lower()
    if mode == "escape":
        return re.escape(s_str)
    elif mode == "wildcard":
        # Convert glob pattern to regex
        result = ""
        for c in s_str:
            if c == "*":
                result += ".*"
            elif c == "?":
                result += "."
            elif c in r"\.[]{}()+^$|":
                result += "\\" + c
            else:
                result += c
        return result
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'escape' or 'wildcard'.")


@_tb("validatestring")
def forge_validatestring(s, valid_strings, func_name=None):
    """Validate string against list of valid options.

    Matches partial strings (case-insensitive prefix matching).
    validatestring('lin', {'linear', 'log', 'cubic'}) -> 'linear'
    """
    s_str = str(s).lower()
    matches = []
    for vs in valid_strings:
        vs_str = str(vs)
        if vs_str.lower().startswith(s_str):
            matches.append(vs_str)
    if len(matches) == 1:
        return matches[0]
    elif len(matches) == 0:
        ctx = f" in {func_name}" if func_name else ""
        raise ValueError(
            f"'{s}' does not match any valid string{ctx}. "
            f"Valid options: {', '.join(str(v) for v in valid_strings)}"
        )
    else:
        # Check for exact match among the ambiguous matches
        for m in matches:
            if m.lower() == s_str:
                return m
        ctx = f" in {func_name}" if func_name else ""
        raise ValueError(
            f"Ambiguous string '{s}'{ctx}. "
            f"Matches: {', '.join(matches)}"
        )


@_tb("upper")
def forge_upper(s):
    """Convert string to uppercase."""
    from forge.engine.containers import ForgeChar
    if isinstance(s, ForgeChar):
        return ForgeChar(s.to_str().upper())
    return str(s).upper()


@_tb("lower")
def forge_lower(s):
    """Convert string to lowercase."""
    from forge.engine.containers import ForgeChar
    if isinstance(s, ForgeChar):
        return ForgeChar(s.to_str().lower())
    return str(s).lower()


@_tb("toupper")
def forge_toupper(s):
    """Convert string to uppercase (alias for upper)."""
    return forge_upper(s)


@_tb("tolower")
def forge_tolower(s):
    """Convert string to lowercase (alias for lower)."""
    return forge_lower(s)


@_tb("num2str")
def forge_num2str(x, fmt=None):
    """Convert number to string."""
    from forge.engine.types import ForgeArray
    from forge.engine.containers import ForgeChar
    import numpy as np
    if isinstance(x, ForgeArray):
        val = x._data.flat[0] if x._data.size == 1 else x._data
        if isinstance(val, np.ndarray):
            parts = []
            for v in val.flat:
                if fmt:
                    parts.append(fmt % v)
                elif v == int(v):
                    parts.append(str(int(v)))
                else:
                    parts.append(str(v))
            return ForgeChar(" ".join(parts))
        x = val
    if fmt:
        return ForgeChar(fmt % x)
    if isinstance(x, (int, float)) and x == int(x):
        return ForgeChar(str(int(x)))
    return ForgeChar(str(x))


@_tb("sprintf")
def forge_sprintf(fmt, *args):
    """Format string with printf-style formatting."""
    from forge.engine.containers import ForgeChar
    from forge.engine.types import ForgeArray
    import numpy as np
    if isinstance(fmt, ForgeChar):
        fmt = fmt.to_str()
    # Convert ForgeArray args to scalars where possible
    converted = []
    for a in args:
        if isinstance(a, ForgeArray):
            if a._data.size == 1:
                converted.append(a._data.flat[0])
            else:
                converted.append(a._data)
        elif isinstance(a, ForgeChar):
            converted.append(a.to_str())
        else:
            converted.append(a)
    try:
        result = fmt % tuple(converted)
    except TypeError:
        result = fmt
    return ForgeChar(result)


@_tb("strcmp")
def forge_strcmp(s1, s2):
    import numpy as np
    """Compare two strings (exact match)."""
    from forge.engine.containers import ForgeChar
    from forge.engine.types import ForgeArray
    a = s1.to_str() if isinstance(s1, ForgeChar) else str(s1)
    b = s2.to_str() if isinstance(s2, ForgeChar) else str(s2)
    return ForgeArray(np.float64(1.0 if a == b else 0.0))


@_tb("strcmpi")
def forge_strcmpi(s1, s2):
    import numpy as np
    """Compare two strings (case-insensitive)."""
    from forge.engine.containers import ForgeChar
    from forge.engine.types import ForgeArray
    a = s1.to_str() if isinstance(s1, ForgeChar) else str(s1)
    b = s2.to_str() if isinstance(s2, ForgeChar) else str(s2)
    return ForgeArray(np.float64(1.0 if a.lower() == b.lower() else 0.0))


@_tb("regexpi")
def forge_regexpi(s, pattern, *args):
    """Case-insensitive regex match."""
    from forge.engine.containers import ForgeChar
    from forge.engine.types import ForgeArray
    import re
    s_str = s.to_str() if isinstance(s, ForgeChar) else str(s)
    pat = pattern.to_str() if isinstance(pattern, ForgeChar) else str(pattern)
    matches = list(re.finditer(pat, s_str, re.IGNORECASE))
    if not matches:
        return ForgeArray._from_list([])
    starts = [m.start() + 1 for m in matches]  # 1-based
    return ForgeArray._from_list(starts)

@_tb("chr")
def forge_chr(n):
    """Convert integer to character. chr(65) -> 'A'."""
    import numpy as np
    from forge.engine.containers import ForgeChar
    from forge.engine.types import ForgeArray
    if isinstance(n, ForgeArray):
        vals = n.data.ravel().astype(int)
        return ForgeChar(''.join(chr(v) for v in vals))
    return ForgeChar(chr(int(n)))


@_tb("char")
def forge_char_func(*args):
    """Convert to character array. char(65) -> 'A', char('hello') -> 'hello'."""
    import numpy as np
    from forge.engine.containers import ForgeChar
    from forge.engine.types import ForgeArray
    if len(args) == 1:
        a = args[0]
        if isinstance(a, ForgeChar):
            return a
        if isinstance(a, ForgeArray):
            vals = a.data.ravel().astype(int)
            return ForgeChar(''.join(chr(v) for v in vals))
        if isinstance(a, str):
            return ForgeChar(a)
        return ForgeChar(chr(int(a)))
    # Multiple args: stack as rows (each becomes a char row)
    rows = []
    for a in args:
        if isinstance(a, ForgeChar):
            rows.append(a.to_str())
        elif isinstance(a, ForgeArray):
            vals = a.data.ravel().astype(int)
            rows.append(''.join(chr(v) for v in vals))
        elif isinstance(a, str):
            rows.append(a)
        else:
            rows.append(chr(int(a)))
    # Pad to same length
    maxlen = max(len(r) for r in rows)
    rows = [r.ljust(maxlen) for r in rows]
    return ForgeChar(rows[0] if len(rows) == 1 else rows)


@_tb("double")
def forge_double_from_char(x):
    """Convert char to double (ASCII values). double('A') -> 65."""
    import numpy as np
    from forge.engine.containers import ForgeChar
    from forge.engine.types import ForgeArray
    if isinstance(x, ForgeChar):
        s = x.to_str()
        return ForgeArray(np.array([ord(c) for c in s], dtype=np.float64))
    if isinstance(x, ForgeArray):
        return ForgeArray(x.data.astype(np.float64))
    return ForgeArray(np.float64(x))


@_tb("num2str")
def forge_num2str(x, fmt=None):
    """Convert number to string."""
    import numpy as np
    from forge.engine.containers import ForgeChar
    from forge.engine.types import ForgeArray
    if isinstance(fmt, ForgeChar):
        fmt = fmt.to_str()
    if isinstance(x, ForgeArray):
        val = x.data.flat[0]
        if fmt:
            return ForgeChar(fmt % val)
        if val == int(val):
            return ForgeChar(str(int(val)))
        return ForgeChar(str(val))
    if fmt:
        return ForgeChar(fmt % float(x))
    return ForgeChar(str(x))


@_tb("str2num")
def forge_str2num(s):
    """Convert string to number."""
    import numpy as np
    from forge.engine.containers import ForgeChar
    from forge.engine.types import ForgeArray
    if isinstance(s, ForgeChar):
        s = s.to_str()
    try:
        return ForgeArray(np.float64(float(s)))
    except ValueError:
        return ForgeArray(np.array([]))


@_tb("str2double")
def forge_str2double(s):
    """Convert string to double."""
    import numpy as np
    from forge.engine.containers import ForgeChar
    from forge.engine.types import ForgeArray
    if isinstance(s, ForgeChar):
        s = s.to_str()
    try:
        return ForgeArray(np.float64(float(s)))
    except ValueError:
        return ForgeArray(np.float64(np.nan))


@_tb("strtrim")
def forge_strtrim(s):
    """Remove leading and trailing whitespace."""
    from forge.engine.containers import ForgeChar
    if isinstance(s, ForgeChar):
        return ForgeChar(s.to_str().strip())
    return ForgeChar(str(s).strip())


@_tb("strsplit")
def forge_strsplit(s, delim=None):
    """Split string into cell array of strings."""
    from forge.engine.containers import ForgeChar, ForgeCell
    if isinstance(s, ForgeChar):
        s = s.to_str()
    if isinstance(delim, ForgeChar):
        delim = delim.to_str()
    if delim:
        parts = str(s).split(delim)
    else:
        parts = str(s).split()
    return ForgeCell([ForgeChar(p) for p in parts])


@_tb("strjoin")
def forge_strjoin(parts, delim=None):
    """Join cell array of strings."""
    from forge.engine.containers import ForgeChar, ForgeCell
    if isinstance(delim, ForgeChar):
        delim = delim.to_str()
    elif delim is None:
        delim = " "
    strs = []
    if isinstance(parts, ForgeCell):
        for p in parts.data.flat:
            if isinstance(p, ForgeChar):
                strs.append(p.to_str())
            else:
                strs.append(str(p))
    return ForgeChar(delim.join(strs))


@_tb("lower")
def forge_lower(s):
    """Convert to lowercase."""
    from forge.engine.containers import ForgeChar
    if isinstance(s, ForgeChar):
        return ForgeChar(s.to_str().lower())
    return ForgeChar(str(s).lower())


@_tb("upper")
def forge_upper(s):
    """Convert to uppercase."""
    from forge.engine.containers import ForgeChar
    if isinstance(s, ForgeChar):
        return ForgeChar(s.to_str().upper())
    return ForgeChar(str(s).upper())


@_tb("strcat")
def forge_strcat(*args):
    """Concatenate strings."""
    from forge.engine.containers import ForgeChar
    result = ""
    for a in args:
        if isinstance(a, ForgeChar):
            result += a.to_str()
        else:
            result += str(a)
    return ForgeChar(result)

