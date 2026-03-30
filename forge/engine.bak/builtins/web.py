# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Web / Network Toolbox for Forge — Octave-compatible functions.

Implements 10 Octave web/network functions: HTTP (web, webread, webwrite,
weboptions) and FTP (ftp_connect, ftp_cd, ftp_dir, ftp_get, ftp_put, ftp_close).

Backend: urllib.request / json for HTTP, ftplib for FTP.

SRS trace: SRS-FUNC-WEB
"""

from __future__ import annotations

import json
import warnings
import numpy as np

# ── Optional imports ─────────────────────────────────────────────
try:
    import urllib.request
    import urllib.parse
    import urllib.error
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False

try:
    import ftplib
    _HAS_FTP = True
except ImportError:
    _HAS_FTP = False

# ── ForgeArray interop ───────────────────────────────────────────
try:
    from forge.engine.types import ForgeArray, _unwrap
except ImportError:
    ForgeArray = np.ndarray

    def _unwrap(x):
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x, dtype=np.float64)

try:
    from forge.engine.containers import ForgeChar
except ImportError:
    ForgeChar = str


def _fa(x):
    """Wrap result as ForgeArray."""
    arr = np.asarray(x)
    if ForgeArray is np.ndarray:
        return arr
    return ForgeArray(arr)


def _scalar(x):
    if isinstance(x, ForgeArray):
        d = x.data if hasattr(x, 'data') else np.asarray(x)
        return d.flat[0].item() if d.size == 1 else d
    if isinstance(x, np.ndarray) and x.size == 1:
        return x.flat[0].item()
    return x


def _to_str(x):
    """Convert ForgeChar or other to plain string."""
    if isinstance(x, str):
        return x
    if hasattr(x, 'char_value'):
        return x.char_value
    return str(_scalar(x))


# ═══════════════════════════════════════════════════════════════════
# 1. WEB OPTIONS
# ═══════════════════════════════════════════════════════════════════

class _WebOptions:
    """Options object for webread/webwrite (like Octave weboptions).

    Stores request configuration: headers, content type, timeout, etc.
    """

    def __init__(self, **kwargs):
        self.ContentType = kwargs.get('ContentType', 'auto')
        self.Timeout = kwargs.get('Timeout', 30)
        self.UserAgent = kwargs.get('UserAgent', 'Forge/1.0')
        self.Username = kwargs.get('Username', '')
        self.Password = kwargs.get('Password', '')
        self.KeyName = kwargs.get('KeyName', '')
        self.KeyValue = kwargs.get('KeyValue', '')
        self.HeaderFields = kwargs.get('HeaderFields', {})
        self.RequestMethod = kwargs.get('RequestMethod', 'auto')
        self.ArrayFormat = kwargs.get('ArrayFormat', 'csv')
        self.CertificateFilename = kwargs.get('CertificateFilename', '')
        self.MediaType = kwargs.get('MediaType', 'application/x-www-form-urlencoded')

    def __repr__(self):
        return (f"weboptions(ContentType='{self.ContentType}', "
                f"Timeout={self.Timeout}, "
                f"UserAgent='{self.UserAgent}', "
                f"RequestMethod='{self.RequestMethod}')")


def weboptions(**kwargs):
    """Create web request options.

    OPTS = weboptions()
    OPTS = weboptions('Name', Value, ...)
    OPTS = weboptions(ContentType='json', Timeout=60)

    Supported options:
        ContentType, Timeout, UserAgent, Username, Password,
        KeyName, KeyValue, HeaderFields, RequestMethod,
        ArrayFormat, CertificateFilename, MediaType
    """
    return _WebOptions(**kwargs)


# ═══════════════════════════════════════════════════════════════════
# 2. HTTP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _build_request(url, opts=None, data=None, method=None):
    """Build a urllib Request object from URL and options."""
    if not _HAS_URLLIB:
        raise RuntimeError("urllib is required for web functions")

    url = _to_str(url)
    if opts is None:
        opts = _WebOptions()

    headers = {'User-Agent': opts.UserAgent}
    if isinstance(opts.HeaderFields, dict):
        headers.update(opts.HeaderFields)

    if opts.ContentType == 'json' or opts.MediaType == 'application/json':
        headers['Content-Type'] = 'application/json'
        headers['Accept'] = 'application/json'

    req_data = None
    if data is not None:
        if isinstance(data, dict):
            if headers.get('Content-Type') == 'application/json':
                req_data = json.dumps(data).encode('utf-8')
            else:
                req_data = urllib.parse.urlencode(data).encode('utf-8')
        elif isinstance(data, str):
            req_data = data.encode('utf-8')
        elif isinstance(data, bytes):
            req_data = data
        else:
            req_data = json.dumps(data).encode('utf-8')

    req = urllib.request.Request(url, data=req_data, headers=headers)
    if method is not None:
        req.method = method
    elif req_data is not None:
        req.method = 'POST'
    else:
        req.method = 'GET'

    # Override method from options if specified
    if opts.RequestMethod != 'auto':
        req.method = opts.RequestMethod.upper()

    return req, opts.Timeout


def web(url, *args):
    """Open URL in system browser (stub) or fetch content.

    web(URL)
    web(URL, '-browser')    — open in browser (stub, prints URL)
    CONTENT = web(URL)      — fetch and return content as string

    In headless mode, 'web' with '-browser' prints a message.
    Without '-browser', it fetches the content.
    """
    url = _to_str(url)

    if args and _to_str(args[0]).lower() == '-browser':
        print(f"web: would open '{url}' in browser (not available in headless mode)")
        return

    # Fetch content
    if not _HAS_URLLIB:
        raise RuntimeError("urllib is required for web functions")

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Forge/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode('utf-8', errors='replace')
        return content
    except urllib.error.URLError as e:
        raise RuntimeError(f"web: failed to fetch '{url}': {e}")


def webread(url, *args):
    """Read content from a web service.

    DATA = webread(URL)
    DATA = webread(URL, NAME1, VALUE1, ...)
    DATA = webread(URL, OPTS)

    Parameters
    ----------
    url : str — The URL to read from.
    args : optional — Query parameters as Name/Value pairs, or a weboptions object.

    Returns
    -------
    data : str or dict — Response data. If JSON, returns parsed dict.
    """
    url = _to_str(url)

    # Parse options and query params
    opts = None
    query_params = {}

    i = 0
    while i < len(args):
        arg = args[i]
        if isinstance(arg, _WebOptions):
            opts = arg
            i += 1
        elif isinstance(arg, str) or (hasattr(arg, 'char_value')):
            # Name/Value pair
            name = _to_str(arg)
            if i + 1 < len(args):
                value = args[i + 1]
                if hasattr(value, 'char_value'):
                    value = value.char_value
                query_params[name] = value
                i += 2
            else:
                i += 1
        else:
            i += 1

    # Append query params to URL
    if query_params:
        sep = '&' if '?' in url else '?'
        url = url + sep + urllib.parse.urlencode(query_params)

    req, timeout = _build_request(url, opts)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get('Content-Type', '')
            raw = resp.read()

            if 'json' in content_type or (opts and opts.ContentType == 'json'):
                try:
                    return json.loads(raw.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

            # Try JSON anyway
            try:
                return json.loads(raw.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

            return raw.decode('utf-8', errors='replace')
    except urllib.error.URLError as e:
        raise RuntimeError(f"webread: failed to read '{url}': {e}")


def webwrite(url, data, *args):
    """Write data to a web service.

    RESPONSE = webwrite(URL, DATA)
    RESPONSE = webwrite(URL, NAME1, VALUE1, ...)
    RESPONSE = webwrite(URL, DATA, OPTS)

    Parameters
    ----------
    url : str — The URL to write to.
    data : dict, str, or Name/Value pairs — Data to send.
    args : optional — weboptions object as last argument.
    """
    url = _to_str(url)

    opts = None
    post_data = data

    # Check if data is actually a name (string) for name/value pairs
    if isinstance(data, str) or (hasattr(data, 'char_value')):
        # Might be Name/Value pairs: webwrite(url, n1, v1, n2, v2, ...)
        name = _to_str(data)
        pairs = {}
        all_args = list(args)

        # Check if last arg is weboptions
        if all_args and isinstance(all_args[-1], _WebOptions):
            opts = all_args.pop()

        if all_args:
            # First value
            pairs[name] = all_args[0] if not hasattr(all_args[0], 'char_value') else all_args[0].char_value
            i = 1
            while i + 1 < len(all_args):
                k = _to_str(all_args[i])
                v = all_args[i + 1]
                if hasattr(v, 'char_value'):
                    v = v.char_value
                pairs[k] = v
                i += 2
            post_data = pairs
        else:
            post_data = {name: ''}
    else:
        # data is a dict or other object
        if args and isinstance(args[-1], _WebOptions):
            opts = args[-1]

    req, timeout = _build_request(url, opts, data=post_data, method='POST')

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return json.loads(raw.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return raw.decode('utf-8', errors='replace')
    except urllib.error.URLError as e:
        raise RuntimeError(f"webwrite: failed to write to '{url}': {e}")


# ═══════════════════════════════════════════════════════════════════
# 3. FTP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

# Module-level FTP connection storage
_ftp_connections = {}
_ftp_counter = 0


def ftp_connect(host, *args):
    """Connect to an FTP server.

    CONN = ftp_connect(HOST)
    CONN = ftp_connect(HOST, USER, PASS)

    Parameters
    ----------
    host : str — FTP server hostname.
    args : optional — Username and password.

    Returns
    -------
    conn : int — Connection handle (integer ID).
    """
    global _ftp_counter

    if not _HAS_FTP:
        raise RuntimeError("ftplib is required for FTP functions")

    host = _to_str(host)
    user = _to_str(args[0]) if len(args) > 0 else 'anonymous'
    passwd = _to_str(args[1]) if len(args) > 1 else 'anonymous@'

    ftp = ftplib.FTP()
    ftp.connect(host)
    ftp.login(user, passwd)

    _ftp_counter += 1
    handle = _ftp_counter
    _ftp_connections[handle] = ftp

    return handle


def _get_ftp(handle):
    """Retrieve FTP connection by handle."""
    handle = int(_scalar(handle))
    ftp = _ftp_connections.get(handle)
    if ftp is None:
        raise RuntimeError(f"ftp: invalid connection handle {handle}")
    return ftp


def ftp_cd(handle, directory):
    """Change remote directory on FTP connection.

    ftp_cd(CONN, DIR)
    """
    ftp = _get_ftp(handle)
    directory = _to_str(directory)
    ftp.cwd(directory)


def ftp_dir(handle, *args):
    """List files in current remote directory.

    FILES = ftp_dir(CONN)
    FILES = ftp_dir(CONN, PATH)

    Returns a list of filename strings.
    """
    ftp = _get_ftp(handle)
    path = _to_str(args[0]) if args else '.'

    try:
        entries = ftp.nlst(path)
    except ftplib.error_perm:
        entries = []

    return entries


def ftp_get(handle, remote_file, local_file=None):
    """Download a file from the FTP server.

    ftp_get(CONN, REMOTE_FILE)
    ftp_get(CONN, REMOTE_FILE, LOCAL_FILE)

    Parameters
    ----------
    handle : int — Connection handle.
    remote_file : str — Remote file path.
    local_file : str or None — Local destination path (defaults to basename of remote).
    """
    import os

    ftp = _get_ftp(handle)
    remote_file = _to_str(remote_file)

    if local_file is None:
        local_file = os.path.basename(remote_file)
    else:
        local_file = _to_str(local_file)

    with open(local_file, 'wb') as f:
        ftp.retrbinary(f'RETR {remote_file}', f.write)


def ftp_put(handle, local_file, remote_file=None):
    """Upload a file to the FTP server.

    ftp_put(CONN, LOCAL_FILE)
    ftp_put(CONN, LOCAL_FILE, REMOTE_FILE)

    Parameters
    ----------
    handle : int — Connection handle.
    local_file : str — Local file path.
    remote_file : str or None — Remote destination (defaults to basename of local).
    """
    import os

    ftp = _get_ftp(handle)
    local_file = _to_str(local_file)

    if remote_file is None:
        remote_file = os.path.basename(local_file)
    else:
        remote_file = _to_str(remote_file)

    with open(local_file, 'rb') as f:
        ftp.storbinary(f'STOR {remote_file}', f)


def ftp_close(handle):
    """Close an FTP connection.

    ftp_close(CONN)
    """
    handle_int = int(_scalar(handle))
    ftp = _ftp_connections.pop(handle_int, None)
    if ftp is not None:
        try:
            ftp.quit()
        except Exception:
            try:
                ftp.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════

WEB_REGISTRY = {
    # ── HTTP ──────────────────────────────────────────────────────
    'web':              web,
    'webread':          webread,
    'webwrite':         webwrite,
    'weboptions':       weboptions,
    # ── FTP ───────────────────────────────────────────────────────
    'ftp_connect':      ftp_connect,
    'ftp_cd':           ftp_cd,
    'ftp_dir':          ftp_dir,
    'ftp_get':          ftp_get,
    'ftp_put':          ftp_put,
    'ftp_close':        ftp_close,
}
