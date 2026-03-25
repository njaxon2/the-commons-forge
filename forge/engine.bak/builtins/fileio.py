"""File I/O toolbox for Forge.

Implements CSV I/O, file read/write, file system operations, archive
utilities, and system information functions.

Backend: numpy (CSV), os/shutil (file ops), platform (system info).

SRS trace: SRS-FUNC-001 (Octave-compatible function library)
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
import tarfile
import zipfile
import gzip
import bz2
import struct as _struct
import time

import numpy as np

from forge.engine.types import ForgeArray, _unwrap
from forge.engine.containers import ForgeChar, ForgeStruct


# ── Helpers ──────────────────────────────────────────────────────

def _wrap(x):
    """Wrap a numpy array as a ForgeArray."""
    return ForgeArray(np.asarray(x))


def _wrap_str(s):
    """Wrap a Python string as a ForgeChar."""
    return ForgeChar(s)


# =====================================================================
# CSV / Delimited I/O
# =====================================================================

def forge_csvread(filename, r0=0, c0=0):
    """Read a CSV file into a numeric array.

    Parameters
    ----------
    filename : str or ForgeChar
        Path to the CSV file.
    r0 : int, optional
        First row to read (0-based). Default 0.
    c0 : int, optional
        First column to read (0-based). Default 0.

    Returns a ForgeArray of numeric values.
    """
    if isinstance(filename, ForgeChar):
        filename = filename.to_str()
    filename = str(filename)
    r0 = int(_unwrap(r0)) if not isinstance(r0, (int, float)) else int(r0)
    c0 = int(_unwrap(c0)) if not isinstance(c0, (int, float)) else int(c0)

    data = np.loadtxt(filename, delimiter=',', skiprows=r0)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if c0 > 0:
        data = data[:, c0:]
    return _wrap(data)


def forge_csvwrite(filename, M, r0=0, c0=0):
    """Write a numeric array to a CSV file.

    Parameters
    ----------
    filename : str or ForgeChar
        Path for the output CSV file.
    M : ForgeArray
        Numeric matrix to write.
    r0 : int, optional
        Row offset (prepend blank rows). Default 0.
    c0 : int, optional
        Column offset (prepend blank columns). Default 0.
    """
    if isinstance(filename, ForgeChar):
        filename = filename.to_str()
    filename = str(filename)
    data = np.asarray(_unwrap(M), dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    r0 = int(_unwrap(r0)) if not isinstance(r0, (int, float)) else int(r0)
    c0 = int(_unwrap(c0)) if not isinstance(c0, (int, float)) else int(c0)

    if r0 > 0 or c0 > 0:
        padded = np.zeros((data.shape[0] + r0, data.shape[1] + c0), dtype=float)
        padded[r0:, c0:] = data
        data = padded

    np.savetxt(filename, data, delimiter=',', fmt='%g')


def forge_dlmwrite(filename, M, delimiter=',', r0=0, c0=0):
    """Write array to a delimited text file.

    Parameters
    ----------
    filename : str or ForgeChar
        Output file path.
    M : ForgeArray
        Data matrix.
    delimiter : str, optional
        Column delimiter. Default ','.
    r0, c0 : int, optional
        Row and column offset. Default 0.
    """
    if isinstance(filename, ForgeChar):
        filename = filename.to_str()
    if isinstance(delimiter, ForgeChar):
        delimiter = delimiter.to_str()
    filename = str(filename)
    data = np.asarray(_unwrap(M), dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    r0 = int(r0)
    c0 = int(c0)

    if r0 > 0 or c0 > 0:
        padded = np.zeros((data.shape[0] + r0, data.shape[1] + c0), dtype=float)
        padded[r0:, c0:] = data
        data = padded

    np.savetxt(filename, data, delimiter=str(delimiter), fmt='%g')


def forge_fileread(filename):
    """Read entire contents of a text file as a character string.

    Parameters
    ----------
    filename : str or ForgeChar
        Path to the text file.

    Returns a ForgeChar containing the file contents.
    """
    if isinstance(filename, ForgeChar):
        filename = filename.to_str()
    filename = str(filename)
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        contents = f.read()
    return _wrap_str(contents)


def forge_importdata(filename, delimiter=None):
    """Import data from a file, auto-detecting format.

    For CSV/text files, returns a ForgeArray of numeric data.
    For other formats, returns the raw text as ForgeChar.

    Parameters
    ----------
    filename : str or ForgeChar
        File path.
    delimiter : str, optional
        Column delimiter for text files.
    """
    if isinstance(filename, ForgeChar):
        filename = filename.to_str()
    filename = str(filename)
    ext = os.path.splitext(filename)[1].lower()

    if ext in ('.csv', '.dat', '.tsv', '.txt'):
        dlm = delimiter if delimiter else ','
        if isinstance(dlm, ForgeChar):
            dlm = dlm.to_str()
        if ext == '.tsv':
            dlm = '\t'
        try:
            data = np.loadtxt(filename, delimiter=dlm)
            return _wrap(data)
        except ValueError:
            with open(filename, 'r', encoding='utf-8', errors='replace') as f:
                return _wrap_str(f.read())
    else:
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
            return _wrap_str(f.read())


def forge_is_valid_file_id(fid):
    """Test whether fid is a valid file identifier.

    Returns True (1.0) if fid looks like a valid file descriptor or
    file-like object, False (0.0) otherwise.
    """
    if isinstance(fid, (int, float, np.integer, np.floating)):
        fid_int = int(fid)
        # stdin=0, stdout=1, stderr=2 are always valid in Octave
        if fid_int in (0, 1, 2):
            return _wrap(np.array(1.0))
        try:
            os.fstat(fid_int)
            return _wrap(np.array(1.0))
        except (OSError, ValueError):
            return _wrap(np.array(0.0))
    return _wrap(np.array(0.0))


def forge_beep():
    """Produce a beep sound.

    Prints the ASCII BEL character. In a terminal this typically produces
    an audible alert.
    """
    print('\a', end='', flush=True)
    return None


# =====================================================================
# File System Operations
# =====================================================================

def forge_copyfile(src, dst):
    """Copy a file.

    Parameters
    ----------
    src : str or ForgeChar
        Source file path.
    dst : str or ForgeChar
        Destination file path or directory.

    Returns (status, msg) where status is 1 for success, 0 for failure.
    """
    if isinstance(src, ForgeChar):
        src = src.to_str()
    if isinstance(dst, ForgeChar):
        dst = dst.to_str()
    src, dst = str(src), str(dst)
    try:
        shutil.copy2(src, dst)
        return (_wrap(np.array(1.0)), _wrap_str(''))
    except Exception as e:
        return (_wrap(np.array(0.0)), _wrap_str(str(e)))


def forge_delete_file(filename):
    """Delete a file.

    Parameters
    ----------
    filename : str or ForgeChar
        Path of the file to delete.
    """
    if isinstance(filename, ForgeChar):
        filename = filename.to_str()
    filename = str(filename)
    if os.path.isfile(filename):
        os.remove(filename)


def forge_dir_list(directory=None):
    """List directory contents.

    Parameters
    ----------
    directory : str or ForgeChar, optional
        Directory path. Default is current working directory.

    Returns a ForgeStruct array-like with fields: name, folder, date,
    bytes, isdir.
    """
    if directory is None:
        directory = os.getcwd()
    if isinstance(directory, ForgeChar):
        directory = directory.to_str()
    directory = str(directory)

    import glob as _glob

    # Support glob patterns
    if any(c in directory for c in ['*', '?']):
        entries = _glob.glob(directory)
    else:
        if os.path.isdir(directory):
            entries = [os.path.join(directory, e) for e in os.listdir(directory)]
        else:
            entries = _glob.glob(directory)

    results = []
    for entry in sorted(entries):
        stat = os.stat(entry)
        results.append(ForgeStruct(
            name=_wrap_str(os.path.basename(entry)),
            folder=_wrap_str(os.path.dirname(os.path.abspath(entry))),
            date=_wrap_str(time.strftime('%d-%b-%Y %H:%M:%S',
                                         time.localtime(stat.st_mtime))),
            bytes=_wrap(np.array(float(stat.st_size))),
            isdir=_wrap(np.array(1.0 if os.path.isdir(entry) else 0.0)),
        ))
    return results


def forge_fileattrib(filename):
    """Get file attributes.

    Parameters
    ----------
    filename : str or ForgeChar
        File path.

    Returns a ForgeStruct with fields: Name, archive, system, hidden,
    directory, UserRead, UserWrite, UserExec.
    """
    if isinstance(filename, ForgeChar):
        filename = filename.to_str()
    filename = str(filename)
    stat = os.stat(filename)
    import stat as stat_mod

    mode = stat.st_mode
    return ForgeStruct(
        Name=_wrap_str(filename),
        archive=_wrap(np.array(0.0)),
        system=_wrap(np.array(0.0)),
        hidden=_wrap(np.array(1.0 if os.path.basename(filename).startswith('.') else 0.0)),
        directory=_wrap(np.array(1.0 if os.path.isdir(filename) else 0.0)),
        UserRead=_wrap(np.array(1.0 if mode & stat_mod.S_IRUSR else 0.0)),
        UserWrite=_wrap(np.array(1.0 if mode & stat_mod.S_IWUSR else 0.0)),
        UserExec=_wrap(np.array(1.0 if mode & stat_mod.S_IXUSR else 0.0)),
    )


def forge_fileparts(filename):
    """Split a file path into directory, name, and extension.

    Parameters
    ----------
    filename : str or ForgeChar
        File path.

    Returns (pathstr, name, ext).
    """
    if isinstance(filename, ForgeChar):
        filename = filename.to_str()
    filename = str(filename)
    d = os.path.dirname(filename)
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    return (_wrap_str(d), _wrap_str(name), _wrap_str(ext))


def forge_fullfile(*parts):
    """Build full file path from parts.

    Parameters
    ----------
    *parts : str or ForgeChar
        Path components to join.

    Returns a ForgeChar with the assembled path.
    """
    strs = []
    for p in parts:
        if isinstance(p, ForgeChar):
            strs.append(p.to_str())
        else:
            strs.append(str(p))
    return _wrap_str(os.path.join(*strs))


def forge_mkdir(dirname):
    """Create a directory (and parents if needed).

    Parameters
    ----------
    dirname : str or ForgeChar
        Directory path.

    Returns (status, msg) where status is 1 for success, 0 for failure.
    """
    if isinstance(dirname, ForgeChar):
        dirname = dirname.to_str()
    dirname = str(dirname)
    try:
        os.makedirs(dirname, exist_ok=True)
        return (_wrap(np.array(1.0)), _wrap_str(''))
    except Exception as e:
        return (_wrap(np.array(0.0)), _wrap_str(str(e)))


def forge_movefile(src, dst):
    """Move or rename a file.

    Parameters
    ----------
    src : str or ForgeChar
        Source path.
    dst : str or ForgeChar
        Destination path.

    Returns (status, msg).
    """
    if isinstance(src, ForgeChar):
        src = src.to_str()
    if isinstance(dst, ForgeChar):
        dst = dst.to_str()
    src, dst = str(src), str(dst)
    try:
        shutil.move(src, dst)
        return (_wrap(np.array(1.0)), _wrap_str(''))
    except Exception as e:
        return (_wrap(np.array(0.0)), _wrap_str(str(e)))


def forge_isfile(filename):
    """Test whether a path refers to an existing regular file.

    Returns 1.0 (true) or 0.0 (false).
    """
    if isinstance(filename, ForgeChar):
        filename = filename.to_str()
    return _wrap(np.array(1.0 if os.path.isfile(str(filename)) else 0.0))


def forge_isfolder(dirname):
    """Test whether a path refers to an existing directory.

    Returns 1.0 (true) or 0.0 (false).
    """
    if isinstance(dirname, ForgeChar):
        dirname = dirname.to_str()
    return _wrap(np.array(1.0 if os.path.isdir(str(dirname)) else 0.0))


def forge_ls(directory=None):
    """List directory contents (simple name listing).

    Parameters
    ----------
    directory : str or ForgeChar, optional
        Directory to list. Defaults to cwd.

    Returns a ForgeChar containing newline-separated file names.
    """
    if directory is None:
        directory = os.getcwd()
    if isinstance(directory, ForgeChar):
        directory = directory.to_str()
    directory = str(directory)
    entries = sorted(os.listdir(directory))
    return _wrap_str('\n'.join(entries))


# =====================================================================
# Archive Operations
# =====================================================================

def forge_tar(tarname, *files):
    """Create a tar archive.

    Parameters
    ----------
    tarname : str or ForgeChar
        Name of the tar file to create.
    *files : str or ForgeChar
        Files/directories to add.
    """
    if isinstance(tarname, ForgeChar):
        tarname = tarname.to_str()
    tarname = str(tarname)
    mode = 'w:gz' if tarname.endswith('.gz') else 'w'
    with tarfile.open(tarname, mode) as tf:
        for f in files:
            if isinstance(f, ForgeChar):
                f = f.to_str()
            tf.add(str(f))


def forge_untar(tarname, destdir=None):
    """Extract a tar archive.

    Parameters
    ----------
    tarname : str or ForgeChar
        Path to the tar file.
    destdir : str or ForgeChar, optional
        Destination directory. Default is cwd.

    Returns a list of extracted file names as ForgeChar entries.
    """
    if isinstance(tarname, ForgeChar):
        tarname = tarname.to_str()
    tarname = str(tarname)
    if destdir is None:
        destdir = os.getcwd()
    if isinstance(destdir, ForgeChar):
        destdir = destdir.to_str()
    destdir = str(destdir)

    with tarfile.open(tarname, 'r:*') as tf:
        names = tf.getnames()
        tf.extractall(destdir)
    return [_wrap_str(n) for n in names]


def forge_zip_file(zipname, *files):
    """Create a zip archive.

    Parameters
    ----------
    zipname : str or ForgeChar
        Name of the zip file.
    *files : str or ForgeChar
        Files/directories to add.
    """
    if isinstance(zipname, ForgeChar):
        zipname = zipname.to_str()
    zipname = str(zipname)
    with zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if isinstance(f, ForgeChar):
                f = f.to_str()
            f = str(f)
            if os.path.isdir(f):
                for root, _dirs, fnames in os.walk(f):
                    for fn in fnames:
                        full = os.path.join(root, fn)
                        zf.write(full, os.path.relpath(full, os.path.dirname(f)))
            else:
                zf.write(f, os.path.basename(f))


def forge_unzip_file(zipname, destdir=None):
    """Extract a zip archive.

    Parameters
    ----------
    zipname : str or ForgeChar
        Path to the zip file.
    destdir : str or ForgeChar, optional
        Destination directory. Default is cwd.

    Returns a list of extracted file names as ForgeChar entries.
    """
    if isinstance(zipname, ForgeChar):
        zipname = zipname.to_str()
    zipname = str(zipname)
    if destdir is None:
        destdir = os.getcwd()
    if isinstance(destdir, ForgeChar):
        destdir = destdir.to_str()
    destdir = str(destdir)

    with zipfile.ZipFile(zipname, 'r') as zf:
        names = zf.namelist()
        zf.extractall(destdir)
    return [_wrap_str(n) for n in names]


def forge_gunzip(filename, destdir=None):
    """Decompress a gzip file.

    Parameters
    ----------
    filename : str or ForgeChar
        Path to the .gz file.
    destdir : str or ForgeChar, optional
        Destination directory. Default is same directory as input.

    Returns the path of the decompressed file as ForgeChar.
    """
    if isinstance(filename, ForgeChar):
        filename = filename.to_str()
    filename = str(filename)
    if destdir is None:
        destdir = os.path.dirname(filename) or '.'
    if isinstance(destdir, ForgeChar):
        destdir = destdir.to_str()
    destdir = str(destdir)

    base = os.path.basename(filename)
    if base.endswith('.gz'):
        outname = base[:-3]
    else:
        outname = base + '.out'
    outpath = os.path.join(destdir, outname)

    with gzip.open(filename, 'rb') as fin:
        with open(outpath, 'wb') as fout:
            shutil.copyfileobj(fin, fout)
    return _wrap_str(outpath)


def forge_bunzip2(filename, destdir=None):
    """Decompress a bzip2 file.

    Parameters
    ----------
    filename : str or ForgeChar
        Path to the .bz2 file.
    destdir : str or ForgeChar, optional
        Destination directory. Default is same directory as input.

    Returns the path of the decompressed file as ForgeChar.
    """
    if isinstance(filename, ForgeChar):
        filename = filename.to_str()
    filename = str(filename)
    if destdir is None:
        destdir = os.path.dirname(filename) or '.'
    if isinstance(destdir, ForgeChar):
        destdir = destdir.to_str()
    destdir = str(destdir)

    base = os.path.basename(filename)
    if base.endswith('.bz2'):
        outname = base[:-4]
    else:
        outname = base + '.out'
    outpath = os.path.join(destdir, outname)

    with bz2.open(filename, 'rb') as fin:
        with open(outpath, 'wb') as fout:
            shutil.copyfileobj(fin, fout)
    return _wrap_str(outpath)


# =====================================================================
# System Information
# =====================================================================

def forge_computer():
    """Return computer type string.

    Returns a ForgeChar like 'x86_64-pc-linux-gnu' or similar.
    """
    machine = platform.machine()
    system = platform.system().lower()
    if system == 'linux':
        arch_str = f'{machine}-pc-linux-gnu'
    elif system == 'darwin':
        arch_str = f'{machine}-apple-darwin'
    elif system == 'windows':
        arch_str = f'{machine}-pc-mswin64'
    else:
        arch_str = f'{machine}-{system}'
    return _wrap_str(arch_str)


def forge_ispc():
    """True if running on a Windows system."""
    return _wrap(np.array(1.0 if platform.system() == 'Windows' else 0.0))


def forge_isunix():
    """True if running on a Unix-like system (Linux, macOS, etc.)."""
    return _wrap(np.array(1.0 if os.name == 'posix' else 0.0))


def forge_ismac():
    """True if running on macOS."""
    return _wrap(np.array(1.0 if platform.system() == 'Darwin' else 0.0))


def forge_ver():
    """Return version information as a ForgeStruct.

    Fields: Name, Version, Release, Date.
    """
    return ForgeStruct(
        Name=_wrap_str('Forge'),
        Version=_wrap_str('1.0.0'),
        Release=_wrap_str(''),
        Date=_wrap_str(''),
    )


def forge_version_str():
    """Return version string."""
    return _wrap_str('1.0.0')


def forge_memory_info():
    """Return memory information as a ForgeStruct.

    Fields: MaxPossibleArrayBytes, MemAvailableAllArrays,
    MemUsedByForge (all in bytes, approximate).
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        return ForgeStruct(
            MaxPossibleArrayBytes=_wrap(np.array(float(vm.available))),
            MemAvailableAllArrays=_wrap(np.array(float(vm.available))),
            MemUsedByForge=_wrap(np.array(float(vm.used))),
        )
    except ImportError:
        # Fallback: report zeros
        return ForgeStruct(
            MaxPossibleArrayBytes=_wrap(np.array(0.0)),
            MemAvailableAllArrays=_wrap(np.array(0.0)),
            MemUsedByForge=_wrap(np.array(0.0)),
        )


def forge_isdeployed():
    """True if running in deployed (compiled) mode.

    Always returns 0 (false) since Forge runs interpreted.
    """
    return _wrap(np.array(0.0))


def forge_license_info(feature=None):
    """Return license information.

    Parameters
    ----------
    feature : str or ForgeChar, optional
        Feature to check (ignored, always returns 'Forge').

    Returns a ForgeStruct with license fields.
    """
    return ForgeStruct(
        license_number=_wrap_str('0'),
        type=_wrap_str('academic'),
        feature=_wrap_str(str(feature) if feature else 'Forge'),
        status=_wrap_str('active'),
    )


# =====================================================================
# Registry
# =====================================================================

FILEIO_REGISTRY = {
    # CSV / delimited I/O
    "csvread": forge_csvread,
    "csvwrite": forge_csvwrite,
    "dlmwrite": forge_dlmwrite,
    "fileread": forge_fileread,
    "importdata": forge_importdata,
    "is_valid_file_id": forge_is_valid_file_id,
    "beep": forge_beep,

    # File system operations
    "copyfile": forge_copyfile,
    "delete": forge_delete_file,
    "dir": forge_dir_list,
    "fileattrib": forge_fileattrib,
    "fileparts": forge_fileparts,
    "fullfile": forge_fullfile,
    "mkdir": forge_mkdir,
    "movefile": forge_movefile,
    "isfile": forge_isfile,
    "isfolder": forge_isfolder,
    "ls": forge_ls,

    # Archive operations
    "tar": forge_tar,
    "untar": forge_untar,
    "zip": forge_zip_file,
    "unzip": forge_unzip_file,
    "gunzip": forge_gunzip,
    "bunzip2": forge_bunzip2,

    # System information
    "computer": forge_computer,
    "ispc": forge_ispc,
    "isunix": forge_isunix,
    "ismac": forge_ismac,
    "ver": forge_ver,
    "version": forge_version_str,
    "memory": forge_memory_info,
    "isdeployed": forge_isdeployed,
    "license": forge_license_info,
}
