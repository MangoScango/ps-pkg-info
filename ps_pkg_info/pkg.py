"""PS4 and PS5 PKG metadata parser.

Both platforms use a big-endian ``\\x7FCNT`` container: a 0x5A0 header, a
0x20-byte entry table, and unencrypted metadata entries.

* PS4 packages are a bare CNT and carry ``param.sfo`` (entry 0x1000).
* PS5 packages are a little-endian ``\\x7FFIH`` image wrapping the CNT at an
  embedded offset, and carry ``param.json`` (entry 0x2000). Entry offsets are
  relative to the CNT, so the embedded offset is added when parsing.

Only the header, entry table and unencrypted metadata entries are read. The
encrypted PFS image after them is never touched.
"""

from __future__ import annotations

import json
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# --- PKG constants -----------------------------------------------------------

PKG_MAGIC = b"\x7fCNT"      # bare metadata container (PS4 pkg, or PS5 CNT)
FIH_MAGIC = b"\x7fFIH"      # PS5 finalized image (little-endian header)
LIH_MAGIC = b"\x7fLIH"      # PS5 intermediate image (little-endian header)

# CNT header field offsets (big-endian). Shared by PS4 and PS5.
_OFF_MAGIC = 0x00
_OFF_ENTRY_COUNT = 0x10
_OFF_ENTRY_TABLE_OFFSET = 0x18
_OFF_CONTENT_ID = 0x40
_OFF_DRM_TYPE = 0x70
_OFF_CONTENT_TYPE = 0x74
_OFF_CONTENT_FLAGS = 0x78
_OFF_PFS_IMAGE_OFFSET = 0x410
_OFF_PACKAGE_SIZE = 0x430  # declared total package size (u64 BE)

# FIH header field offsets (little-endian).
_FIH_SIGNED_BYTE = 0x05
_FIH_FORMAT_VERSION = 0x06    # u16
_FIH_PFS_IMAGE_OFFSET = 0x10  # u64
_FIH_PFS_IMAGE_SIZE = 0x18    # u64
_FIH_SBLOCK_OFFSET = 0x20     # u64, absolute offset of the outer superblock
_FIH_SBLOCK_SIZE = 0x28       # u64
_FIH_EMBEDDED_CNT_OFFSET = 0x58
_LIH_EMBEDDED_CNT_OFFSET = 0x30

# The outer PFS superblock seed is random in a stock build. fpkg-style builds
# (PLAINTEXT_NOAUTH) write a fixed marker there instead, and nothing else in
# the superblock differs.
_SBLOCK_MAGIC_OFFSET = 0x08
_SBLOCK_MAGIC = 20130315
_SBLOCK_SEED_OFFSET = 0x370
_PLAINTEXT_NOAUTH_MARKER = b"PPRPLAIN-NOAUTH!"

# The 0x06 version field is little-endian in FIH/LIH headers.
_FIH_REQUIRED_VERSION = 3
_LIH_REQUIRED_VERSION = 1

_CONTENT_ID_SIZE = 0x30
_ENTRY_SIZE = 0x20
_HEADER_SIZE = 0x5A0

ENTRY_DIGESTS = 0x00000001
ENTRY_ENTRY_KEYS = 0x00000010
ENTRY_IMAGE_KEY = 0x00000020
ENTRY_GENERAL_DIGESTS = 0x00000080
ENTRY_METAS = 0x00000100
ENTRY_ENTRY_NAMES = 0x00000200
ENTRY_IMAGEDIGS = 0x0000040A
ENTRY_PARAM_SFO = 0x00001000    # PS4
ENTRY_PIC1_PNG = 0x00001006
ENTRY_ICON0_PNG = 0x00001200
ENTRY_PIC0_PNG = 0x00001220
ENTRY_PARAM_JSON = 0x00002000   # PS5
ENTRY_PLAYGO_HASH_TABLE_DAT = 0x00002010
ENTRY_PLAYGO_FICM_DAT = 0x00002011
ENTRY_PLAYGO_CHUNK_DAT = 0x00001001      # base package
ENTRY_APP_PLAYGO_CHUNK_DAT = 0x00001008  # update package
ENTRY_PLAYGO_SCENARIO_JSON = 0x00003000
ENTRY_PLAYGO_SCENARIO_PNG_BASE = 0x00003010  # plus the scenario id

# A PS4 base and update are compatible when the DIGESTS-table digests of the
# playgo-chunk entries match (0x1001 for a base, 0x1008 for an update).
_DIGEST_SIZE = 0x20
_CONTENT_FLAGS_KIND_MASK = 0x0F000000
_CONTENT_FLAGS_BASE = 0x0A000000
_CONTENT_FLAGS_UPDATE = 0x02000000

CONTENT_TYPES = {
    0x1A: "GD",  # app, patch or remaster
    0x1B: "AC",  # additional content or theme
    0x1C: "AL",  # additional content, no data
    0x1E: "DP",  # delta patch
}

# PS5 types follow the PS4 scheme shifted to 0x20. They classify a package more
# reliably than the content flags, which are identical for base games and DLC.
CONTENT_TYPES_PS5 = {
    0x20: "GD",
    0x21: "AC",
    0x22: "AL",
    0x23: "DP",
}

_PS5_CONTENT_TYPE_MIN = 0x20

_FLAG_DELTA_PATCH = 0x41000000
_FLAG_NON_GAME = 0x04000000

# Header flags (0x04) bit 31 marks Sony finalization. It separates retail from
# fpkg on PS4 and bare CNTs; FIH images use the signed byte instead.
_FLAG_FINALIZED = 0x80000000
# FIH signed byte (0x05): 0x80 is retail, 0x00 is debug or fpkg.
_FIH_SIGNED_RETAIL = 0x80

_SFO_UTF8_SPECIAL = 0x0004
_SFO_UTF8 = 0x0204
_SFO_INTEGER = 0x0404


class PkgError(Exception):
    """Raised when a file is not a valid PS4/PS5 PKG or is truncated."""


# --- Byte sources ------------------------------------------------------------

class ByteSource(ABC):
    """Random-access read-only byte source."""

    @abstractmethod
    def read(self, offset: int, size: int) -> bytes:
        """Return exactly ``size`` bytes starting at ``offset``."""

    def close(self) -> None:  # pragma: no cover
        pass

    def __enter__(self) -> "ByteSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class FileSource(ByteSource):
    """Reads from a local file on disk."""

    def __init__(self, path: str):
        self._path = path
        self._f = open(path, "rb")

    def read(self, offset: int, size: int) -> bytes:
        self._f.seek(offset)
        data = self._f.read(size)
        if len(data) != size:
            raise PkgError(
                f"unexpected EOF: wanted {size} bytes at {offset:#x}, got {len(data)}"
            )
        return data

    def close(self) -> None:
        self._f.close()


class ConcatSource(ByteSource):
    """Reads an ordered list of ``(path, size)`` split parts as one stream."""

    def __init__(self, parts):
        self._paths = []
        self._starts = []
        self._sizes = []
        total = 0
        for path, size in parts:
            self._paths.append(path)
            self._starts.append(total)
            self._sizes.append(size)
            total += size
        self.total_size = total
        self._open = {}  # part index -> handle, opened on first read

    def _file(self, i: int):
        f = self._open.get(i)
        if f is None:
            f = open(self._paths[i], "rb")
            self._open[i] = f
        return f

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > self.total_size:
            raise PkgError(
                f"read out of range: {size} bytes at {offset:#x} (total {self.total_size:#x})"
            )
        out = bytearray()
        remaining = size
        pos = offset
        i = 0
        while i < len(self._paths) and self._starts[i] + self._sizes[i] <= pos:
            i += 1
        while remaining > 0 and i < len(self._paths):
            part_start = self._starts[i]
            local = pos - part_start
            take = min(remaining, self._sizes[i] - local)
            f = self._file(i)
            f.seek(local)
            chunk = f.read(take)
            if len(chunk) != take:
                raise PkgError(f"short read in part {self._paths[i]}")
            out += chunk
            remaining -= take
            pos += take
            i += 1
        if remaining:
            raise PkgError("unexpected EOF across split parts")
        return bytes(out)

    def close(self) -> None:
        for f in self._open.values():
            try:
                f.close()
            except OSError:
                pass
        self._open.clear()


class BytesSource(ByteSource):
    """Reads from an in-memory buffer."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self, offset: int, size: int) -> bytes:
        chunk = self._data[offset : offset + size]
        if len(chunk) != size:
            raise PkgError(
                f"unexpected EOF: wanted {size} bytes at {offset:#x}, got {len(chunk)}"
            )
        return chunk


# --- Data structures ---------------------------------------------------------

@dataclass
class PkgEntry:
    """One record from the PKG entry table (0x20 bytes)."""

    id: int
    name_offset: int
    flags1: int
    flags2: int
    offset: int
    size: int
    index: int = 0  # position in the entry table; indexes the DIGESTS table

    @property
    def encrypted(self) -> bool:
        return (self.flags1 & 0x80000000) != 0

    @property
    def key_index(self) -> int:
        return (self.flags2 & 0xF000) >> 12


@dataclass
class Pkg:
    """Parsed PKG metadata (PS4 or PS5)."""

    platform: str  # "PS4" or "PS5"
    content_id: str
    drm_type: int
    content_type: int
    content_flags: int
    entry_count: int
    entry_table_offset: int
    pfs_image_offset: int
    cnt_base: int = 0  # offset of the CNT; 0 for a bare CNT
    container: str = "CNT"  # "CNT", "FIH" or "LIH"
    edition: str = ""  # Retail / Debug / fpkg
    digests_base: int = 0  # absolute offset of the DIGESTS table
    package_size: int = 0  # size declared in the header
    entries: Dict[int, PkgEntry] = field(default_factory=dict)
    sfo: Dict[str, object] = field(default_factory=dict)    # PS4
    param: Dict[str, object] = field(default_factory=dict)  # PS5

    _source: Optional[ByteSource] = None

    @property
    def title(self) -> Optional[str]:
        if self.platform == "PS5":
            return _ps5_title(self.param)
        return self.sfo.get("TITLE")

    @property
    def title_id(self) -> Optional[str]:
        if self.platform == "PS5":
            return self.param.get("titleId") or title_id_from_content_id(self.content_id)
        return self.sfo.get("TITLE_ID")

    @property
    def version(self) -> Optional[str]:
        if self.platform == "PS5":
            # contentVersion is the patched version; masterVersion the original.
            return self.param.get("contentVersion") or self.param.get("masterVersion")
        # Updates carry APP_VER, base packages VERSION.
        return self.sfo.get("APP_VER") or self.sfo.get("VERSION")

    @property
    def min_sdk(self) -> Optional[str]:
        """Minimum system software version to install, e.g. "11.50", or None."""
        if self.platform == "PS5":
            return _version_from_hex(self.param.get("requiredSystemSoftwareVersion"))
        return _version_from_u32(self.sfo.get("SYSTEM_VER"))

    @property
    def min_ps5_fw(self) -> Optional[str]:
        """Lowest PS5 system software able to run this package."""
        if self.platform == "PS5":
            return self.min_sdk
        return ps5_firmware_for_ps4_system_ver(self.sfo.get("SYSTEM_VER"))

    @property
    def category(self) -> Optional[str]:
        if self.platform == "PS5":
            v = self.param.get("applicationCategoryType")
            return str(v) if v is not None else None
        return self.sfo.get("CATEGORY")

    @property
    def content_type_name(self) -> str:
        table = CONTENT_TYPES_PS5 if self.platform == "PS5" else CONTENT_TYPES
        return table.get(self.content_type, f"0x{self.content_type:X}")

    @property
    def kind(self) -> str:
        """High-level package kind: Game / Update / DLC / App / Other."""
        if self.platform == "PS5":
            return classify_kind_ps5(self.content_flags, self.content_type, self.container)
        # Base and update share content_type GD, so PS4 needs the SFO CATEGORY.
        return classify_kind(self.category, self.content_type)

    @property
    def region(self) -> str:
        return region_from_content_id(self.content_id)

    @property
    def is_delta_patch(self) -> bool:
        """True for a delta patch."""
        if self.platform == "PS5":
            return self.content_type == 0x23 or (
                (self.content_flags & _FLAG_DELTA_PATCH) == _FLAG_DELTA_PATCH
            )
        return self.content_type == 0x1E  # PS4 DP

    def is_metadata_fragment(self, available_size: int) -> bool:
        """True for a metadata-only SC tail: the declared package size is more
        than twice the bytes available. Only a bare CNT can be one, since an LIH
        declares the size of the package it builds into."""
        if self.cnt_base != 0:
            return False
        return self.package_size > available_size * 2

    def has_icon0(self) -> bool:
        return ENTRY_ICON0_PNG in self.entries

    def read_entry(self, entry_id: int) -> Optional[bytes]:
        """Raw bytes of an entry, or None if absent."""
        entry = self.entries.get(entry_id)
        if entry is None:
            return None
        if self._source is None:
            raise PkgError("no source attached; re-open with Pkg.open()")
        if entry.encrypted:
            raise PkgError(f"entry {entry_id:#x} is encrypted; decryption unsupported")
        return self._source.read(entry.offset, entry.size)

    def read_icon0(self) -> Optional[bytes]:
        return self.read_entry(ENTRY_ICON0_PNG)

    def _playgo_chunk_entry_id(self) -> int:
        """playgo-chunk.dat entry to read: 0x1008 for an update, else 0x1001."""
        if self.kind == "Update" and ENTRY_APP_PLAYGO_CHUNK_DAT in self.entries:
            return ENTRY_APP_PLAYGO_CHUNK_DAT
        return ENTRY_PLAYGO_CHUNK_DAT

    def playgo_scenarios(self) -> List["Scenario"]:
        """PS5 PlayGo scenarios, or [] for PS4 or a package without them.

        Raises PkgError if the table is malformed.
        """
        if self.platform != "PS5":
            return []
        raw = self.read_entry(self._playgo_chunk_entry_id())
        if raw is None:
            return []
        try:
            return parse_playgo_scenarios(raw)
        except ValueError as e:
            raise PkgError(f"malformed playgo-chunk.dat: {e}") from e

    def playgo_scenario_names(self) -> Dict[int, Dict[str, object]]:
        """Display text per scenario id from playgo-scenario.json, or {}."""
        raw = self.read_entry(ENTRY_PLAYGO_SCENARIO_JSON)
        if raw is None:
            return {}
        try:
            return parse_scenario_names(raw)
        except ValueError:
            return {}

    def read_playgo_scenario_image(self, scenario_id: int) -> Optional[bytes]:
        """The 1008x1008 PNG for a scenario (entry 0x3010 + id), or None."""
        return self.read_entry(ENTRY_PLAYGO_SCENARIO_PNG_BASE + scenario_id)

    def marriage_digest(self) -> Optional[str]:
        """Hex digest binding a PS4 package to its base build, or None.

        A base and an update are compatible when their digests match. The
        entry is chosen by ``kind``, not the header content flags, because
        retail bases carry the flags an update would.
        """
        if self.platform != "PS4":
            return None
        if self.content_type == 0x1B:
            return None

        kind = self.kind
        if kind in ("Game", "App"):
            target_id = ENTRY_PLAYGO_CHUNK_DAT
        elif kind == "Update":
            target_id = ENTRY_APP_PLAYGO_CHUNK_DAT
        else:
            return None

        target = self.entries.get(target_id)
        if target is None:
            return None
        if self._source is None:
            raise PkgError("no source attached; re-open with Pkg.open()")

        pos = self.digests_base + target.index * _DIGEST_SIZE
        digest = self._source.read(pos, _DIGEST_SIZE)
        return digest.hex().upper()

    @classmethod
    def open(cls, path: str) -> "Pkg":
        """Parse a PKG from a local file path."""
        return cls.from_source(FileSource(path), keep_source=True)

    @classmethod
    def open_split(cls, parts) -> "Pkg":
        """Parse a split PKG from ordered ``[(path, size), ...]`` parts."""
        return cls.from_source(ConcatSource(parts), keep_source=True)

    @classmethod
    def from_source(cls, source: ByteSource, keep_source: bool = True) -> "Pkg":
        """Parse a PKG from any ByteSource."""
        pkg = _parse(source)
        pkg._source = source if keep_source else None
        return pkg

    def close(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None

    def __enter__(self) -> "Pkg":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --- Parsing -----------------------------------------------------------------

def _u32be(b: bytes, off: int) -> int:
    return struct.unpack_from(">I", b, off)[0]


def _u64be(b: bytes, off: int) -> int:
    return struct.unpack_from(">Q", b, off)[0]


def _cstr(b: bytes) -> str:
    end = b.find(b"\x00")
    if end != -1:
        b = b[:end]
    return b.decode("utf-8", errors="replace")


def _u64le(b: bytes, off: int) -> int:
    return struct.unpack_from("<Q", b, off)[0]


def _locate_cnt(source: ByteSource):
    """Return (cnt_base, fih_signed_byte, container) for a package.

    The signed byte is None unless the container is an FIH.
    """
    head = source.read(0, 0x60)
    magic = head[0:4]
    if magic == PKG_MAGIC:
        # Offset 0x06 holds header flags here, not a format version.
        return 0, None, "CNT"
    if magic == FIH_MAGIC:
        version = struct.unpack_from("<H", head, _FIH_FORMAT_VERSION)[0]
        if version != _FIH_REQUIRED_VERSION:
            raise PkgError(
                f"unsupported FIH format version {version} (expected {_FIH_REQUIRED_VERSION})"
            )
        return _u64le(head, _FIH_EMBEDDED_CNT_OFFSET), head[_FIH_SIGNED_BYTE], "FIH"
    if magic == LIH_MAGIC:
        version = struct.unpack_from("<H", head, _FIH_FORMAT_VERSION)[0]
        if version != _LIH_REQUIRED_VERSION:
            raise PkgError(
                f"unsupported LIH format version {version} (expected {_LIH_REQUIRED_VERSION})"
            )
        return _u64le(head, _LIH_EMBEDDED_CNT_OFFSET), None, "LIH"
    raise PkgError("not a PS4/PS5 PKG (bad magic)")


def _has_plaintext_noauth_marker(source: ByteSource, fih_head: bytes) -> bool:
    """True if the outer PFS superblock seed is the PLAINTEXT_NOAUTH marker.

    ``fih_head`` is the first 0x60 bytes of the FIH. Returns False when the
    superblock is missing, unreadable or lacks the PFS magic.
    """
    sblock_offset = _u64le(fih_head, _FIH_SBLOCK_OFFSET)
    sblock_size = _u64le(fih_head, _FIH_SBLOCK_SIZE)
    if sblock_offset == 0 or sblock_size < _SBLOCK_SEED_OFFSET + len(_PLAINTEXT_NOAUTH_MARKER):
        return False
    try:
        sblock = source.read(sblock_offset, sblock_size)
    except PkgError:
        return False
    if _u64le(sblock, _SBLOCK_MAGIC_OFFSET) != _SBLOCK_MAGIC:
        return False
    seed = sblock[_SBLOCK_SEED_OFFSET : _SBLOCK_SEED_OFFSET + len(_PLAINTEXT_NOAUTH_MARKER)]
    return seed == _PLAINTEXT_NOAUTH_MARKER


@dataclass
class SplitHeader:
    """Facts for pairing a headless main image with its SC tail."""

    kind: str            # "headless_main" | "sc" | "standalone"
    pfs_image_size: int  # equal between a main and its SC
    content_id: str      # empty for a headless main

    @property
    def content_label(self) -> str:
        return content_label_from_content_id(self.content_id)


def probe_split_header(source: ByteSource, total_size: int) -> Optional[SplitHeader]:
    """Classify a file for SC pairing from its first blocks, or None.

    * ``headless_main``: an FIH whose embedded CNT lies past the end of the file.
    * ``sc``: a bare CNT whose PFS image extends past the bytes present.
    * ``standalone``: anything that already contains its own CNT and image.
    """
    try:
        head = source.read(0, 0x60)
    except PkgError:
        return None
    magic = head[0:4]

    if magic == FIH_MAGIC:
        cnt_off = _u64le(head, _FIH_EMBEDDED_CNT_OFFSET)
        pfs_size = _u64le(head, _FIH_PFS_IMAGE_SIZE)
        kind = "headless_main" if cnt_off >= total_size else "standalone"
        return SplitHeader(kind=kind, pfs_image_size=pfs_size, content_id="")

    if magic == PKG_MAGIC:
        try:
            cnt = source.read(0, 0x600)
        except PkgError:
            return None
        content_id = _cstr(cnt[_OFF_CONTENT_ID : _OFF_CONTENT_ID + _CONTENT_ID_SIZE])
        pfs_image_offset = _u64be(cnt, _OFF_PFS_IMAGE_OFFSET)
        pfs_size = _u64be(cnt, _OFF_PFS_IMAGE_OFFSET + 0x08)
        if pfs_image_offset + pfs_size > total_size:
            kind = "sc"
        else:
            kind = "standalone"
        return SplitHeader(kind=kind, pfs_image_size=pfs_size, content_id=content_id)

    return None


def _parse(source: ByteSource) -> Pkg:
    cnt_base, fih_signed, container = _locate_cnt(source)

    is_plaintext_noauth = False
    if fih_signed is not None and fih_signed != _FIH_SIGNED_RETAIL:
        is_plaintext_noauth = _has_plaintext_noauth_marker(source, source.read(0, 0x60))

    # 0x600 covers the header fields through 0x438.
    header = source.read(cnt_base, 0x600)
    if header[_OFF_MAGIC : _OFF_MAGIC + 4] != PKG_MAGIC:
        raise PkgError("embedded container is not a \\x7FCNT")

    header_flags = _u32be(header, 0x04)
    entry_count = _u32be(header, _OFF_ENTRY_COUNT)
    entry_table_offset = _u32be(header, _OFF_ENTRY_TABLE_OFFSET)
    content_id = _cstr(header[_OFF_CONTENT_ID : _OFF_CONTENT_ID + _CONTENT_ID_SIZE])
    drm_type = _u32be(header, _OFF_DRM_TYPE)
    content_type = _u32be(header, _OFF_CONTENT_TYPE)
    content_flags = _u32be(header, _OFF_CONTENT_FLAGS)
    pfs_image_offset = _u64be(header, _OFF_PFS_IMAGE_OFFSET)
    package_size = _u64be(header, _OFF_PACKAGE_SIZE)

    if entry_count > 0x10000:
        raise PkgError(f"implausible entry_count {entry_count}; file likely corrupt")

    table = source.read(cnt_base + entry_table_offset, entry_count * _ENTRY_SIZE)
    entries: Dict[int, PkgEntry] = {}
    digests_base = 0
    for i in range(entry_count):
        base = i * _ENTRY_SIZE
        eid = _u32be(table, base)
        entry = PkgEntry(
            id=eid,
            name_offset=_u32be(table, base + 0x04),
            flags1=_u32be(table, base + 0x08),
            flags2=_u32be(table, base + 0x0C),
            offset=cnt_base + _u32be(table, base + 0x10),
            size=_u32be(table, base + 0x14),
            index=i,
        )
        if i == 0:
            digests_base = entry.offset
        entries.setdefault(eid, entry)

    platform = detect_platform(cnt_base, content_type, entries)

    pkg = Pkg(
        platform=platform,
        content_id=content_id,
        drm_type=drm_type,
        content_type=content_type,
        content_flags=content_flags,
        entry_count=entry_count,
        entry_table_offset=entry_table_offset,
        pfs_image_offset=pfs_image_offset,
        cnt_base=cnt_base,
        container=container,
        edition=detect_edition(fih_signed, header_flags, is_plaintext_noauth),
        digests_base=digests_base,
        package_size=package_size,
        entries=entries,
    )

    if platform == "PS5":
        json_entry = entries.get(ENTRY_PARAM_JSON)
        if json_entry is not None and not json_entry.encrypted:
            raw = source.read(json_entry.offset, json_entry.size)
            try:
                pkg.param = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                pkg.param = {}
    else:
        sfo_entry = entries.get(ENTRY_PARAM_SFO)
        if sfo_entry is not None and not sfo_entry.encrypted:
            sfo_bytes = source.read(sfo_entry.offset, sfo_entry.size)
            try:
                pkg.sfo = parse_sfo(sfo_bytes)
            except PkgError:
                pkg.sfo = {}

    return pkg


def _ps5_title(param: Dict[str, object]) -> Optional[str]:
    """Title from the default language, else any language, else ``titleName``."""
    lp = param.get("localizedParameters")
    if isinstance(lp, dict):
        default_lang = lp.get("defaultLanguage")
        if isinstance(default_lang, str):
            entry = lp.get(default_lang)
            if isinstance(entry, dict) and entry.get("titleName"):
                return str(entry["titleName"])
        for value in lp.values():
            if isinstance(value, dict) and value.get("titleName"):
                return str(value["titleName"])
    name = param.get("titleName")
    return str(name) if name else None


def title_id_from_content_id(content_id: str) -> str:
    """Title id = the segment between the first '-' and the following '_'.

    e.g. ``UP4433-PPSA19639_00-...`` -> ``PPSA19639``.
    """
    if not content_id:
        return ""
    dash = content_id.find("-")
    if dash < 0:
        return ""
    start = dash + 1
    underscore = content_id.find("_", start)
    end = underscore if underscore >= 0 else len(content_id)
    return content_id[start:end]


def content_label_from_content_id(content_id: str) -> str:
    """Content label = the final ``-`` segment of a content id.

    e.g. ``UP4433-PPSA19639_00-LABEL00000000000`` -> ``LABEL00000000000``.
    """
    if not content_id:
        return ""
    dash = content_id.rfind("-")
    return content_id[dash + 1 :] if dash >= 0 else content_id


# (ps5_system_software, ps4_system_ver_stamp), listing only releases that raised
# the PS4 compatibility level.
_PS5_PS4_COMPAT_LEVELS = (
    ("1.00", 0x07590001),
    ("2.00", 0x08050001),
    ("3.00", 0x08540001),
    ("4.00", 0x09040001),
    ("4.50", 0x09090001),
    ("5.00", 0x09590001),
    ("5.02", 0x09690001),
    ("6.00", 0x10090001),
    ("7.01", 0x10590001),
    ("7.40", 0x10790001),
    ("8.00", 0x11090001),
    ("9.00", 0x11590001),
    ("10.01", 0x12090001),
    ("11.00", 0x12590001),
    ("12.00", 0x13090001),
    ("13.60", 0x13590001),
)


def ps5_firmware_for_ps4_system_ver(system_ver) -> Optional[str]:
    """Lowest PS5 system software that hosts a PS4 ``SYSTEM_VER``, or None.

    Only the top two bytes are compared; the rest are patch bits.
    """
    if system_ver is None:
        return None
    try:
        want = (int(system_ver) >> 16) & 0xFFFF
    except (TypeError, ValueError):
        return None
    for ps5_version, stamp in _PS5_PS4_COMPAT_LEVELS:
        if (stamp >> 16) & 0xFFFF >= want:
            return ps5_version
    return None


def _bcd_major_minor(major: int, minor: int) -> str:
    """Format a BCD major/minor byte pair, e.g. (0x11, 0x50) -> "11.50"."""
    return "%x.%02x" % (major, minor)


def _version_from_u32(value) -> Optional[str]:
    """PS4 SYSTEM_VER: BCD major/minor in the top two bytes (0x11500000 -> 11.50)."""
    if value is None:
        return None
    try:
        v = int(value) & 0xFFFFFFFF
    except (TypeError, ValueError):
        return None
    return _bcd_major_minor((v >> 24) & 0xFF, (v >> 16) & 0xFF)


def _version_from_hex(value) -> Optional[str]:
    """PS5 requiredSystemSoftwareVersion: hex string, BCD major/minor in the top
    two bytes ("0x0403000000000000" -> 4.03)."""
    if not isinstance(value, str) or not value:
        return None
    try:
        v = int(value, 16)
    except ValueError:
        return None
    return _bcd_major_minor((v >> 56) & 0xFF, (v >> 48) & 0xFF)


def parse_sfo(data: bytes) -> Dict[str, object]:
    """Parse a param.sfo blob into a name -> int or str dict.

    An optional SCEC wrapper (0x53434543) shifts the SFO start by 0x800.
    """
    start = 0
    if len(data) >= 4 and struct.unpack_from(">I", data, 0)[0] == 0x53434543:
        start = 0x800

    if len(data) < start + 0x14:
        raise PkgError("SFO too small")

    # b"\0PSF" read big-endian.
    if struct.unpack_from(">I", data, start)[0] != 0x00505346:
        raise PkgError("missing SFO magic")

    key_table_start = struct.unpack_from("<I", data, start + 0x08)[0]
    data_table_start = struct.unpack_from("<I", data, start + 0x0C)[0]
    num_values = struct.unpack_from("<i", data, start + 0x10)[0]

    result: Dict[str, object] = {}
    for i in range(num_values):
        rec = start + 0x14 + i * 0x10
        key_offset = struct.unpack_from("<H", data, rec)[0]
        fmt = struct.unpack_from("<H", data, rec + 0x02)[0]
        length = struct.unpack_from("<i", data, rec + 0x04)[0]
        data_offset = struct.unpack_from("<I", data, rec + 0x0C)[0]

        name = _cstr(data[start + key_table_start + key_offset : start + key_table_start + key_offset + 256])
        vpos = start + data_table_start + data_offset

        if fmt == _SFO_INTEGER:
            value: object = struct.unpack_from("<i", data, vpos)[0]
        elif fmt == _SFO_UTF8:
            raw = data[vpos : vpos + (length - 1 if length > 0 else 0)]
            value = raw.decode("utf-8", errors="replace")
        elif fmt == _SFO_UTF8_SPECIAL:
            raw = data[vpos : vpos + length]
            value = raw.decode("utf-8", errors="replace")
        else:
            continue
        result[name] = value

    return result


def classify_kind(category: Optional[str], content_type: int) -> str:
    """Classify a package as Game / Update / DLC / App / Other.

    The SFO CATEGORY decides: ``gp*`` is an update, ``ac`` is DLC, ``gd``,
    ``gc``, ``bd`` and ``gdo`` (PS2 classics) are games, and other ``gd*``
    codes are apps. Without a CATEGORY, content_type decides.
    """
    c = (category or "").lower()
    if c.startswith("gp"):
        return "Update"
    if c == "ac":
        return "DLC"
    if c == "gd" or c == "gc" or c == "bd" or c == "gdo":
        return "Game"
    if c.startswith("gd"):
        return "App"
    if content_type in (0x1B, 0x1C):
        return "DLC"
    if content_type == 0x1E:
        return "Update"
    if content_type == 0x1A:
        return "Game"
    return "Other"


def classify_kind_ps5(content_flags: int, content_type: int, container: str = "FIH") -> str:
    """Classify a PS5 package as Update / DLC / App / Game.

    An LIH is a patch. content_type picks out DLC and delta patches, because
    the content flags are identical for base games and data DLC. The flags are
    only used for the delta-patch and non-game bits.
    """
    if container == "LIH":
        return "Update"
    if content_type == 0x23 or (content_flags & _FLAG_DELTA_PATCH) == _FLAG_DELTA_PATCH:
        return "Update"
    if content_type in (0x21, 0x22):
        return "DLC"
    if content_flags & _FLAG_NON_GAME:
        return "App"
    return "Game"


def detect_platform(cnt_base: int, content_type: int, entries: Dict[int, PkgEntry]) -> str:
    """Return "PS4" or "PS5" from the wrapper, then content_type, then metadata."""
    if cnt_base != 0:
        return "PS5"
    if content_type >= _PS5_CONTENT_TYPE_MIN:
        return "PS5"
    # Some PS4 packages also carry a param.json, so check the SFO first.
    if ENTRY_PARAM_SFO in entries:
        return "PS4"
    if ENTRY_PARAM_JSON in entries:
        return "PS5"
    return "PS4"


def detect_edition(
    fih_signed: Optional[int],
    header_flags: int,
    is_plaintext_noauth: bool = False,
) -> str:
    """Classify a package as Retail / Debug / fpkg.

    An FIH is Retail when signed, otherwise fpkg if it has the
    PLAINTEXT_NOAUTH marker and Debug if not. Other packages are Retail when
    the FINALIZED header flag is set, else fpkg.
    """
    if fih_signed is not None:
        if fih_signed == _FIH_SIGNED_RETAIL:
            return "Retail"
        return "fpkg" if is_plaintext_noauth else "Debug"
    if header_flags & _FLAG_FINALIZED:
        return "Retail"
    return "fpkg"


def region_from_content_id(content_id: str) -> str:
    """Region from the first letter of the content id, e.g. ``UP0006-...``."""
    if not content_id:
        return "Unknown"
    prefix = content_id[0].upper()
    return {
        "U": "US",
        "E": "EU",
        "J": "JP",
        "H": "Asia",
        "K": "KR",
        "I": "INT",
    }.get(prefix, "Unknown")


# --- PlayGo scenarios --------------------------------------------------------

# playgo-chunk.dat ("plgx", little-endian) holds the scenario tables. Display
# text is in playgo-scenario.json (entry 0x3000) and images are entries
# 0x3010+N and 0x3050+N.
#
# * 0x0A u16 chunk count, 0x0E u16 scenario count
# * 0xC0: 7 x (u32 offset, u32 size) section table
# * section 4: 0x20 bytes per scenario: u16 initial count at +0x14, u16 total
#   at +0x16, u32 list offset (section 5) at +0x18, u32 name offset
#   (section 6) at +0x1C
# * section 5: u16 chunk ids per scenario, in launch order, unsorted
# * section 6: NUL-terminated internal scenario names

# Indexes into the section table at 0xC0.
_PLGX_SECTION_SCENARIO_TABLE = 4
_PLGX_SECTION_SCENARIO_LISTS = 5
_PLGX_SECTION_SCENARIO_NAMES = 6

_PLGX_SCENARIO_ENTRY_SIZE = 0x20


@dataclass
class Scenario:
    """One PlayGo scenario: which chunks it needs, and how many before launch."""

    id: int  # scenario table index, shared by the JSON and image entries
    name: str  # internal id, not display text
    initial: int  # leading chunks required before launch
    chunks: List[int]  # in launch order

    @property
    def total(self) -> int:
        return len(self.chunks)

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "initial": self.initial,
            "total": self.total,
            "chunks": list(self.chunks),
        }


def parse_playgo_scenarios(data: bytes) -> List[Scenario]:
    """Scenarios from a playgo-chunk.dat; raises ValueError if malformed."""
    if data[:4] != b"plgx" or len(data) < 0x100:
        raise ValueError("not a plgx playgo-chunk.dat")
    chunk_count, = struct.unpack_from("<H", data, 0x0A)
    count, = struct.unpack_from("<H", data, 0x0E)
    table_at, table_size = struct.unpack_from("<II", data, 0xC0 + _PLGX_SECTION_SCENARIO_TABLE * 8)
    lists_at, lists_size = struct.unpack_from("<II", data, 0xC0 + _PLGX_SECTION_SCENARIO_LISTS * 8)
    names_at, names_size = struct.unpack_from("<II", data, 0xC0 + _PLGX_SECTION_SCENARIO_NAMES * 8)
    if table_size != count * _PLGX_SCENARIO_ENTRY_SIZE or max(
        table_at + table_size, lists_at + lists_size, names_at + names_size
    ) > len(data):
        raise ValueError("scenario sections do not match the header")
    out = []
    for i in range(count):
        entry = table_at + i * _PLGX_SCENARIO_ENTRY_SIZE
        initial, total = struct.unpack_from("<HH", data, entry + 0x14)
        list_off, name_off = struct.unpack_from("<II", data, entry + 0x18)
        if initial > total or list_off + total * 2 > lists_size or name_off >= names_size:
            raise ValueError(f"scenario {i} points outside its sections")
        ids = list(struct.unpack_from(f"<{total}H", data, lists_at + list_off))
        if any(c >= chunk_count for c in ids):
            raise ValueError(f"scenario {i} names a chunk that does not exist")
        end = data.index(b"\0", names_at + name_off, names_at + names_size)
        out.append(Scenario(i, data[names_at + name_off:end].decode("utf-8"), initial, ids))
    return out


def parse_scenario_names(raw: bytes) -> Dict[int, Dict[str, object]]:
    """``{scenario_id: {"title", "description"}}`` from the JSON.

    The scenario named by ``scenarioDefaultId`` also gets ``"default": True``.

    Language preference is en-US, then ``scenarioDefaultLanguage``, then any.
    Raises ValueError if the blob is not the expected JSON.
    """
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise ValueError(f"not playgo-scenario.json: {e}") from e
    scenarios = doc.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("not playgo-scenario.json: missing scenarios list")

    default_language = doc.get("scenarioDefaultLanguage")
    if not isinstance(default_language, str):
        default_language = ""
    default_id = doc.get("scenarioDefaultId")

    out: Dict[int, Dict[str, object]] = {}
    for s in scenarios:
        if not isinstance(s, dict) or not isinstance(s.get("id"), int):
            continue
        langs = {k: v for k, v in s.items() if isinstance(v, dict)}
        pick = None
        for lang in ("en-US", default_language):
            if lang and isinstance(langs.get(lang), dict):
                pick = langs[lang]
                break
        if pick is None and langs:
            pick = next(iter(langs.values()))
        info: Dict[str, object] = {}
        if pick is not None:
            for key in ("title", "description"):
                value = pick.get(key)
                if isinstance(value, str):
                    info[key] = value
        if default_id is not None and s["id"] == default_id:
            info["default"] = True
        out[s["id"]] = info
    return out
