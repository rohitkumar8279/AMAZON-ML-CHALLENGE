"""High-recall, capped candidate generation and pair features for Kaggle runs.

No external business data is used. Candidate channels are deliberately kept
separate as provenance features, and the hard per-source cap is applied only
after a cheap lexical ranking step.
"""

from __future__ import annotations

import math
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Union

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from unidecode import unidecode


_SPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_DIGITS = re.compile(r"\d+")
_SUFFIX = re.compile(
    r"\b(?:inc|incorporated|llc|corp|corporation|ltd|limited|co|company|plc|pvt|private|gmbh|sarl)\b"
)
_ADDRESS_REPLACEMENTS = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "hwy": "highway", "ste": "suite",
    "apt": "apartment", "fl": "floor", "sq": "square", "ct": "court",
}

CHANNELS = (
    "exact_name", "compact_name", "core_name", "name_token",
    "name_prefix", "name_number", "number_address_token", "translit_name", "translit_address",
)
FEATURE_NAMES = (
    "source2", "name_exact", "name_compact_exact", "name_ratio", "name_wratio",
    "name_translit_ratio",
    "name_token_sort", "name_token_set", "name_jaro_winkler", "name_levenshtein",
    "name_jaccard", "name_min_coverage", "name_weighted_jaccard", "name_tfidf_cosine", "name_weighted_overlap",
    "name_char3_jaccard", "name_char4_jaccard", "name_char5_jaccard", "name_prefix4",
    "name_first_token_exact", "name_length_ratio", "name_token_count_ratio",
    "address_exact", "address_ratio", "address_wratio", "address_translit_ratio", "address_token_sort",
    "address_token_set", "address_jaro_winkler", "address_levenshtein",
    "address_jaccard", "address_min_coverage", "address_weighted_jaccard", "address_tfidf_cosine",
    "address_weighted_overlap", "address_char3_jaccard", "address_char4_jaccard", "address_char5_jaccard",
    "address_length_ratio", "address_token_count_ratio", "number_jaccard",
    "number_overlap_count", "first_number_exact", "number_conflict", "name_address_product",
    "name_address_min", "name_address_max", "country_equal", "channel_exact_name",
    "channel_compact_name", "channel_core_name", "channel_name_token", "channel_name_prefix",
    "channel_name_number", "channel_number_address_token", "channel_translit_name", "channel_translit_address",
    "channel_count", "retrieval_rank_norm",
    "cheap_retrieval_score",
)
CHANNEL_BIT = {name: 1 << index for index, name in enumerate(CHANNELS)}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = _NON_WORD.sub(" ", value)
    return _SPACE.sub(" ", value).strip()


def normalize_name(value: str) -> str:
    return _SPACE.sub(" ", _SUFFIX.sub(" ", normalize_text(value))).strip()


def normalize_address(value: str) -> str:
    tokens = normalize_text(value).split()
    return " ".join(_ADDRESS_REPLACEMENTS.get(token, token) for token in tokens)


def compact(value: str) -> str:
    return "".join(value.split())


def _tokens(value: str) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(value.split()))


_ADDRESS_ANCHOR_STOP = {"street", "road", "avenue", "boulevard", "drive", "lane", "near", "opposite", "main", "city", "district"}


def _address_anchor_tokens(tokens: Sequence[str]) -> Tuple[str, ...]:
    useful = [token for token in tokens if len(token) >= 4 and not token.isdigit() and token not in _ADDRESS_ANCHOR_STOP]
    # Limit address composite blocks so long addresses do not multiply index size.
    return tuple(sorted(useful, key=lambda token: (-len(token), token))[:2])


def _numbers(value: str) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(_DIGITS.findall(value or "")))


@dataclass(frozen=True, slots=True)
class Record:
    entity_id: str
    country: str
    name: str
    address: str
    source: int
    name_tokens: Tuple[str, ...]
    address_tokens: Tuple[str, ...]
    numbers: Tuple[str, ...]
    first_number: str
    compact_name: str
    roman_name: str
    roman_address: str
    roman_name_tokens: Tuple[str, ...]
    roman_address_tokens: Tuple[str, ...]


def make_record(entity_id: str, name: str, address: str, country: str, source: int) -> Record:
    norm_name = normalize_name(name)
    norm_address = normalize_address(address)
    numbers = _numbers(norm_address)
    roman_name = norm_name if norm_name.isascii() else normalize_name(unidecode(norm_name))
    roman_address = norm_address if norm_address.isascii() else normalize_address(unidecode(norm_address))
    roman_name_tokens = _tokens(roman_name) if roman_name != norm_name else _tokens(norm_name)
    roman_address_tokens = _tokens(roman_address) if roman_address != norm_address else _tokens(norm_address)
    return Record(
        entity_id=entity_id,
        country=sys.intern((country or "").strip().casefold()),
        name=norm_name,
        address=norm_address,
        source=source,
        name_tokens=_tokens(norm_name),
        address_tokens=_tokens(norm_address),
        numbers=numbers,
        first_number=numbers[0] if numbers else "",
        compact_name=compact(norm_name),
        roman_name=roman_name,
        roman_address=roman_address,
        roman_name_tokens=roman_name_tokens,
        roman_address_tokens=roman_address_tokens,
    )


@dataclass(frozen=True, slots=True)
class Candidate:
    record_index: int
    channel_mask: int
    retrieval_score: float
    retrieval_rank: int


class CandidateIndex:
    """Frequency-capped inverted index for one target source.

    A block that exceeds ``max_posting`` is removed in full. Keeping the first
    N rows of a frequent block would make retrieval depend on file order.
    """

    def __init__(self, max_posting: int = 200):
        if max_posting < 2:
            raise ValueError("max_posting must be at least 2")
        self.max_posting = max_posting
        self.records: List[Record] = []
        self.postings: Dict[int, Dict[str, Dict[str, Union[int, List[int]]]]] = {}
        self.overflow: Dict[int, Dict[str, set[str]]] = {}
        self.name_df: Dict[str, int] = defaultdict(int)
        self.address_df: Dict[str, int] = defaultdict(int)

    def _add(self, channel: str, country: str, key: str, record_index: int) -> None:
        if not key or not country:
            return
        channel_id = CHANNEL_BIT[channel]
        overflow_for_country = self.overflow.setdefault(channel_id, {}).setdefault(country, set())
        if key in overflow_for_country:
            return
        postings_for_country = self.postings.setdefault(channel_id, {}).setdefault(country, {})
        posting = postings_for_country.get(key)
        if posting is None:
            # Most exact/core/composite blocks are singletons. Keeping their
            # target index as a scalar avoids millions of one-item Python lists.
            postings_for_country[key] = record_index
            return
        if isinstance(posting, int):
            postings_for_country[key] = [posting, record_index]
            return
        if len(posting) >= self.max_posting:
            del postings_for_country[key]
            overflow_for_country.add(key)
            return
        posting.append(record_index)

    def add(self, record: Record) -> None:
        index = len(self.records)
        self.records.append(record)
        for token in record.name_tokens:
            self.name_df[token] += 1
        for token in record.address_tokens:
            self.address_df[token] += 1

        country = record.country
        name_tokens = record.name_tokens
        self._add("exact_name", country, record.name, index)
        self._add("compact_name", country, record.compact_name if len(record.compact_name) >= 6 else "", index)
        if len(name_tokens) >= 2:
            self._add("core_name", country, "\x1f".join(sorted(name_tokens[:2])), index)
        if len(record.name) >= 4:
            self._add("name_prefix", country, record.name[:4], index)
        for token in name_tokens:
            if len(token) >= 4:
                self._add("name_token", country, token, index)
        if record.first_number:
            if name_tokens:
                self._add("name_number", country, record.first_number + "\x1f" + name_tokens[0], index)
            for token in _address_anchor_tokens(record.address_tokens):
                self._add("number_address_token", country, record.first_number + "\x1f" + token, index)
        if record.roman_name != record.name:
            self._add("translit_name", country, "F\x1f" + record.roman_name, index)
            for token in record.roman_name_tokens:
                if len(token) >= 4:
                    self._add("translit_name", country, "T\x1f" + token, index)
        if record.first_number and record.roman_address != record.address:
            for token in _address_anchor_tokens(record.roman_address_tokens):
                self._add("translit_address", country, record.first_number + "\x1f" + token, index)

    def _lookup(self, channel: str, country: str, key: str) -> Sequence[int]:
        if not key:
            return ()
        posting = self.postings.get(CHANNEL_BIT[channel], {}).get(country, {}).get(key)
        if posting is None:
            return ()
        if isinstance(posting, int):
            return (posting,)
        return posting

    @property
    def block_count(self) -> int:
        return sum(len(blocks) for countries in self.postings.values() for blocks in countries.values())

    @property
    def pruned_count(self) -> int:
        return sum(len(blocks) for countries in self.overflow.values() for blocks in countries.values())

    def query(self, query: Record, cap: int = 64) -> List[Candidate]:
        if cap < 1:
            return []
        if not query.country:
            return []

        found: Dict[int, int] = {}

        def collect(channel: str, key: str) -> None:
            if not key:
                return
            bit = CHANNEL_BIT[channel]
            for index in self._lookup(channel, query.country, key):
                found[index] = found.get(index, 0) | bit

        collect("exact_name", query.name)
        collect("compact_name", query.compact_name if len(query.compact_name) >= 6 else "")
        if len(query.name_tokens) >= 2:
            collect("core_name", "\x1f".join(sorted(query.name_tokens[:2])))
        collect("name_prefix", query.name[:4] if len(query.name) >= 4 else "")

        rare_name_tokens = sorted(
            (token for token in query.name_tokens if len(token) >= 4 and self._lookup("name_token", query.country, token)),
            key=lambda token: (len(self._lookup("name_token", query.country, token)), token),
        )
        for token in rare_name_tokens[:3]:
            collect("name_token", token)

        collect("translit_name", "F\x1f" + query.roman_name)
        rare_roman_name_tokens = sorted(
            (token for token in query.roman_name_tokens if len(token) >= 4 and self._lookup("translit_name", query.country, "T\x1f" + token)),
            key=lambda token: (len(self._lookup("translit_name", query.country, "T\x1f" + token)), token),
        )
        for token in rare_roman_name_tokens[:2]:
            collect("translit_name", "T\x1f" + token)

        # When only the query needs transliteration, probe the ordinary index
        # with its romanized form so Latin target records remain reachable.
        if query.roman_name != query.name:
            collect("exact_name", query.roman_name)
            collect("compact_name", compact(query.roman_name) if len(compact(query.roman_name)) >= 6 else "")
            roman_tokens = _tokens(query.roman_name)
            if len(roman_tokens) >= 2:
                collect("core_name", "\x1f".join(sorted(roman_tokens[:2])))
            if len(query.roman_name) >= 4:
                collect("name_prefix", query.roman_name[:4])
            for token in sorted(
                (token for token in roman_tokens if len(token) >= 4 and self._lookup("name_token", query.country, token))
            )[:2]:
                collect("name_token", token)
            if query.first_number and roman_tokens:
                collect("name_number", query.first_number + "\x1f" + roman_tokens[0])

        if query.first_number and query.name_tokens:
            collect("name_number", query.first_number + "\x1f" + query.name_tokens[0])
        if query.first_number:
            rare_address_tokens = sorted(
                (token for token in _address_anchor_tokens(query.address_tokens) if self._lookup("number_address_token", query.country, query.first_number + "\x1f" + token)),
                key=lambda token: (
                    len(self._lookup("number_address_token", query.country, query.first_number + "\x1f" + token)),
                    token,
                ),
            )
            for token in rare_address_tokens[:2]:
                collect("number_address_token", query.first_number + "\x1f" + token)
            rare_roman_address_tokens = sorted(
                (token for token in _address_anchor_tokens(query.roman_address_tokens) if self._lookup("translit_address", query.country, query.first_number + "\x1f" + token)),
                key=lambda token: (
                    len(self._lookup("translit_address", query.country, query.first_number + "\x1f" + token)),
                    token,
                ),
            )
            for token in rare_roman_address_tokens[:2]:
                collect("translit_address", query.first_number + "\x1f" + token)
            if query.roman_address != query.address:
                for token in _address_anchor_tokens(query.roman_address_tokens):
                    if not self._lookup("number_address_token", query.country, query.first_number + "\x1f" + token):
                        continue
                    collect("number_address_token", query.first_number + "\x1f" + token)

        if not found:
            return []

        ranked: List[Tuple[float, int, int]] = []
        query_name = set(query.name_tokens)
        query_address = set(query.address_tokens)
        query_numbers = set(query.numbers)
        for index, channel_mask in found.items():
            candidate = self.records[index]
            shared_name = query_name.intersection(candidate.name_tokens)
            shared_address = query_address.intersection(candidate.address_tokens)
            shared_roman_name = set(query.roman_name_tokens).intersection(candidate.roman_name_tokens)
            # Rarity-weighted token agreement is a cheap proxy for TF-IDF cosine.
            name_weight = sum(math.log1p(len(self.records) / (1 + self.name_df.get(token, 0))) for token in shared_name)
            address_weight = sum(math.log1p(len(self.records) / (1 + self.address_df.get(token, 0))) for token in shared_address)
            score = (
                2.0 * name_weight + address_weight + 1.5 * len(shared_roman_name)
                + 2.0 * bool(query_numbers.intersection(candidate.numbers))
                + 1.5 * bool(query.name and candidate.name and (query.name in candidate.name or candidate.name in query.name))
                + 0.75 * bool(query.name[:4] and query.name[:4] == candidate.name[:4])
                + 0.45 * channel_mask.bit_count()
            )
            ranked.append((score, index, channel_mask))

        ranked.sort(key=lambda item: (-item[0], self.records[item[1]].entity_id))
        return [
            Candidate(index, mask, score, rank)
            for rank, (score, index, mask) in enumerate(ranked[:cap], start=1)
        ]

    def idf(self, token: str, field: str) -> float:
        counts = self.name_df if field == "name" else self.address_df
        return math.log((len(self.records) + 1) / (counts.get(token, 0) + 1)) + 1.0


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 0.0


def _min_coverage(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def _weighted_scores(left: Iterable[str], right: Iterable[str], index: CandidateIndex, field: str) -> Tuple[float, float]:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0, 0.0
    overlap = sum(index.idf(token, field) for token in a & b)
    union = sum(index.idf(token, field) for token in a | b)
    return (overlap / union if union else 0.0), overlap


def _tfidf_cosine(left: Iterable[str], right: Iterable[str], index: CandidateIndex, field: str) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    weights_a = {token: index.idf(token, field) for token in a}
    weights_b = {token: index.idf(token, field) for token in b}
    dot = sum(weights_a[token] * weights_b[token] for token in a & b)
    norm_a = math.sqrt(sum(weight * weight for weight in weights_a.values()))
    norm_b = math.sqrt(sum(weight * weight for weight in weights_b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _char_jaccard(left: str, right: str, n: int) -> float:
    if len(left) < n or len(right) < n:
        return 0.0
    a = {left[i:i + n] for i in range(len(left) - n + 1)}
    b = {right[i:i + n] for i in range(len(right) - n + 1)}
    return len(a & b) / len(a | b) if a or b else 0.0


def _length_ratio(a: int, b: int) -> float:
    return min(a, b) / max(a, b) if max(a, b) else 0.0


def pair_features(query: Record, target: Record, candidate: Candidate, index: CandidateIndex) -> List[float]:
    name_a, name_b = query.name, target.name
    addr_a, addr_b = query.address, target.address
    name_tokens_a, name_tokens_b = query.name_tokens, target.name_tokens
    addr_tokens_a, addr_tokens_b = query.address_tokens, target.address_tokens
    name_weighted_j, name_weighted_overlap = _weighted_scores(name_tokens_a, name_tokens_b, index, "name")
    addr_weighted_j, addr_weighted_overlap = _weighted_scores(addr_tokens_a, addr_tokens_b, index, "address")
    name_ratio = fuzz.ratio(name_a, name_b) / 100.0 if name_a and name_b else 0.0
    addr_ratio = fuzz.ratio(addr_a, addr_b) / 100.0 if addr_a and addr_b else 0.0
    number_intersection = set(query.numbers) & set(target.numbers)
    number_union = set(query.numbers) | set(target.numbers)
    number_jaccard = len(number_intersection) / len(number_union) if number_union else 0.0
    channel_values = [float(bool(candidate.channel_mask & CHANNEL_BIT[channel])) for channel in CHANNELS]
    features = [
        float(target.source == 2),
        float(bool(name_a) and name_a == name_b),
        float(bool(query.compact_name) and query.compact_name == target.compact_name),
        name_ratio,
        fuzz.WRatio(name_a, name_b) / 100.0 if name_a and name_b else 0.0,
        fuzz.ratio(query.roman_name, target.roman_name) / 100.0 if query.roman_name and target.roman_name else 0.0,
        fuzz.token_sort_ratio(name_a, name_b) / 100.0 if name_a and name_b else 0.0,
        fuzz.token_set_ratio(name_a, name_b) / 100.0 if name_a and name_b else 0.0,
        JaroWinkler.normalized_similarity(name_a, name_b) if name_a and name_b else 0.0,
        Levenshtein.normalized_similarity(name_a, name_b) if name_a and name_b else 0.0,
        _jaccard(name_tokens_a, name_tokens_b),
        _min_coverage(name_tokens_a, name_tokens_b),
        name_weighted_j,
        _tfidf_cosine(name_tokens_a, name_tokens_b, index, "name"),
        name_weighted_overlap,
        _char_jaccard(name_a, name_b, 3),
        _char_jaccard(name_a, name_b, 4),
        _char_jaccard(name_a, name_b, 5),
        float(len(name_a) >= 4 and len(name_b) >= 4 and name_a[:4] == name_b[:4]),
        float(bool(name_tokens_a) and bool(name_tokens_b) and name_tokens_a[0] == name_tokens_b[0]),
        _length_ratio(len(name_a), len(name_b)),
        _length_ratio(len(name_tokens_a), len(name_tokens_b)),
        float(bool(addr_a) and addr_a == addr_b),
        addr_ratio,
        fuzz.WRatio(addr_a, addr_b) / 100.0 if addr_a and addr_b else 0.0,
        fuzz.ratio(query.roman_address, target.roman_address) / 100.0 if query.roman_address and target.roman_address else 0.0,
        fuzz.token_sort_ratio(addr_a, addr_b) / 100.0 if addr_a and addr_b else 0.0,
        fuzz.token_set_ratio(addr_a, addr_b) / 100.0 if addr_a and addr_b else 0.0,
        JaroWinkler.normalized_similarity(addr_a, addr_b) if addr_a and addr_b else 0.0,
        Levenshtein.normalized_similarity(addr_a, addr_b) if addr_a and addr_b else 0.0,
        _jaccard(addr_tokens_a, addr_tokens_b),
        _min_coverage(addr_tokens_a, addr_tokens_b),
        addr_weighted_j,
        _tfidf_cosine(addr_tokens_a, addr_tokens_b, index, "address"),
        addr_weighted_overlap,
        _char_jaccard(addr_a, addr_b, 3),
        _char_jaccard(addr_a, addr_b, 4),
        _char_jaccard(addr_a, addr_b, 5),
        _length_ratio(len(addr_a), len(addr_b)),
        _length_ratio(len(addr_tokens_a), len(addr_tokens_b)),
        number_jaccard,
        float(len(number_intersection)),
        float(bool(query.first_number) and query.first_number == target.first_number),
        float(bool(query.numbers) and bool(target.numbers) and not number_intersection),
        name_ratio * addr_ratio,
        min(name_ratio, addr_ratio),
        max(name_ratio, addr_ratio),
        float(bool(query.country) and query.country == target.country),
        *channel_values,
        float(candidate.channel_mask.bit_count()),
        1.0 / (1.0 + candidate.retrieval_rank),
        candidate.retrieval_score,
    ]
    if len(features) != len(FEATURE_NAMES):
        raise AssertionError(f"Feature schema mismatch: {len(features)} != {len(FEATURE_NAMES)}")
    return features
