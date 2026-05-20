import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from Levenshtein import ratio as levenshtein_ratio


CROSSREF_WORKS_URL = "https://api.crossref.org/works"
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_MAILTO = "test@example.com"
REQUEST_TIMEOUT_SECONDS = 12
MAX_RETRIES = 4


def clean_whitespace(value: str) -> str:
    """Collapse repeated whitespace so regex parsing sees a predictable string."""
    return re.sub(r"\s+", " ", value or "").strip()


def extract_doi(citation: str) -> Optional[str]:
    """Extract a DOI-like token and trim punctuation often attached in citations."""
    doi_match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", citation, re.IGNORECASE)
    if not doi_match:
        return None
    return doi_match.group(0).rstrip(".,;)")


def split_author_names(author_text: str) -> List[str]:
    """Split a best-effort author segment while preserving names with initials."""
    if not author_text:
        return []

    normalized = re.sub(r"\bet\s+al\.?", "", author_text, flags=re.IGNORECASE)
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"\s+and\s+", ";", normalized, flags=re.IGNORECASE)

    # APA citations use commas inside individual names, so semicolons are safer.
    if ";" in normalized:
        parts = [part.strip(" .,") for part in normalized.split(";")]
    else:
        parts = [normalized.strip(" .,")]

    return [part for part in parts if part]


def extract_title_after_year(citation_without_doi: str, year_match: Optional[re.Match]) -> Optional[str]:
    """Find a likely title segment after the publication year."""
    if not year_match:
        return None

    after_year = citation_without_doi[year_match.end() :]
    after_year = after_year.lstrip(").,;: -")
    if not after_year:
        return None

    quoted = re.match(r"^[\"'“‘](.+?)[\"'”’]", after_year)
    if quoted:
        return quoted.group(1).strip(" .")

    sentence_parts = [part.strip() for part in re.split(r"\.\s+", after_year) if part.strip()]
    if not sentence_parts:
        return None

    # The first substantial sentence after the year is usually the article title.
    first = sentence_parts[0].strip(" .")
    if len(first.split()) >= 3:
        return first
    if len(sentence_parts) > 1:
        return f"{first}. {sentence_parts[1]}".strip(" .")
    return first or None


def extract_quoted_title(citation: str) -> Optional[str]:
    """Return a quoted title when the citation style uses quotation marks."""
    quoted = re.search(r"[\"“‘']([^\"”’']{8,})[\"”’']", citation)
    if quoted:
        return quoted.group(1).strip(" .")
    return None


def parse_citation(citation: str) -> Dict[str, Any]:
    """
    Parse an unstructured academic citation into a compact metadata object.

    Citation formats vary widely, so this function uses conservative regex
    heuristics and returns None-like values rather than raising on weak parses.
    """
    cleaned = clean_whitespace(citation)
    doi = extract_doi(cleaned)
    citation_without_doi = re.sub(re.escape(doi), "", cleaned, flags=re.IGNORECASE) if doi else cleaned

    year_match = re.search(r"\b(19|20)\d{2}\b", citation_without_doi)
    year = year_match.group(0) if year_match else None

    author_segment = ""
    if year_match:
        author_segment = citation_without_doi[: year_match.start()]
        author_segment = re.sub(r"^\s*\[\d+\]\s*", "", author_segment).strip(" .(")

    title = extract_quoted_title(citation_without_doi)
    if not title:
        title = extract_title_after_year(citation_without_doi, year_match)

    # Last fallback: choose a middle sentence when no year-based parse worked.
    if not title:
        parts = [part.strip() for part in re.split(r"\.\s+", citation_without_doi) if part.strip()]
        if len(parts) >= 2:
            title = parts[1].strip(" .")
        elif parts:
            title = parts[0].strip(" .")

    return {
        "Title": title,
        "Authors": split_author_names(author_segment),
        "Year": year,
        "DOI": doi,
        "raw_author_text": author_segment,
    }


def normalize_for_matching(value: Optional[str]) -> str:
    """Lowercase, remove accents, strip punctuation, and collapse spaces."""
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return clean_whitespace(value)


def normalize_author_string(value: Optional[str]) -> str:
    """
    Normalize author names for fuzzy comparison.

    Single-letter initials are dropped and a token-sorted variant is used by the
    scorer so "Smith J Doe A" and "John Smith Alice Doe" compare more fairly.
    """
    normalized = normalize_for_matching(value)
    tokens = [token for token in normalized.split() if len(token) > 1]
    return " ".join(tokens)


def similarity_percent(left: Optional[str], right: Optional[str]) -> float:
    """Return a Levenshtein ratio as a 0-100 percentage."""
    left_norm = normalize_for_matching(left)
    right_norm = normalize_for_matching(right)
    if not left_norm and not right_norm:
        return 100.0
    if not left_norm or not right_norm:
        return 0.0
    return levenshtein_ratio(left_norm, right_norm) * 100


def author_similarity_percent(user_authors: str, official_authors: str) -> float:
    """Compare authors using both sequence order and token-sorted names."""
    left = normalize_author_string(user_authors)
    right = normalize_author_string(official_authors)
    if not left and not right:
        return 100.0
    if not left or not right:
        return 0.0

    ordered = levenshtein_ratio(left, right) * 100
    sorted_left = " ".join(sorted(left.split()))
    sorted_right = " ".join(sorted(right.split()))
    token_sorted = levenshtein_ratio(sorted_left, sorted_right) * 100
    return max(ordered, token_sorted)


def request_with_backoff(
    url: str,
    params: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = MAX_RETRIES,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Call an HTTP JSON API with exponential backoff for 429 and transient errors."""
    last_error = None
    headers = headers or {}

    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                time.sleep(min(sleep_seconds, 20))
                last_error = "Rate limited by API; retried with exponential backoff."
                continue

            if 500 <= response.status_code < 600:
                time.sleep(min(2**attempt, 20))
                last_error = f"Server error {response.status_code}; retried."
                continue

            response.raise_for_status()
            return response.json(), None

        except requests.Timeout:
            last_error = "Request timed out; retried."
            time.sleep(min(2**attempt, 20))
        except requests.RequestException as exc:
            return None, f"Request failed: {exc}"
        except ValueError:
            return None, "API returned invalid JSON."

    return None, last_error or "API request failed after retries."


def first_list_value(value: Any) -> str:
    """Crossref often returns title-like fields as single-item lists."""
    if isinstance(value, list) and value:
        return str(value[0])
    if value:
        return str(value)
    return ""


def crossref_author_name(author: Dict[str, Any]) -> str:
    """Render a Crossref author object as a human-readable name."""
    given = author.get("given", "")
    family = author.get("family", "")
    literal = author.get("name", "")
    return clean_whitespace(f"{given} {family}") or literal


def crossref_year(item: Dict[str, Any]) -> str:
    """Extract the print publication year from the selected Crossref metadata."""
    date_parts = item.get("published-print", {}).get("date-parts", [])
    if date_parts and date_parts[0]:
        return str(date_parts[0][0])
    return ""


def query_crossref(citation: str, mailto: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Search Crossref with a bibliographic query and polite-pool mailto."""
    params = {
        "query.bibliographic": citation,
        "rows": 5,
        "mailto": mailto or DEFAULT_MAILTO,
        "select": "DOI,title,author,container-title,published-print",
    }
    payload, error = request_with_backoff(CROSSREF_WORKS_URL, params=params)
    if error:
        return [], f"Crossref: {error}"

    items = payload.get("message", {}).get("items", []) if payload else []
    candidates = []
    for item in items:
        authors = [crossref_author_name(author) for author in item.get("author", [])]
        candidates.append(
            {
                "source": "Crossref",
                "title": first_list_value(item.get("title")),
                "authors": ", ".join([author for author in authors if author]),
                "year": crossref_year(item),
                "doi": item.get("DOI", ""),
                "venue": first_list_value(item.get("container-title")),
            }
        )
    return candidates, None


def query_semantic_scholar(
    parsed: Dict[str, Any],
    api_key: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Search Semantic Scholar by extracted title plus author context."""
    title = parsed.get("Title") or ""
    author_text = " ".join(parsed.get("Authors") or [])
    query = clean_whitespace(f"{title} {author_text}") or title
    if not query:
        return [], "Semantic Scholar: no title or author text was available for search."

    params = {
        "query": query,
        "limit": 5,
        "fields": "paperId,title,authors,year,externalIds,venue,publicationDate",
    }
    headers = {"x-api-key": api_key} if api_key else {}
    payload, error = request_with_backoff(S2_SEARCH_URL, params=params, headers=headers)
    if error:
        return [], f"Semantic Scholar: {error}"

    papers = payload.get("data", []) if payload else []
    candidates = []
    for paper in papers:
        authors = [author.get("name", "") for author in paper.get("authors", [])]
        external_ids = paper.get("externalIds") or {}
        candidates.append(
            {
                "source": "Semantic Scholar",
                "title": paper.get("title", ""),
                "authors": ", ".join([author for author in authors if author]),
                "year": str(paper.get("year") or ""),
                "doi": external_ids.get("DOI", ""),
                "venue": paper.get("venue", ""),
            }
        )
    return candidates, None


def score_candidate(parsed: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Compute field-level similarity scores and the final asymmetric penalty."""
    user_title = parsed.get("Title") or ""
    user_authors = ", ".join(parsed.get("Authors") or []) or parsed.get("raw_author_text", "")
    user_year = parsed.get("Year") or ""

    title_score = similarity_percent(user_title, candidate.get("title"))
    author_score = author_similarity_percent(user_authors, candidate.get("authors", ""))
    year_score = 100.0 if user_year and user_year == str(candidate.get("year", "")) else 0.0

    base_score = (title_score + author_score + year_score) / 3
    chimeric_penalty = title_score > 80 and author_score < 90
    final_score = max(0.0, base_score - 25.0) if chimeric_penalty else base_score

    return {
        **candidate,
        "title_score": round(title_score, 1),
        "author_score": round(author_score, 1),
        "year_score": round(year_score, 1),
        "base_score": round(base_score, 1),
        "chimeric_penalty": chimeric_penalty,
        "final_score": round(final_score, 1),
    }


def classify_result(scored_candidate: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    """Map the final score and penalty state to the requested UI category."""
    if not scored_candidate:
        return "Hallucinated", "No matching metadata was found in Crossref or Semantic Scholar."

    final_score = scored_candidate["final_score"]
    if scored_candidate["chimeric_penalty"]:
        return "Chimeric / Suspicious", "The title matches strongly, but the author metadata diverges."
    if final_score >= 85:
        return "Valid", "High confidence match across title, author, and year metadata."
    if final_score >= 60:
        return "Chimeric / Suspicious", "Partial metadata match; inspect the discrepancies below."
    return "Hallucinated", "Low confidence match against retrieved official metadata."


def select_best_candidate(scored_candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Choose the candidate that should drive the final classification.

    A suspicious high-title/low-author match is intentionally elevated unless a
    clearly valid candidate exists, because that is the core chimeric signal.
    """
    if not scored_candidates:
        return None

    valid_candidates = [
        candidate
        for candidate in scored_candidates
        if not candidate["chimeric_penalty"] and candidate["final_score"] >= 85
    ]
    if valid_candidates:
        return max(valid_candidates, key=lambda item: item["final_score"])

    suspicious_candidates = [
        candidate for candidate in scored_candidates if candidate["chimeric_penalty"]
    ]
    if suspicious_candidates:
        return max(
            suspicious_candidates,
            key=lambda item: (item["title_score"], item["final_score"]),
        )

    return max(scored_candidates, key=lambda item: item["final_score"])


def render_metadata(label: str, metadata: Dict[str, Any]) -> None:
    """Render comparable metadata fields in a compact panel."""
    st.subheader(label)
    st.write(f"**Title:** {metadata.get('Title') or metadata.get('title') or 'Not found'}")
    authors = metadata.get("Authors") or metadata.get("authors") or []
    if isinstance(authors, list):
        authors = ", ".join(authors)
    st.write(f"**Authors:** {authors or 'Not found'}")
    st.write(f"**Year:** {metadata.get('Year') or metadata.get('year') or 'Not found'}")
    st.write(f"**DOI:** {metadata.get('DOI') or metadata.get('doi') or 'Not found'}")
    venue = metadata.get("venue")
    if venue:
        st.write(f"**Venue:** {venue}")


st.set_page_config(
    page_title="Citation Hallucination Detector",
    layout="wide",
)

st.title("Citation Hallucination Detector")
st.caption("Deterministic existence-only metadata verification for generated academic citations.")

with st.sidebar:
    st.header("API Settings")
    mailto = st.text_input(
        "Crossref polite-pool email",
        value=DEFAULT_MAILTO,
        help="Crossref recommends a mailto query parameter for more reliable polite-pool access.",
    )
    s2_api_key = st.text_input(
        "Semantic Scholar API key",
        value="",
        type="password",
        help="Optional. The public endpoint works without a key but may be more rate limited.",
    )

citation_input = st.text_area(
    "Paste one academic citation",
    height=180,
    placeholder=(
        "Example: Smith, J., & Doe, A. (2021). A deterministic approach to citation verification. "
        "Journal of AI Systems, 12(3), 45-60. https://doi.org/10.1234/example"
    ),
)

verify_clicked = st.button("Verify citation", type="primary")

if verify_clicked:
    if not clean_whitespace(citation_input):
        st.warning("Paste a citation before running verification.")
        st.stop()

    parsed_metadata = parse_citation(citation_input)

    with st.expander("Parsed citation metadata", expanded=True):
        st.json(
            {
                "Title": parsed_metadata.get("Title"),
                "Authors": parsed_metadata.get("Authors"),
                "Year": parsed_metadata.get("Year"),
                "DOI": parsed_metadata.get("DOI"),
            }
        )

    if not parsed_metadata.get("Title"):
        st.warning("The parser could not identify a reliable title. API search will still use the raw citation.")

    with st.spinner("Querying Crossref and Semantic Scholar metadata..."):
        crossref_candidates, crossref_error = query_crossref(citation_input, mailto)
        s2_candidates, s2_error = query_semantic_scholar(parsed_metadata, s2_api_key or None)

    api_errors = [error for error in [crossref_error, s2_error] if error]
    if api_errors:
        with st.expander("API notices", expanded=False):
            for error in api_errors:
                st.info(error)

    candidates = crossref_candidates + s2_candidates
    scored_candidates = [score_candidate(parsed_metadata, candidate) for candidate in candidates]
    best_candidate = select_best_candidate(scored_candidates)
    category, explanation = classify_result(best_candidate)

    if category == "Valid":
        st.success(f"Valid. {explanation}")
    elif category == "Chimeric / Suspicious":
        st.warning(f"Chimeric / Suspicious. {explanation}")
    else:
        st.error(f"Hallucinated. {explanation}")

    if best_candidate:
        col_user, col_official = st.columns(2)
        with col_user:
            render_metadata("User Input", parsed_metadata)
        with col_official:
            render_metadata(f"Best Retrieved Match ({best_candidate['source']})", best_candidate)

        st.subheader("Similarity Scores")
        st.table(
            [
                {
                    "Title": best_candidate["title_score"],
                    "Authors": best_candidate["author_score"],
                    "Year": best_candidate["year_score"],
                    "Base Confidence": best_candidate["base_score"],
                    "Chimeric Penalty": "Yes" if best_candidate["chimeric_penalty"] else "No",
                    "Final Confidence": best_candidate["final_score"],
                }
            ]
        )

        with st.expander("All retrieved candidates", expanded=False):
            st.table(
                [
                    {
                        "Source": candidate["source"],
                        "Title": candidate["title"],
                        "Authors": candidate["authors"],
                        "Year": candidate["year"],
                        "DOI": candidate["doi"],
                        "Final Confidence": candidate["final_score"],
                    }
                    for candidate in sorted(
                        scored_candidates,
                        key=lambda item: item["final_score"],
                        reverse=True,
                    )
                ]
            )
