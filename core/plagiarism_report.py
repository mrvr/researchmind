"""
core/plagiarism_report.py
Applies the plagiarism threshold and builds the structured report
returned by POST /api/check-plagiarism.
"""

from config import PLAGIARISM_SIMILARITY_THRESHOLD
from utils.logger import get_logger

log = get_logger("PlagiarismReport")


def _risk_level(highest_score: float) -> str:
    if highest_score >= 0.80:
        return "high"
    if highest_score >= PLAGIARISM_SIMILARITY_THRESHOLD:
        return "moderate"
    if highest_score >= 0.40:
        return "low"
    return "none"


def _verdict_label(highest_score: float) -> str:
    if highest_score >= 0.80:
        return "LIKELY COPY — HIGH SIMILARITY"
    if highest_score >= PLAGIARISM_SIMILARITY_THRESHOLD:
        return "REVIEW SUGGESTED — POSSIBLE PLAGIARISM"
    if highest_score >= 0.40:
        return "LOW OVERLAP — LIKELY ORIGINAL"
    return "ORIGINAL"


def build_report(paper_title: str, scored_candidates: list[dict], sources_queried: list[str]) -> dict:
    """
    Args:
        paper_title:        Title of the uploaded paper
        scored_candidates:  [{"candidate": CandidatePaper, "score": score_dict}, ...]
        sources_queried:    Connector names actually queried

    Returns the structured plagiarism report dict — the "top of the
    output box" the WebUI renders (score, verdict, matching titles) is
    built entirely from this.
    """
    all_results = []
    for item in scored_candidates:
        c = item["candidate"]
        s = item["score"]
        all_results.append({
            "title":            c.title,
            "authors":          c.authors,
            "source":           c.source,
            "url":              c.url,
            "doi":              c.doi,
            "year":             c.year,
            "contentLevel":     c.content_level,
            "similarityScore":  s["blendedScore"],
            "semanticScore":    s["semanticScore"],
            "lexicalScore":     s["lexicalScore"],
            "matchedExcerpts":  s["matchedExcerpts"],
        })

    all_results.sort(key=lambda r: r["similarityScore"], reverse=True)
    flagged = [r for r in all_results if r["similarityScore"] >= PLAGIARISM_SIMILARITY_THRESHOLD]
    highest_score = all_results[0]["similarityScore"] if all_results else 0.0

    if flagged:
        log.warning(
            f"'{paper_title}' — {len(flagged)} source(s) at/above "
            f"{PLAGIARISM_SIMILARITY_THRESHOLD:.0%} similarity threshold"
        )

    return {
        "status":                  "checked",
        "paperTitle":              paper_title,
        "sourcesQueried":          sources_queried,
        "totalCandidatesSearched": len(all_results),
        "highestScore":            highest_score,
        "riskLevel":               _risk_level(highest_score),
        "verdict":                 _verdict_label(highest_score),
        "threshold":               PLAGIARISM_SIMILARITY_THRESHOLD,
        "flagged":                 flagged,
        "allCandidates":           all_results,
    }
