"""
Utility functions for FastAPI service.
Isolated ranking logic for easy modification.
"""

from typing import List, Dict
from collections import Counter


def rank_documents_by_occurrence(passage_results: List[Dict]) -> List[Dict]:
    """
    Rank documents by number of occurrences in passage results.
    Pure count-based ranking (can be changed to weighted later).

    Args:
        passage_results: List of passage results with doc_id

    Returns:
        List of documents sorted by occurrence count (descending)
    """
    # Count occurrences of each doc_id
    doc_counter = Counter()
    doc_info = {}  # Store first occurrence of each doc for metadata

    for passage in passage_results:
        doc_id = passage['doc_id']
        doc_counter[doc_id] += 1

        # Store doc info from first occurrence
        if doc_id not in doc_info:
            doc_info[doc_id] = {
                'doc_text': passage.get('doc_text', ''),
                'doc_file': passage.get('doc_file', ''),
                'doc_position': passage.get('doc_position', 0)
            }

    # Create ranked document list
    ranked_docs = []
    for rank, (doc_id, count) in enumerate(doc_counter.most_common(), 1):
        ranked_docs.append({
            'rank': rank,
            'doc_id': doc_id,
            'occurrences': count,
            'doc_text': doc_info[doc_id]['doc_text'],
            'doc_file': doc_info[doc_id]['doc_file'],
            'doc_position': doc_info[doc_id]['doc_position']
        })

    return ranked_docs


def rank_documents_by_score_sum(passage_results: List[Dict]) -> List[Dict]:
    """
    Alternative ranking: Rank documents by sum of passage scores.
    Not used currently, but available for future use.

    Args:
        passage_results: List of passage results with doc_id and score

    Returns:
        List of documents sorted by score sum (descending)
    """
    doc_scores = {}
    doc_info = {}
    doc_counts = Counter()

    for passage in passage_results:
        doc_id = passage['doc_id']
        score = passage.get('score', 0.0)

        if doc_id not in doc_scores:
            doc_scores[doc_id] = 0.0
            doc_info[doc_id] = {
                'doc_text': passage.get('doc_text', ''),
                'doc_file': passage.get('doc_file', ''),
                'doc_position': passage.get('doc_position', 0)
            }

        doc_scores[doc_id] += score
        doc_counts[doc_id] += 1

    # Sort by total score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    ranked_docs = []
    for rank, (doc_id, total_score) in enumerate(sorted_docs, 1):
        ranked_docs.append({
            'rank': rank,
            'doc_id': doc_id,
            'occurrences': doc_counts[doc_id],
            'total_score': total_score,
            'doc_text': doc_info[doc_id]['doc_text'],
            'doc_file': doc_info[doc_id]['doc_file'],
            'doc_position': doc_info[doc_id]['doc_position']
        })

    return ranked_docs


def rank_documents_by_best_passage(passage_results: List[Dict]) -> List[Dict]:
    """
    Alternative ranking: Rank documents by their best (highest-ranked) passage.
    Not used currently, but available for future use.

    Args:
        passage_results: List of passage results with doc_id and rank

    Returns:
        List of documents sorted by best passage rank (ascending)
    """
    doc_best_rank = {}
    doc_info = {}
    doc_counts = Counter()

    for passage in passage_results:
        doc_id = passage['doc_id']
        rank = passage.get('rank', float('inf'))

        if doc_id not in doc_best_rank:
            doc_best_rank[doc_id] = rank
            doc_info[doc_id] = {
                'doc_text': passage.get('doc_text', ''),
                'doc_file': passage.get('doc_file', ''),
                'doc_position': passage.get('doc_position', 0)
            }
        else:
            doc_best_rank[doc_id] = min(doc_best_rank[doc_id], rank)

        doc_counts[doc_id] += 1

    # Sort by best rank (lower is better)
    sorted_docs = sorted(doc_best_rank.items(), key=lambda x: x[1])

    ranked_docs = []
    for rank, (doc_id, best_rank) in enumerate(sorted_docs, 1):
        ranked_docs.append({
            'rank': rank,
            'doc_id': doc_id,
            'occurrences': doc_counts[doc_id],
            'best_passage_rank': best_rank,
            'doc_text': doc_info[doc_id]['doc_text'],
            'doc_file': doc_info[doc_id]['doc_file'],
            'doc_position': doc_info[doc_id]['doc_position']
        })

    return ranked_docs
