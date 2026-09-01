from knowledge.pubmed_search import (
    PubMedRows,
    build_pubmed_query_ladder,
    parse_pubmed_articles,
    rank_pubmed_rows,
)
from knowledge.research_search import parse_arxiv_entries, parse_stackexchange_items


def test_pubmed_parser_retains_citation_and_abstract_sections():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
      <PMID>123</PMID><Article><ArticleTitle>A <i>useful</i> trial</ArticleTitle>
      <Abstract><AbstractText Label="BACKGROUND">Why</AbstractText>
      <AbstractText Label="RESULTS">What happened</AbstractText></Abstract>
      <AuthorList><Author><LastName>Smith</LastName><Initials>AB</Initials></Author></AuthorList>
      <Journal><Title>Journal</Title><JournalIssue><PubDate><Year>2025</Year><Month>Jan</Month></PubDate></JournalIssue></Journal>
      </Article></MedlineCitation><PubmedData><ArticleIdList>
      <ArticleId IdType="doi">10.1/example</ArticleId></ArticleIdList></PubmedData>
      </PubmedArticle></PubmedArticleSet>"""
    rows = parse_pubmed_articles(xml)
    assert rows[0]["source_id"] == "pmid:123"
    assert rows[0]["title"] == "A useful trial"
    assert "BACKGROUND: Why" in rows[0]["abstract"]
    assert rows[0]["doi"] == "10.1/example"
    assert rows[0]["published_date"] == "2025-Jan"


def test_pubmed_query_ladder_is_structured_and_broadens_without_prose():
    queries = build_pubmed_query_ladder(
        "cariprazine irritability autism spectrum clinical study",
        supporting_facets=["behavioral endpoint"],
    )
    assert queries
    assert "therapist" not in " ".join(queries)
    assert " AND " in queries[0]
    assert any("autism" in query and "irritability" in query for query in queries)
    assert len(queries) <= 8


def test_pubmed_ranker_requires_two_concept_hits_and_marks_relevance():
    rows = rank_pubmed_rows([
        {
            "pmid": "37437109",
            "title": "Cariprazine in autism spectrum disorder",
            "abstract": "Irritability was assessed as a behavioral endpoint.",
        },
        {
            "pmid": "noise",
            "title": "Cariprazine pharmacology",
            "abstract": "A receptor study unrelated to the population.",
        },
    ], "cariprazine irritability autism", max_results=10)
    assert isinstance(rows, PubMedRows)
    assert [row["pmid"] for row in rows] == ["37437109"]
    assert rows.status == "succeeded"
    assert rows[0]["relevance_hits"] >= 2


def test_pubmed_ranker_distinguishes_empty_from_no_relevant():
    assert rank_pubmed_rows([], "a named intervention").status == "no_results"
    assert rank_pubmed_rows([
        {"pmid": "1", "title": "Unrelated", "abstract": "Nothing here"},
    ], "named intervention").status == "no_relevant_results"


def test_arxiv_parser_retains_stable_url_authors_and_dates():
    xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>https://arxiv.org/abs/1234.5678</id><updated>2026-01-02T00:00:00Z</updated>
      <published>2026-01-01T00:00:00Z</published><title> A paper </title>
      <summary> The abstract. </summary><author><name>A. Author</name></author>
      </entry></feed>"""
    rows = parse_arxiv_entries(xml)
    assert rows[0]["source_id"] == "https://arxiv.org/abs/1234.5678"
    assert rows[0]["authors"] == ["A. Author"]
    assert rows[0]["published_date"].startswith("2026-01-01")


def test_stackexchange_parser_strips_html_but_keeps_question_identity():
    rows = parse_stackexchange_items({"items": [{
        "question_id": 42,
        "title": "A &amp; B",
        "body": "<p>Use <code>x</code>.</p>",
        "link": "https://stackoverflow.com/q/42",
        "creation_date": 123,
        "score": 9,
        "is_answered": True,
    }]})
    assert rows[0]["source_id"] == "stackexchange:42"
    assert rows[0]["title"] == "A & B"
    assert rows[0]["text"] == "Use x ."
    assert rows[0]["date"] == "123"
