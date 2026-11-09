from __future__ import annotations

from gpt_researcher.scraper.pdf_router.gates import looks_empty, markdown_tables, table_issues, word_recall

GOOD_TABLE = """Some prose before the table.

|Model|SQuAD 1.1/2.0|MNLI-m|SST-2|RACE|
|---|---|---|---|---|
|SEGMENT-PAIR|90.4/78.7|84.0|92.9|64.2|
|SENTENCE-PAIR|88.7/76.2|82.9|92.1|63.0|
|FULL-SENTENCES|90.4/79.1|84.7|92.5|64.8|

Prose after.
"""

# Verbatim shape of pymupdf4llm's output for two side-by-side tables on BERT page 7.
GARBLED_TABLE = """|System|D|ev|Te|st|System|Dev|Test|
|---|---|---|---|---|---|---|---|
||EM|F1|EM|F1|ESIM+GloVe|51.9|52.7|
|Top Leaderboard Systems|(Dec|10th,|2018)||ESIM+ELMo|59.1|59.2|
|Human|-|-|82.3|91.2|OpenAI GPT|-|78.0|
|#2 Ensemble - QANet|-|-|845|905|BERTBASE|81.6|-|
||||.|.|BERTLARGE|**86.6 **|**86.3**|
|Publishe<br>BiDAF+ELMo (Single)|d<br>-|85.6|-|85.8|Human (expert)|-|85.0|
"""

RAGGED_TABLE = """||MNLI|QNLI|QQP|RTE|SST|MRPC|CoLA|STS|WNLI|Avg|
|---|---|---|---|---|---|---|---|---|---|---|
|Single-task si<br>BERTLARGE|ngle models<br>86.6/-|on dev<br>92.3|91.3|70.4|93.2|88.0|60.6|90.0|-|-|
|XLNetLARGE|89.8/-|93.9|91.8|83.8|95.6|89.2|63.6|91.8|-|-|
|Ensembles on<br>|test (from le<br>|aderboa<br>|rd as of<br>|July 25,<br>|2019)<br>||||||
|ALICE|88.2/87.9|95.7|90.7|83.5|95.2|92.6|68.6|91.1|80.8|86.3|
"""


def test_markdown_tables_parsing_skips_separator_rows():
    tables = markdown_tables(GOOD_TABLE)
    assert len(tables) == 1
    assert tables[0][0] == ["Model", "SQuAD 1.1/2.0", "MNLI-m", "SST-2", "RACE"]
    assert len(tables[0]) == 4  # header + 3 rows


def test_good_table_has_no_issues():
    assert table_issues(GOOD_TABLE) == []


def test_garbled_side_by_side_table_is_flagged():
    issues = table_issues(GARBLED_TABLE)
    joined = " ".join(issues)
    assert "fragmented header" in joined
    assert "orphan punctuation" in joined
    assert "repeated header labels" in joined
    assert "split word" in joined


def test_ragged_table_is_flagged():
    issues = table_issues(RAGGED_TABLE)
    assert any("ragged rows" in i for i in issues)
    assert any("split across cells" in i for i in issues)


def test_short_lowercase_headers_are_not_split_text():
    table = "|Model|data|bsz|steps|SQuAD|\n|---|---|---|---|---|\n|RoBERTa|16GB|8K|100K|93.6/87.3|\n|+ more data|160GB|8K|100K|94.0/87.7|\n"
    assert table_issues(table) == []


def test_word_recall_identical_text():
    text = "Language model pretraining has led to significant performance gains."
    assert word_recall(text, text) == 1.0


def test_word_recall_joins_hyphenation_and_ligatures():
    reference = "sig-\nniﬁcant performance gains but careful com-\nparison between approaches"
    candidate = "significant performance gains but careful comparison between approaches"
    assert word_recall(reference, candidate) == 1.0


def test_word_recall_detects_dropped_paragraph():
    reference = ("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 3) + "lambda mu nu xi omicron"
    candidate = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    assert word_recall(reference, candidate) < 0.9


def test_word_recall_ignores_markdown_decoration_and_short_tokens():
    reference = "We introduce a new language representation model called BERT"
    candidate = "## We introduce a **new** language representation model called _BERT_"
    assert word_recall(reference, candidate) == 1.0


def test_word_recall_with_empty_reference_is_one():
    assert word_recall("", "anything") == 1.0


def test_looks_empty():
    assert looks_empty("")
    assert looks_empty("   \n\n  ")
    assert looks_empty("tiny")
    assert not looks_empty("x" * 25)
