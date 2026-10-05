from paper_summarizer.structure import Paragraph, Section, _kind_of, _section_id, fix_text
from paper_summarizer.summarize import cited_ids, reading_units


def test_fix_text_private_use_small_caps():
    assert fix_text("S\uf775\uf76d\uf76d\uf761\uf772\uf779") == "Summary"
    assert fix_text("a b s t r a c t") == "abstract"
    assert fix_text("a normal sentence here") == "a normal sentence here"


def test_section_ids_and_kinds():
    assert _section_id("REFERENCE S", None, 9) == "Ref"
    assert _kind_of("REFERENCE S", None, True) == "references"
    assert _section_id("Algorithm 2 SMC algorithm.", None, 3) == "Alg2"
    assert _kind_of("Summary", None, False) == "abstract"
    assert _kind_of("Journal of X", None, False) == "front"
    assert _section_id("Methods", "2.1", 4) == "2.1"


def _sec(sid, sizes, kind="body"):
    return Section(id=sid, title=sid, kind=kind,
                   paragraphs=[Paragraph(f"§{sid}-p{i + 1}", "text", "x" * n) for i, n in enumerate(sizes)])


def test_reading_units_groups_and_splits():
    secs = [_sec("Front", [100], "front"), _sec("1", [300, 300]), _sec("2", [300]),
            _sec("3", [900, 900, 900]), _sec("Ref", [50], "references")]
    units = reading_units(secs, 1000)
    covered = [p.id for u in units for _, ps, _ in u.parts for p in ps]
    assert covered == ["§1-p1", "§1-p2", "§2-p1", "§3-p1", "§3-p2", "§3-p3"]  # 前付け・参考文献は除外
    assert all(u.chars <= 1000 for u in units)
    assert units[0].parts[0][0].id == "1" and units[0].parts[1][0].id == "2"  # 小さい節はまとめる
    conts = [c for u in units for s, _, c in u.parts if s.id == "3"]
    assert conts == [False, True, True]  # 長い節は分割し、2 つ目以降は「続き」


def test_cited_ids():
    assert cited_ids("- a [§1-p1, §2.3-p4]\n- b [§1-p1][§Abs-p2]") == ["§1-p1", "§2.3-p4", "§Abs-p2"]
