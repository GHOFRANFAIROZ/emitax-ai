"""RAG katmani icin cevrimdisi testler (gercek resmi mevzuat korpusu)."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "legal_corpus"

from emitax.rag.chunker import load_corpus                      # noqa: E402
from emitax.rag.index import build_index                        # noqa: E402
from emitax.rag.retrieve import retrieve                        # noqa: E402
from emitax.rag.recommend import recommend, build_query         # noqa: E402


@pytest.fixture(scope="module")
def index():
    return build_index(load_corpus(CORPUS))


def test_corpus_loads():
    chunks = load_corpus(CORPUS)
    assert len(chunks) > 1000
    c = chunks[0]
    assert c.doc_id and c.title and c.page >= 1 and c.group


def test_index_builds(index):
    assert index.matrix.shape[0] == len(index)
    assert len(index) > 1000


def test_measurement_returns_regulation(index):
    rec = recommend(index, {"fired_checks": "measurement", "reason": "SO2 beyani olcumun altinda"}, k=3)
    assert rec.retrieved
    # boost sayesinde en iyi sonuc Turk mevzuatindan (SKHKKY/SEOS) gelmeli
    assert rec.retrieved[0].chunk.group == "mevzuat"
    assert "sayfa" in rec.text and "Kaynak:" in rec.text


def test_physics_returns_ipcc(index):
    rec = recommend(index, {"fired_checks": "physics", "reason": "CO2 tutarsiz"}, k=3)
    assert rec.retrieved and rec.retrieved[0].chunk.group == "ipcc"


def test_sector_filter(index):
    # sektorel bir dosyadan parca; sektor verilmezse gelmemeli
    q = "kursun uretimi lead production emisyon"
    without = retrieve(index, q, k=5)
    assert all(c.chunk.sector_independent for c in without)
    withsec = retrieve(index, q, k=5, sectors=["kursun"])
    assert any(not c.chunk.sector_independent for c in withsec)


def test_build_query_maps_checks():
    q = build_query({"fired_checks": "physics", "reason": "x"})
    assert "IPCC" in q
