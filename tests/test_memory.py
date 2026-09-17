from replycoach.memory import m0_no_memory, m1_last3, m2_dynamic

SESSION = [
    "are you coming to the lecture",
    "already inside",
    "need to print later",
    "library packed today",
    "I can go now",
    "lecture ending soon sia",
    "yeah I'm starving",
    "shall we walk together",
]


def test_m0_empty():
    assert m0_no_memory(SESSION, "go makan") == []


def test_m1_last_three():
    assert m1_last3(SESSION, "go makan") == SESSION[-3:]


def test_m2_keeps_newest_and_relevant_older():
    selected = m2_dynamic(SESSION, "library seats or makan")
    assert SESSION[-2] in selected
    assert SESSION[-1] in selected
    assert len(selected) <= 4
    # library-related older turn should be preferred over lecture small-talk
    assert any("library" in turn or "print" in turn for turn in selected)
