
import pandas as pd

from scamless.data.download import fetch


class FlakySession:
    """Fails twice per URL, then succeeds; records written bytes."""

    def __init__(self, fail_times=2):
        self.fail_times = fail_times
        self.calls = {}

    def get(self, url, timeout=None):
        self.calls[url] = self.calls.get(url, 0) + 1
        if self.calls[url] <= self.fail_times:
            raise ConnectionError(f"flaky {self.calls[url]}")
        resp = type("Resp", (), {})()
        resp.content = b"ok-bytes"
        resp.raise_for_status = lambda: None
        return resp


def test_fetch_retries_and_writes_atomically(tmp_path):
    dest = tmp_path / "sub" / "file.txt"
    session = FlakySession(fail_times=2)
    fetch("http://example.test/x", dest, session=session, attempts=3, backoff_seconds=0)
    assert dest.read_bytes() == b"ok-bytes"
    assert session.calls["http://example.test/x"] == 3
    # atomic: no .part leftovers
    assert not list(tmp_path.rglob("*.part"))


def test_fetch_gives_up_after_attempts(tmp_path):
    dest = tmp_path / "never.txt"
    session = FlakySession(fail_times=99)
    try:
        fetch("http://example.test/y", dest, session=session, attempts=2, backoff_seconds=0)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    assert not dest.exists()
    assert not list(tmp_path.rglob("*.part"))


def test_fetch_skips_existing(tmp_path):
    dest = tmp_path / "already.txt"
    dest.write_bytes(b"keep-me")
    session = FlakySession(fail_times=99)
    fetch("http://example.test/z", dest, session=session)
    assert dest.read_bytes() == b"keep-me"
    assert session.calls == {}


def test_build_survives_corrupt_files(tmp_path, monkeypatch):
    # one valid SMS file + one corrupt spamassassin file must still produce records
    from scamless.data import build

    raw = tmp_path / "raw"
    (raw / "sms").mkdir(parents=True)
    (raw / "sms" / "SMSSpamCollection").write_text(
        "spam\tWin a free prize now please claim it quickly today\n"
        "ham\tPlease review the attached invoice for the march project\n",
        encoding="utf-8",
    )
    sa = raw / "spamassassin" / "spam_2"
    sa.mkdir(parents=True)
    (sa / "good.txt").write_bytes(
        b"Subject: hi\r\n\r\nThis is a long enough body for the parser to accept it.\r\n"
    )
    (sa / "bad.bin").write_bytes(b"\x00\xff\xfe garbage that breaks the email parser \x00")
    monkeypatch.setattr(build, "RAW", raw)

    records = build.collect_messages()
    texts = [r["text"] for r in records]
    assert any("invoice" in t for t in texts)  # sms ham parsed
    assert any("prize" in t for t in texts)  # sms spam parsed
    # corrupt file did not crash the build, whatever it yielded


def test_build_split_disjoint_and_replay(tmp_path, monkeypatch):
    from scamless.data import build

    msgs = [
        {
            "text": f"message number {i:03d} about the quarterly planning things",
            "labels": [],
            "source": "a",
            "language": "en",
        }
        for i in range(50)
    ]
    df = pd.DataFrame(build.dedupe(msgs))
    train, val, test = build.split_records(df, seed=1)
    assert not set(train["text"]) & set(test["text"])
    assert not set(val["text"]) & set(test["text"])

    # replay: old parquet rows merge into train only
    old = pd.DataFrame(
        [
            {
                "text": f"old replay row {i} about something else entirely",
                "labels": ["generic_spam"],
                "source": "old",
                "language": "en",
            }
            for i in range(3)
        ]
    )
    old_path = tmp_path / "old_train.parquet"
    old.to_parquet(old_path)
    merged = build.merge_replay(train, [str(old_path)])
    assert len(merged) == len(train) + 3
    assert merged["text"].str.startswith("old replay row").sum() == 3
