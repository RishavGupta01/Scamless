from scamless.data.urls import parse_majestic_csv, parse_openphish_lines


def test_parse_openphish_lines():
    lines = [
        "http://secure-login.example-verify.com/account",
        "https://update-billing.example.ru/pay",
        "",
        "# comment style lines are skipped",
    ]
    records = parse_openphish_lines(lines)
    assert len(records) == 2
    assert all(r["malicious"] for r in records)
    assert records[0]["source"] == "openphish"


def test_parse_majestic_csv_top_n():
    csv_text = (
        "GlobalRank,TldRank,Domain,TLD,RefSubNets,RefIPs,IDN_Domain,IDN_Encoding\n"
        "1,1,google.com,com,1000,2000,google.com,google.com\n"
        "2,1,facebook.com,com,900,1800,facebook.com,facebook.com\n"
        "3,1,badsite.example,example,1,2,badsite.example,badsite.example\n"
    )
    records = parse_majestic_csv(csv_text, top_n=2)
    assert [r["url"] for r in records] == ["http://google.com", "http://facebook.com"]
    assert all(not r["malicious"] for r in records)
    assert records[0]["source"] == "majestic_million"
