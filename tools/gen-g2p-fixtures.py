#!/usr/bin/env python3
"""Generate the golden G2P fixtures from Python Misaki.

This is the one place Python is allowed in this repo, and it earns it: Misaki
*is* the specification. The Rust G2P in `src/g2p/` is a port of it, and the only
way to know a port is faithful is to compare against the thing it was ported
from, character for character.

Misaki ships no tests, so there is nothing to port — these fixtures are the
substitute. They are committed, and the Rust tests read them directly; Python is
needed only to regenerate them after a deliberate Misaki version bump.

    python3 -m venv .venv
    .venv/bin/pip install 'misaki[ja,zh]==0.7.4' phonemizer kokoro
    brew install espeak-ng          # 1.52.0, must match the vendored espeak-ng
    .venv/bin/python tools/gen-g2p-fixtures.py

Writes fixtures/g2p/<lang>.json. Each entry is {"text", "phonemes"} where
`phonemes` is exactly what Misaki hands to Kokoro.
"""

import json
import pathlib
import sys

# Kokoro's own language table (kokoro/pipeline.py LANG_CODES). Copied rather
# than imported so regenerating fixtures does not require the kokoro package,
# and asserted against it below when it is importable.
ESPEAK_LANGUAGES = {
    "es": "es",
    "fr": "fr-fr",
    "hi": "hi",
    "it": "it",
    "pt": "pt-br",
}

OUT = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "g2p"

# The corpus is chosen to exercise the parts of Misaki that a port gets wrong:
# punctuation preservation and its position codes (begin/end/interior/alone),
# decimal separators that must NOT be split, quotes and parentheses (which
# EspeakG2P swaps through angle brackets), digits (each language expands them
# differently), and language switches (which "remove-flags" must strip).
CORPUS = {
    "es": [
        "Hola.",
        "Hola, ¿cómo estás?",
        "Tengo 19,99 euros en el bolsillo.",
        "El proyecto (una idea vieja) por fin funciona.",
        "«Buenos días», dijo ella.",
        "Son las 3.5 veces más rápido.",
        "¡Increíble! No me lo puedo creer…",
        "Madrid, Barcelona; Sevilla: tres ciudades.",
        "El software de Microsoft no es barato.",
        "1, 2, 3 — vamos.",
    ],
    "fr": [
        "Bonjour.",
        "J'aime le football et les croissants.",
        "Il coûte 19,99 euros.",
        "Les enfants (les miens) sont partis.",
        "« Bonjour », dit-elle.",
        "Qu'est-ce que c'est ?",
        "Paris, Lyon ; Marseille : trois villes.",
        "C'est un weekend de rêve…",
        "Nous avons 3,5 fois plus.",
        "Le manager a un meeting.",
    ],
    "it": [
        "Ciao.",
        "Ciao, come stai oggi?",
        "Costa 19,99 euro.",
        "Il progetto (una vecchia idea) funziona.",
        "«Buongiorno», disse lei.",
        "Che cosa vuoi fare?",
        "Roma, Milano; Napoli: tre città.",
        "Incredibile! Non ci posso credere…",
        "Siamo 3,5 volte più veloci.",
        "Il computer non funziona.",
    ],
    "pt": [
        "Olá.",
        "Olá, tudo bem com você?",
        "Custa 19,99 reais.",
        "O projeto (uma ideia antiga) funciona.",
        "«Bom dia», disse ela.",
        "O que você quer fazer?",
        "Rio, São Paulo; Recife: três cidades.",
        "Incrível! Não acredito…",
        "Somos 3,5 vezes mais rápidos.",
        "O computador não funciona.",
    ],
    "hi": [
        "नमस्ते।",
        "नमस्ते, आप कैसे हैं?",
        "इसकी कीमत 19.99 रुपये है।",
        "यह परियोजना (एक पुराना विचार) काम करती है।",
        "आप क्या करना चाहते हैं?",
        "दिल्ली, मुंबई; चेन्नई: तीन शहर।",
        "अविश्वसनीय! मुझे यकीन नहीं होता…",
        "हम 3.5 गुना तेज हैं।",
        "कंप्यूटर काम नहीं कर रहा है।",
        "मेरा नाम राहुल है।",
    ],
    "zh": [
        # The first four exercise the digit-leading bug documented in gen_zh().
        "2026年的测试。",
        "2026年",
        "2026",
        "3.5倍的增长。",
        "你好。",
        "你好，世界！",
        "我们在银行工作。",
        "他长大了，头发很长。",
        "2026年的测试。",
        "这个项目（一个老想法）终于成功了。",
        "北京、上海和广州都是大城市。",
        "他说：“今天天气很好。”",
        "我买了3.5公斤苹果。",
        "中文和English混在一起。",
        "一二三四五六七八九十。",
        "重要的事情说三遍。",
    ],
    "ja": [
        "こんにちは。",
        "こんにちは、世界。",
        "東京都に住んでいます。",
        "2026年のテストです。",
        "このプロジェクト（古いアイデア）はうまくいった。",
        "「おはよう」と彼女は言った。",
        "私は3.5キロのりんごを買いました。",
        "カタカナ、ひらがな、漢字。",
        "彼はまっすぐ歩いて行った。",
        "そうですね……難しいです。",
        "コンピューターが動きません。",
        "一二三四五六七八九十。",
    ],
}

# American and British English go through the already-ported `en` path. They are
# included so a change to the shared tokenizer/retokenizer shows up here too.
EN_CORPUS = [
    "Hello world.",
    "Revenue grew 3.5x to $4.2M in Q3.",
    "It's a state-of-the-art system.",
    "We raised $4.2 million last year.",
    "Growth was 15% year over year.",
    "The camelCase identifier was renamed.",
    "Dr. Smith arrived at 5:30 p.m.",
    "“Stop,” she said — and he did…",
]


def gen_espeak(short: str, language: str):
    from misaki.espeak import EspeakG2P

    g2p = EspeakG2P(language=language)
    return [{"text": t, "phonemes": g2p(t)} for t in CORPUS[short]]


def gen_zh():
    """Chinese, with the one deliberate divergence from stock Misaki.

    `misaki.zh.ZHG2P.__call__` decides whether the first segment is Chinese by
    testing `text[0]` — the ORIGINAL text — but then iterates over segments of
    `zhstr`, the text after `cn2an` has rewritten Arabic numerals into Chinese
    numerals. When that rewrite changes the class of the leading character (any
    string starting with a digit), the is-Chinese flag is inverted for the whole
    string: Chinese runs are emitted as raw hanzi and punctuation is fed to the
    pinyin converter.

    The damage is not cosmetic. "2026" comes back as the literal characters
    '二千零二十六', which are not in Kokoro's vocabulary, and "2026年的测试。"
    raises ValueError outright. Text beginning with a year or a figure is
    ordinary narration, so this is a defect we cannot ship around.

    The fix is to read the flag off `zhstr`, the string actually being iterated.
    Measured over the corpus below plus digit-leading variants: 15 of 18 cases
    are byte-identical to stock Misaki and the 3 that change are exactly the
    broken ones. Entries carry a `misaki` field recording what stock Misaki
    returns, so the divergence stays visible and auditable.
    """
    import re

    import cn2an
    import jieba
    from misaki.zh import ZHG2P

    stock = ZHG2P()

    def corrected(text, zh="一-鿿"):
        if not text:
            return ""
        zhstr = cn2an.transform(text, "an2cn")
        is_zh = re.match(f"[{zh}]", zhstr[0])  # zhstr, not text
        result = ""
        for segment in re.findall(f"[{zh}]+|[^{zh}]+", zhstr):
            if is_zh:
                words = jieba.lcut(segment, cut_all=False)
                segment = " ".join(ZHG2P.word2ipa(w) for w in words)
            result += segment
            is_zh = not is_zh
        return result.replace(chr(815), "")

    out = []
    for t in CORPUS["zh"]:
        try:
            raw = stock(t)
        except Exception as exc:  # noqa: BLE001 - recording the failure IS the point
            raw = f"<raises {type(exc).__name__}>"
        entry = {"text": t, "phonemes": corrected(t)}
        if raw != entry["phonemes"]:
            entry["misaki"] = raw
        out.append(entry)
    return out


def gen_ja():
    from misaki.ja import JAG2P

    g2p = JAG2P()
    return [{"text": t, "phonemes": g2p(t)} for t in CORPUS["ja"]]


def gen_en(british: bool):
    from misaki.en import G2P

    g2p = G2P(trf=False, british=british, fallback=None, unk="")
    out = []
    for t in EN_CORPUS:
        phonemes, _ = g2p(t)
        out.append({"text": t, "phonemes": phonemes})
    return out


def check_language_table():
    """Fail loudly if Kokoro's language table ever stops matching ours."""
    try:
        from kokoro.pipeline import LANG_CODES
    except ImportError:
        print("  (kokoro not installed; skipping LANG_CODES cross-check)")
        return
    expected = {"e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br"}
    for code, language in expected.items():
        if LANG_CODES[code] != language:
            raise SystemExit(
                f"Kokoro's LANG_CODES[{code!r}] is {LANG_CODES[code]!r}, "
                f"not {language!r}. The Rust language table must be updated too."
            )
    print("  LANG_CODES matches Kokoro")


def write(name, entries):
    path = OUT / f"{name}.json"
    path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    diverged = sum(1 for e in entries if "misaki" in e)
    note = f", {diverged} diverging from stock Misaki" if diverged else ""
    print(f"  {path.name}: {len(entries)} cases{note}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    check_language_table()

    # Written as each language finishes: the English path needs misaki[en]
    # (spacy, num2words), which is a much heavier install than the rest, and a
    # missing optional dependency must not throw away the other eight languages.
    for short, language in ESPEAK_LANGUAGES.items():
        write(short, gen_espeak(short, language))
    write("zh", gen_zh())
    write("ja", gen_ja())
    try:
        write("en_us", gen_en(british=False))
        write("en_gb", gen_en(british=True))
    except ImportError as exc:
        print(f"\n  SKIPPED en_us/en_gb: {exc}")
        print("  Install them with: pip install 'misaki[en]'")

    print(f"\nfixtures in {OUT}")


if __name__ == "__main__":
    sys.exit(main())
