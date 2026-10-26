//! Numbers to English words.
//!
//! Port of `EnglishNum2Word.swift` in `packages/kokoro-swift` (itself a port of
//! the Python `num2words` package), **including the vendored `MisakiSwift#11`
//! patch** that moves "twenty" into the tens table — without it "21" reads as
//! "one" and "2026" mangles. **Keep this in lockstep with the Swift file**:
//! a number that reads differently on Linux than on macOS is worse than a
//! missing feature.

const NEG_WORD: &str = "minus ";
const POINT_WORD: &str = "point";

/// VENDORED PATCH (MisakiSwift#11): "twenty" belongs here, not in `LOW`.
const MID: &[(i64, &str)] = &[
    (1000, "thousand"),
    (100, "hundred"),
    (90, "ninety"),
    (80, "eighty"),
    (70, "seventy"),
    (60, "sixty"),
    (50, "fifty"),
    (40, "forty"),
    (30, "thirty"),
    (20, "twenty"),
];

/// Indexed from the top: `LOW[19 - n]` is the word for `n` (0..=19).
const LOW: &[&str] = &[
    "nineteen",
    "eighteen",
    "seventeen",
    "sixteen",
    "fifteen",
    "fourteen",
    "thirteen",
    "twelve",
    "eleven",
    "ten",
    "nine",
    "eight",
    "seven",
    "six",
    "five",
    "four",
    "three",
    "two",
    "one",
    "zero",
];

const HIGH: &[&str] = &[
    "m", "b", "tr", "quadr", "quint", "sext", "sept", "oct", "non", "dec",
];

fn ordinal_word(word: &str) -> Option<&'static str> {
    Some(match word {
        "one" => "first",
        "two" => "second",
        "three" => "third",
        "four" => "fourth",
        "five" => "fifth",
        "six" => "sixth",
        "seven" => "seventh",
        "eight" => "eighth",
        "nine" => "ninth",
        "ten" => "tenth",
        "eleven" => "eleventh",
        "twelve" => "twelfth",
        _ => return None,
    })
}

/// 10^6, 10^9, … paired with "million", "billion", …
fn cards() -> Vec<(i64, String)> {
    let mut out = Vec::new();
    for (i, w) in HIGH.iter().enumerate() {
        let power = 6 + i * 3;
        if power > 18 {
            break; // beyond i64
        }
        out.push((10i64.pow(power as u32), format!("{w}illion")));
    }
    out.sort_by(|a, b| b.0.cmp(&a.0));
    out
}

pub fn to_cardinal(number: i64) -> String {
    if number < 0 {
        return format!("{NEG_WORD}{}", to_cardinal(number.abs()));
    }
    if number < 20 {
        return LOW[(19 - number) as usize].to_string();
    }
    if number < 100 {
        let tens = (number / 10) * 10;
        let ones = number % 10;
        let tens_word = MID
            .iter()
            .find(|(v, _)| *v == tens)
            .map(|(_, w)| *w)
            .unwrap_or("");
        if ones == 0 {
            return tens_word.to_string();
        }
        return format!("{tens_word}-{}", LOW[(19 - ones) as usize]);
    }
    if number < 1000 {
        let hundreds_word = format!("{} hundred", to_cardinal(number / 100));
        let remainder = number % 100;
        if remainder == 0 {
            return hundreds_word;
        }
        return format!("{hundreds_word} and {}", to_cardinal(remainder));
    }
    // VENDORED PATCH: millions and up are tried FIRST. `MID` tops out at 1000,
    // so trying it first swallowed every large number and the millions table
    // was unreachable — 1,000,000 read as "one thousand thousand".
    for (value, word) in cards() {
        if number >= value {
            let quotient = to_cardinal(number / value);
            let remainder = number % value;
            return if remainder == 0 {
                format!("{quotient} {word}")
            } else {
                format!("{quotient} {word}, {}", to_cardinal(remainder))
            };
        }
    }
    let mut mid_desc: Vec<(i64, &str)> = MID.to_vec();
    mid_desc.sort_by(|a, b| b.0.cmp(&a.0));
    for (value, word) in mid_desc {
        if number >= value {
            let quotient = to_cardinal(number / value);
            let remainder = number % value;
            return if remainder == 0 {
                format!("{quotient} {word}")
            } else {
                format!("{quotient} {word}, {}", to_cardinal(remainder))
            };
        }
    }
    String::new()
}

pub fn to_ordinal(number: i64) -> String {
    if number <= 0 {
        return String::new();
    }
    let cardinal = to_cardinal(number);
    let mut out_words: Vec<String> = cardinal.split(' ').map(str::to_string).collect();
    let last_idx = out_words.len() - 1;
    let mut last_words: Vec<String> = out_words[last_idx].split('-').map(str::to_string).collect();
    let last_last = last_words.len() - 1;
    let lower = last_words[last_last].to_lowercase();

    let replaced = match ordinal_word(&lower) {
        Some(o) => o.to_string(),
        None => {
            if let Some(stem) = lower.strip_suffix('y') {
                format!("{stem}ieth")
            } else {
                format!("{lower}th")
            }
        }
    };
    // Swift capitalizes here (`lastWord.capitalized`); the caller lowercases
    // for lexicon lookup, but keep the shape identical so any future
    // case-sensitive consumer sees the same string.
    let mut chars = replaced.chars();
    let capitalized = match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => replaced,
    };
    last_words[last_last] = capitalized;
    out_words[last_idx] = last_words.join("-");
    out_words.join(" ")
}

pub fn to_year(year: i64) -> String {
    let (val, suffix) = if year < 0 {
        (year.abs(), Some("BC"))
    } else {
        (year, None)
    };
    let high = val / 100;
    let low = val % 100;

    // 00XX, X00X, or beyond 9999 read as plain cardinals.
    let val_text = if high == 0 || (high % 10 == 0 && low < 10) || high >= 100 {
        to_cardinal(val)
    } else {
        let low_text = if low == 0 {
            "hundred".to_string()
        } else if low < 10 {
            format!("oh-{}", to_cardinal(low))
        } else {
            to_cardinal(low)
        };
        format!("{} {low_text}", to_cardinal(high))
    };
    match suffix {
        Some(s) => format!("{val_text} {s}"),
        None => val_text,
    }
}

/// "3.5" → "three point five". Fractional digits are read one at a time, which
/// is what `toDecimal` does on the Swift side.
pub fn to_decimal(text: &str) -> String {
    let cleaned = text.replace(',', "");
    let (int_part, frac_part) = match cleaned.split_once('.') {
        Some((a, b)) => (a, b),
        None => (cleaned.as_str(), ""),
    };
    let negative = int_part.starts_with('-');
    let int_digits = int_part.trim_start_matches('-');
    let int_value: i64 = if int_digits.is_empty() {
        0
    } else {
        int_digits.parse().unwrap_or(0)
    };
    let int_words = to_cardinal(if negative { -int_value } else { int_value });

    // Trailing zeros are not spoken ("2.0" → "two"), matching Decimal's
    // normalization on the Swift side.
    let frac = frac_part.trim_end_matches('0');
    if frac.is_empty() {
        return int_words;
    }
    let frac_words: Vec<String> = frac
        .chars()
        .map(|c| to_cardinal(c.to_digit(10).unwrap_or(0) as i64))
        .collect();
    format!("{int_words} {POINT_WORD} {}", frac_words.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_vendored_patch_case() {
        // Without MisakiSwift#11 these are the two that break.
        assert_eq!(to_cardinal(21), "twenty-one");
        assert_eq!(to_year(2026), "twenty twenty-six");
    }

    #[test]
    fn cardinals() {
        assert_eq!(to_cardinal(0), "zero");
        assert_eq!(to_cardinal(7), "seven");
        assert_eq!(to_cardinal(15), "fifteen");
        assert_eq!(to_cardinal(20), "twenty");
        assert_eq!(to_cardinal(99), "ninety-nine");
        assert_eq!(to_cardinal(100), "one hundred");
        assert_eq!(to_cardinal(250), "two hundred and fifty");
        assert_eq!(to_cardinal(1250), "one thousand, two hundred and fifty");
        assert_eq!(to_cardinal(-5), "minus five");
    }

    #[test]
    fn millions_and_up() {
        assert_eq!(to_cardinal(1_000_000), "one million");
        assert_eq!(to_cardinal(2_500_000), "two million, five hundred thousand");
        assert_eq!(to_cardinal(1_000_000_000), "one billion");
    }

    #[test]
    fn ordinals() {
        assert_eq!(to_ordinal(1), "First");
        assert_eq!(to_ordinal(3), "Third");
        assert_eq!(to_ordinal(12), "Twelfth");
        assert_eq!(to_ordinal(21), "twenty-First");
        assert_eq!(to_ordinal(20), "Twentieth");
        assert_eq!(to_ordinal(0), "");
    }

    #[test]
    fn years() {
        assert_eq!(to_year(1999), "nineteen ninety-nine");
        assert_eq!(to_year(2000), "two thousand");
        // Comma, not "and": num2words' grouping separator. It never reaches the
        // listener — the caller splits on non-letters before lexicon lookup.
        assert_eq!(to_year(2005), "two thousand, five");
        assert_eq!(to_year(1900), "nineteen hundred");
        assert_eq!(to_year(1905), "nineteen oh-five");
    }

    #[test]
    fn decimals() {
        assert_eq!(to_decimal("3.5"), "three point five");
        assert_eq!(to_decimal("4.2"), "four point two");
        assert_eq!(to_decimal("47.5"), "forty-seven point five");
        assert_eq!(to_decimal("12.99"), "twelve point nine nine");
        assert_eq!(to_decimal("2.0"), "two", "a trailing zero is not spoken");
        assert_eq!(to_decimal("1,250"), "one thousand, two hundred and fifty");
    }
}
