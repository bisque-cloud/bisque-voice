//! Numbers → Japanese kana, ported from `misaki/num2kana.py`.
//!
//! Japanese groups digits by ten thousand (万) and a hundred million (億), not
//! by thousands, so 12345678 is 千二百三十四万五千六百七十八 — "1234 man, 5678"
//! — and there is no way to read it off the digits positionally.
//!
//! Only the hiragana output is ported; upstream also has kanji and romaji
//! modes, and `misaki/ja.py` calls neither.
//!
//! Several readings change shape when they combine, which is why the tables
//! carry entries like `300` and `8000` rather than composing them: 300 is
//! さんびゃく, not さんひゃく, and 8000 is はっせん, not はちせん.

/// Digit and magnitude readings. Keys are the numeral as written, so the
/// irregular combinations can sit alongside the regular ones.
const HIRAGANA: &[(&str, &str)] = &[
    (".", "てん"),
    ("0", "ゼロ"),
    ("1", "いち"),
    ("2", "に"),
    ("3", "さん"),
    ("4", "よん"),
    ("5", "ご"),
    ("6", "ろく"),
    ("7", "なな"),
    ("8", "はち"),
    ("9", "きゅう"),
    ("10", "じゅう"),
    ("100", "ひゃく"),
    ("1000", "せん"),
    ("10000", "まん"),
    ("100000000", "おく"),
    // Irregular combinations.
    ("300", "さんびゃく"),
    ("600", "ろっぴゃく"),
    ("800", "はっぴゃく"),
    ("3000", "さんぜん"),
    ("8000", "はっせん"),
    ("01000", "いっせん"),
];

fn get(key: &str) -> &'static str {
    HIRAGANA
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, v)| *v)
        .unwrap_or("")
}

fn one(digit: char) -> String {
    get(&digit.to_string()).to_string()
}

/// 10-99.
fn two(digits: &str) -> String {
    let c: Vec<char> = digits.chars().collect();
    if c[0] == '0' {
        return one(c[1]);
    }
    if digits == "10" {
        return get("10").to_string();
    }
    if c[0] == '1' {
        // Eleven is "ten one", not "one ten one".
        return format!("{} {}", get("10"), one(c[1]));
    }
    if c[1] == '0' {
        return format!("{} {}", one(c[0]), get("10"));
    }
    format!("{} {} {}", one(c[0]), get("10"), one(c[1]))
}

/// 100-999.
fn three(digits: &str) -> String {
    let c: Vec<char> = digits.chars().collect();
    let mut parts: Vec<String> = Vec::new();
    match c[0] {
        // A leading 1 is silent: 100 is ひゃく, not いちひゃく.
        '1' => parts.push(get("100").to_string()),
        '3' => parts.push(get("300").to_string()),
        '6' => parts.push(get("600").to_string()),
        '8' => parts.push(get("800").to_string()),
        d => {
            parts.push(one(d));
            parts.push(get("100").to_string());
        }
    }
    let rest: String = c[1..].iter().collect();
    if !(rest == "00" && digits.len() == 3) {
        if c[1] == '0' {
            parts.push(one(c[2]));
        } else {
            parts.push(two(&rest));
        }
    }
    parts.join(" ")
}

/// 1000-9999. `stand_alone` distinguishes a bare thousand from one inside a
/// larger number, which are read differently.
fn four(digits: &str, stand_alone: bool) -> String {
    if digits == "0000" {
        return String::new();
    }
    let trimmed = digits.trim_start_matches('0');
    match trimmed.len() {
        0 => return String::new(),
        1 => return one(trimmed.chars().next().unwrap()),
        2 => return two(trimmed),
        3 => return three(trimmed),
        _ => {}
    }

    let c: Vec<char> = trimmed.chars().collect();
    let mut parts: Vec<String> = Vec::new();
    match c[0] {
        '1' if stand_alone => parts.push(get("1000").to_string()),
        // Inside a larger number a leading thousand is いっせん, not せん.
        '1' => parts.push(get("01000").to_string()),
        '3' => parts.push(get("3000").to_string()),
        '8' => parts.push(get("8000").to_string()),
        d => {
            parts.push(one(d));
            parts.push(get("1000").to_string());
        }
    }
    let rest: String = c[1..].iter().collect();
    if !(rest == "000" && trimmed.len() == 4) {
        if c[1] == '0' {
            parts.push(two(&c[2..].iter().collect::<String>()));
        } else {
            parts.push(three(&rest));
        }
    }
    parts.join(" ")
}

/// Five digits and up, to nine. Grouped by 万 and 億.
fn large(digits: &str) -> Option<String> {
    let head_len = digits.len().checked_sub(4)?;
    let head = &digits[..head_len];
    let mut parts: Vec<String> = Vec::new();
    match head.len() {
        1 => {
            parts.push(one(head.chars().next().unwrap()));
            parts.push(get("10000").to_string());
        }
        2 => {
            parts.push(two(&digits[..2]));
            parts.push(get("10000").to_string());
        }
        3 => {
            parts.push(three(&digits[..3]));
            parts.push(get("10000").to_string());
        }
        4 => {
            parts.push(four(&digits[..4], false));
            parts.push(get("10000").to_string());
        }
        5 => {
            parts.push(one(digits.chars().next().unwrap()));
            parts.push(get("100000000").to_string());
            parts.push(four(&digits[1..5], false));
            if &digits[1..5] != "0000" {
                parts.push(get("10000").to_string());
            }
        }
        // Upstream asserts here; the caller leaves the number alone instead.
        _ => return None,
    }
    parts.push(four(&digits[digits.len() - 4..], false));
    Some(parts.join(" "))
}

fn convert_integer(digits: &str) -> Option<String> {
    Some(match digits.len() {
        1 => one(digits.chars().next()?),
        2 => two(digits),
        3 => three(digits),
        4 => four(digits, true),
        _ => large(digits)?,
    })
}

/// `num2kana.Convert(n, 'hiragana')`.
///
/// Returns `None` for anything upstream refuses — more than nine digits, or a
/// shape that is not a number — so the caller can leave the text alone rather
/// than emit a Japanese error message into the narration.
pub fn convert(number: &str) -> Option<String> {
    let number = number.replace(',', "");
    if number.is_empty() || number.len() > 9 {
        return None;
    }
    if !number.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return None;
    }

    // Leading zeros are dropped, but a lone "0" survives.
    let mut trimmed = number.as_str();
    while trimmed.len() > 1 && trimmed.starts_with('0') {
        trimmed = &trimmed[1..];
    }

    let result = match trimmed.split_once('.') {
        None => convert_integer(trimmed)?,
        Some((integer, decimal)) => {
            let mut tail = String::from(" ");
            for c in decimal.chars() {
                tail.push_str(&one(c));
                tail.push(' ');
            }
            let int_chars: Vec<char> = integer.chars().collect();
            // A whole part ending in a single zero takes a small tsu before
            // the decimal point: 10.5 is じゅっ・てん, not じゅう・てん.
            let ends_in_one_zero = int_chars.len() >= 2
                && int_chars[int_chars.len() - 1] == '0'
                && int_chars[int_chars.len() - 2] != '0';
            if ends_in_one_zero {
                let whole = convert(integer)?;
                let mut chars: Vec<char> = whole.chars().collect();
                chars.pop();
                chars.push('っ');
                format!("{}{}{}", chars.into_iter().collect::<String>(), get("."), tail)
            } else {
                format!("{} {}{}", convert(integer)?, get("."), tail)
            }
        }
    };

    // Hiragana output carries no spaces; they are only a construction aid.
    Some(result.replace(' ', ""))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(s: &str) -> String {
        convert(s).unwrap_or_else(|| panic!("{s} did not convert"))
    }

    #[test]
    fn single_digits() {
        assert_eq!(c("0"), "ゼロ");
        assert_eq!(c("1"), "いち");
        assert_eq!(c("9"), "きゅう");
    }

    #[test]
    fn a_leading_one_is_silent_in_tens_and_hundreds() {
        assert_eq!(c("10"), "じゅう", "ten, not one-ten");
        assert_eq!(c("11"), "じゅういち");
        assert_eq!(c("100"), "ひゃく", "hundred, not one-hundred");
    }

    #[test]
    fn tens_combine_the_obvious_way() {
        assert_eq!(c("20"), "にじゅう");
        assert_eq!(c("21"), "にじゅういち");
        assert_eq!(c("99"), "きゅうじゅうきゅう");
    }

    /// The readings that change shape when they combine. These are why the
    /// table carries 300/600/800/3000/8000 as their own entries.
    #[test]
    fn irregular_combinations_are_not_composed_from_parts() {
        assert_eq!(c("300"), "さんびゃく", "not さんひゃく");
        assert_eq!(c("600"), "ろっぴゃく");
        assert_eq!(c("800"), "はっぴゃく");
        assert_eq!(c("3000"), "さんぜん", "not さんせん");
        assert_eq!(c("8000"), "はっせん");
    }

    #[test]
    fn japanese_groups_by_ten_thousand_not_by_thousand() {
        assert_eq!(c("10000"), "いちまん");
        assert_eq!(c("100000"), "じゅうまん");
        assert_eq!(c("100000000"), "いちおく");
    }

    /// Standalone 1000 is せん; the same thousand inside a larger number is
    /// いっせん. This is what the `stand_alone` flag exists for.
    #[test]
    fn a_thousand_inside_a_larger_number_reads_differently() {
        assert_eq!(c("1000"), "せん");
        assert_eq!(c("10000000"), "いっせんまん");
    }

    /// Checked against Python Misaki directly, since these are the cases where
    /// the 万/億 grouping is easiest to get subtly wrong.
    #[test]
    fn large_numbers_match_upstream_exactly() {
        assert_eq!(c("1234"), "せんにひゃくさんじゅうよん");
        assert_eq!(c("2026"), "にせんにじゅうろく");
        assert_eq!(c("56789"), "ごまんろくせんななひゃくはちじゅうきゅう");
        assert_eq!(
            c("12345678"),
            "いっせんにひゃくさんじゅうよんまんごせんろっぴゃくななじゅうはち"
        );
        assert_eq!(
            c("999999999"),
            "きゅうおくきゅうせんきゅうひゃくきゅうじゅうきゅうまんきゅうせんきゅうひゃくきゅうじゅうきゅう"
        );
    }

    /// The decimal branch is **unreachable from `JAG2P`** and is ported only
    /// because it is in the source.
    ///
    /// `ja.py` splits text with `re.findall(r'\d+|\D+', ...)` and converts a
    /// run only `if t.isdigit()`, so "3.5" arrives as "3", ".", "5" — three
    /// tokens, and `"3.5".isdigit()` is False regardless. Nothing containing a
    /// point ever reaches here.
    ///
    /// Just as well: upstream *crashes* on a decimal whose whole part is a
    /// single digit (`Convert("0.25")` raises IndexError, indexing `[-2]` of a
    /// one-character string). Only the values Python actually produces are
    /// asserted.
    #[test]
    fn the_unreachable_decimal_branch_matches_upstream_where_upstream_works() {
        assert_eq!(c("3.5"), "さんてんご");
        // A whole part ending in a single zero takes a small tsu.
        assert_eq!(c("10.5"), "じゅってんご");
    }

    #[test]
    fn leading_zeros_are_dropped_but_zero_itself_survives() {
        assert_eq!(c("007"), "なな");
        assert_eq!(c("0"), "ゼロ");
    }

    #[test]
    fn commas_are_ignored() {
        assert_eq!(c("1,000"), c("1000"));
    }

    /// Upstream asserts past nine digits. Refusing lets the caller leave the
    /// digits in place rather than narrate an error string.
    #[test]
    fn more_than_nine_digits_is_refused_rather_than_guessed() {
        assert_eq!(convert("1234567890"), None);
        assert_eq!(convert(""), None);
        assert_eq!(convert("abc"), None);
    }

    #[test]
    fn output_carries_no_spaces() {
        for n in ["1", "21", "300", "1234", "56789"] {
            assert!(!c(n).contains(' '), "{n} produced spaces");
        }
    }
}
