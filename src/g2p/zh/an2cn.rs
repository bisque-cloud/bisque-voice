//! Arabic numerals → Chinese numerals, ported from the `cn2an` package.
//!
//! Misaki rewrites digits into Chinese characters *before* phonemizing, so
//! "3.5倍" is spoken "three point five times" rather than having the digits
//! skipped. Only the `an2cn` direction is ported; `cn2an` (the reverse) is
//! never called from this path.
//!
//! Two conversion modes matter, and the difference is not cosmetic:
//!
//! - **low** reads the number as a quantity: `2026` → 二千零二十六, "two
//!   thousand and twenty-six".
//! - **direct** reads it digit by digit: `2026` → 二零二六, "two zero two six".
//!
//! Years take `direct` and everything else takes `low`, which is why "2026年"
//! is read as a year and a bare "2026" as a count.

use fancy_regex::{Captures, Regex};

const DIGITS: [&str; 10] = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];

/// Positional units, least significant first: ones, tens, hundreds, thousands,
/// then the cycle repeats above 万 (ten thousand) and 亿 (a hundred million).
const UNITS: [&str; 16] = [
    "", "十", "百", "千", "万", "十", "百", "千", "亿", "十", "百", "千", "万", "十", "百", "千",
];

/// Measure words, used only to spot a numeric range ("10到20斤").
const MEASURE_WORDS: &str = "斤|克|千克|公斤|吨|米|厘米|毫米|公里|升|毫升|元|角|分|个|只|条|张|块|瓶|杯|份|本|辆|台|匹|头|位|亩|小时|分钟|秒|天|半";

/// Read a digit string as a quantity. `An2Cn.__integer_convert(mode="low")`.
fn integer_low(digits: &str) -> String {
    // Strip leading zeros the way `str(int(x))` does, but without overflowing
    // on a long run of digits.
    let trimmed = digits.trim_start_matches('0');
    let trimmed = if trimmed.is_empty() { "0" } else { trimmed };

    let chars: Vec<char> = trimmed.chars().collect();
    let len = chars.len();
    if len > UNITS.len() {
        // Upstream raises; the caller leaves the number alone instead.
        return String::new();
    }

    let mut out = String::new();
    for (i, c) in chars.iter().enumerate() {
        let d = c.to_digit(10).unwrap_or(0) as usize;
        let position = len - i - 1;
        if d != 0 {
            out.push_str(DIGITS[d]);
            out.push_str(UNITS[position]);
        } else {
            // A zero still carries its unit at a 万/亿 boundary, so
            // 100000000 is 一亿 rather than 一.
            if position % 4 == 0 {
                out.push_str(DIGITS[0]);
                out.push_str(UNITS[position]);
            }
            // ...and otherwise contributes a single 零, never two in a row.
            if i > 0 && !out.ends_with('零') {
                out.push_str(DIGITS[0]);
            }
        }
    }

    let mut out = out
        .replace("零零", "零")
        .replace("零万", "万")
        .replace("零亿", "亿")
        .replace("亿万", "亿");
    out = out.trim_matches('零').to_string();

    // 万零X千 → 万X千: a thousand immediately after a myriad needs no filler.
    let re = Regex::new(r"([万亿])零([一二三四五六七八九壹贰叁肆伍陆柒捌玖][千仟])")
        .expect("static regex");
    out = re
        .replace_all(&out, |c: &Captures| format!("{}{}", &c[1], &c[2]))
        .into_owned();

    // 一十 is said 十: "ten", not "one ten".
    if out.starts_with("一十") {
        out = out.chars().skip(1).collect();
    }
    if out.is_empty() {
        out = DIGITS[0].to_string();
    }
    out
}

/// Read a digit string one digit at a time. `An2Cn.__direct_convert`.
fn direct(digits: &str) -> String {
    digits
        .chars()
        .map(|c| match c {
            '.' => "点".to_string(),
            c => c
                .to_digit(10)
                .map(|d| DIGITS[d as usize].to_string())
                .unwrap_or_else(|| c.to_string()),
        })
        .collect()
}

/// The decimal part, read digit by digit after 点 ("point").
fn decimal_low(digits: &str) -> String {
    if digits.is_empty() {
        return String::new();
    }
    // Upstream truncates past 16 places with a warning.
    let digits: String = digits.chars().take(16).collect();
    let mut out = "点".to_string();
    for c in digits.chars() {
        if let Some(d) = c.to_digit(10) {
            out.push_str(DIGITS[d as usize]);
        }
    }
    out
}

/// `An2Cn.an2cn(inputs, mode)` for the two modes this path uses.
fn an2cn(input: &str, direct_mode: bool) -> String {
    if input.is_empty() {
        return String::new();
    }
    let (sign, body) = match input.strip_prefix('-') {
        Some(rest) => ("负", rest),
        None => ("", input),
    };
    if body.chars().any(|c| !c.is_ascii_digit() && c != '.') {
        // Upstream raises on anything outside 0-9 . -; every caller here has
        // already matched a digit pattern, so this is a guard, not a path.
        return input.to_string();
    }
    if direct_mode {
        return format!("{sign}{}", direct(body));
    }
    let mut parts = body.splitn(2, '.');
    let integer = parts.next().unwrap_or("");
    let converted = match parts.next() {
        None => integer_low(integer),
        Some(decimal) => format!("{}{}", integer_low(integer), decimal_low(decimal)),
    };
    if converted.is_empty() {
        return input.to_string();
    }
    format!("{sign}{converted}")
}

/// `Transform.transform(inputs, "an2cn")`.
///
/// Six substitutions in order. Each is a distinct reading of the same digits —
/// a year is read differently from a count, a percentage carries 百分之 in
/// front — so the specific patterns and their order are the specification.
pub struct An2Cn {
    range: Regex,
    date: Regex,
    fraction: Regex,
    percent: Regex,
    celsius: Regex,
    number: Regex,
    year: Regex,
    digits: Regex,
}

impl Default for An2Cn {
    fn default() -> Self {
        Self::new()
    }
}

impl An2Cn {
    pub fn new() -> Self {
        An2Cn {
            range: Regex::new(&format!(
                r"\d+(\.\d+)?-\d+(\.\d+)?(?=({MEASURE_WORDS}))"
            ))
            .expect("static regex"),
            date: Regex::new(r"(\d{2,4}年)?(\d{1,2}月)?(\d{1,2}日)?").expect("static regex"),
            fraction: Regex::new(r"\d+/\d+").expect("static regex"),
            percent: Regex::new(r"-?(\d+\.)?\d+%").expect("static regex"),
            celsius: Regex::new(r"\d+℃").expect("static regex"),
            number: Regex::new(r"-?(\d+\.)?\d+").expect("static regex"),
            year: Regex::new(r"\d+(?=年)").expect("static regex"),
            digits: Regex::new(r"\d+").expect("static regex"),
        }
    }

    /// Replace every non-empty match. A zero-width match is left alone, which
    /// is what Python's `re.sub` does when the replacement function returns
    /// `None` — and the date pattern, being entirely optional, matches
    /// zero-width at every position.
    fn sub(re: &Regex, text: &str, f: impl Fn(&str) -> String) -> String {
        let mut out = String::with_capacity(text.len());
        let mut last = 0usize;
        for m in re.find_iter(text) {
            let Ok(m) = m else { continue };
            if m.start() == m.end() {
                continue;
            }
            out.push_str(&text[last..m.start()]);
            out.push_str(&f(m.as_str()));
            last = m.end();
        }
        out.push_str(&text[last..]);
        out
    }

    pub fn transform(&self, text: &str) -> String {
        // A range between measure words: "10到20斤".
        let out = Self::sub(&self.range, text, |m| match m.split_once('-') {
            Some((start, end)) => format!("{}到{}", an2cn(start, false), an2cn(end, false)),
            None => m.to_string(),
        });

        // Dates. The year is read digit by digit; the month and day are not.
        let out = Self::sub(&self.date, &out, |m| {
            let with_year = Self::sub(&self.year, m, |y| an2cn(y, true));
            Self::sub(&self.digits, &with_year, |d| an2cn(d, false))
        });

        // Fractions invert: 1/2 is 二分之一, "of two parts, one".
        let out = Self::sub(&self.fraction, &out, |m| {
            let converted = Self::sub(&self.digits, m, |d| an2cn(d, false));
            match converted.split_once('/') {
                Some((numerator, denominator)) => format!("{denominator}分之{numerator}"),
                None => m.to_string(),
            }
        });

        let out = Self::sub(&self.percent, &out, |m| {
            format!("百分之{}", an2cn(m.trim_end_matches('%'), false))
        });

        let out = Self::sub(&self.celsius, &out, |m| {
            format!("{}摄氏度", an2cn(m.trim_end_matches('℃'), false))
        });

        Self::sub(&self.number, &out, |m| an2cn(m, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str) -> String {
        An2Cn::new().transform(s)
    }

    /// The distinction that matters most: a year is read digit by digit, a
    /// bare number as a quantity. Both of these are "2026".
    #[test]
    fn a_year_is_read_digit_by_digit_and_a_count_is_not() {
        assert_eq!(t("2026年"), "二零二六年");
        assert_eq!(t("2026"), "二千零二十六");
    }

    #[test]
    fn quantities_place_their_units() {
        assert_eq!(t("10"), "十", "not 一十");
        assert_eq!(t("11"), "十一");
        assert_eq!(t("20"), "二十");
        assert_eq!(t("100"), "一百");
        assert_eq!(t("1000"), "一千");
        assert_eq!(t("10000"), "一万");
        assert_eq!(t("100000000"), "一亿");
        assert_eq!(t("1234567890"), "十二亿三千四百五十六万七千八百九十");
    }

    #[test]
    fn interior_zeros_collapse_to_a_single_filler() {
        assert_eq!(t("2026"), "二千零二十六");
        assert_eq!(t("1.05"), "一点零五");
        assert_eq!(t("0"), "零");
        assert_eq!(t("007"), "七", "leading zeros are dropped");
    }

    #[test]
    fn decimals_are_read_digit_by_digit_after_the_point() {
        assert_eq!(t("3.5"), "三点五");
        assert_eq!(t("0.5"), "零点五");
        assert_eq!(t("19.99"), "十九点九九");
        assert_eq!(t("3.0"), "三点零");
    }

    #[test]
    fn a_negative_number_keeps_its_sign() {
        assert_eq!(t("-3"), "负三");
    }

    #[test]
    fn percentages_fractions_and_temperatures_have_their_own_readings() {
        assert_eq!(t("100%"), "百分之一百");
        assert_eq!(t("1/2"), "二分之一", "the denominator is said first");
        assert_eq!(t("25℃"), "二十五摄氏度");
    }

    #[test]
    fn full_dates_read_the_year_apart_from_the_month_and_day() {
        assert_eq!(t("2026年1月5日"), "二零二六年一月五日");
        assert_eq!(t("2026年1月"), "二零二六年一月");
        assert_eq!(t("3月5日"), "三月五日");
    }

    #[test]
    fn text_without_digits_is_returned_unchanged() {
        // The date pattern matches zero-width at every position; if those
        // matches were substituted rather than skipped, every character would
        // be rewritten.
        assert_eq!(t("abc"), "abc");
        assert_eq!(t("测试"), "测试");
        assert_eq!(t(""), "");
    }

    #[test]
    fn digits_inside_a_sentence_are_converted_in_place() {
        assert_eq!(t("3.5倍的增长。"), "三点五倍的增长。");
        assert_eq!(t("我买了3.5公斤苹果。"), "我买了三点五公斤苹果。");
        assert_eq!(t("2026年的测试。"), "二零二六年的测试。");
        assert_eq!(t("第3章"), "第三章");
    }

    #[test]
    fn a_range_between_measure_words_becomes_x_to_y() {
        assert_eq!(t("10到20"), "十到二十");
    }
}
