//! The retake gate — catching a garbled Studio take before it ships.
//!
//! Qwen3-TTS samples its speech, so one chunk in a few dozen can come out as
//! fluent-sounding nonsense: right voice, plausible rhythm, no words. The same
//! text, voice and settings is clean on the next take. Nothing in the audio's
//! level, length or silence tells the two apart, so no amplitude rule here
//! does either.
//!
//! The forced aligner does tell them apart. Asked to place words that were
//! never said, it cannot, and it stacks them on top of each other: runs of
//! consecutive words get zero length at one instant. On the Skill sync
//! presentation that shipped garbled (`present-ab557771…`, slide 14), 34 of
//! 84 words sat in such stacks. A clean take has a lone zero-length word now
//! and then (Japanese, with its short morae, most often) but almost never two
//! in a row. So the gate is a measurement the pipeline already takes, read for
//! what it says about the take.
//!
//! What it does, in order:
//!
//! 1. Each Studio chunk is aligned against its own text as soon as it is
//!    synthesized (`Gate::judge`).
//! 2. A take whose words collapse is synthesized again, same text, same voice,
//!    same settings — at most `MAX_TAKES` times in all (`take_until_accepted`).
//! 3. If every take collapses, narration fails with the chunk named, before
//!    anything is written. The publish path stops on that exit status, so the
//!    slide's audio is never uploaded.
//!
//! Three more things are retaken, never failed:
//!
//! - A take that opens with a sound the aligner mistook for its first word
//!   (`lead_absorbed`). The trim below cuts sound before the first word, so a
//!   chuckle the aligner calls "A" is kept as if it were speech.
//! - A take with a stretch the aligner was unsure of (`UNSURE_BELOW`): babble
//!   before the first word that the aligner stretched a word over, or a
//!   made-up word in the middle ("A skill is Naxmourning instructions"). The
//!   timings look ordinary; how sure the aligner was of them does not.
//! - A take that stopped before its last words (`dropped_tail`). Asked to
//!   place words the audio ran out before, the aligner piles them onto the
//!   take's last instant or past it ("statt sie liegen zu lassen" ending on
//!   "sie").
//!
//! What it does not do: judge pitch or style drift (the aligner cannot see
//! them, and a retake would not fix the voice's own range), judge Kokoro
//! (deterministic — a retake returns the same audio), or judge anything when
//! no aligner is installed. A one-syllable insertion can still pass, so can a
//! take missing only its last word, and Korean is not judged for sureness
//! (`read`); the measured misses are listed in `fixtures/quality/sureness.json`.
//! The thresholds come from measured takes, and `fixtures/quality/` holds
//! those takes' real timing shapes. Their words were swapped for neutral ones
//! of the same script, and the `heard` transcripts with them; every timing,
//! sureness value and label is as measured. Nothing here reads more of a
//! token's text than whether it holds a letter or Hangul.

use anyhow::{bail, Result};

/// Takes per chunk, the first included. A clean retake followed the garbled
/// slide-14 chunk three times out of three, and the measured rate is about one
/// garbled chunk in 36, so a third collapse in a row is not bad luck — it is
/// text this voice cannot say, and more takes would only spend minutes.
pub const MAX_TAKES: usize = 3;

/// The aligner's resolution, in milliseconds. A span shorter than one unit is
/// a span of zero: the word was given no time at all.
const UNIT_MS: i64 = 80;

/// Below this many placeable tokens there is too little to judge: a two-word
/// chunk with both words stacked is 100% stacked and means nothing.
const MIN_TOKENS: usize = 8;

/// The share of placeable tokens sitting in stacks at which a take is judged
/// collapsed. On 280 held takes the two garbled ones read 40% and 30%, and the
/// highest clean one 5% (a Japanese take); every take and its reading is in
/// `fixtures/quality/takes.json`.
const STACKED_SHARE: f64 = 0.15;

/// The gap between a take's first two words at which the first may have
/// been placed on something else. The two absorbed openings measured left
/// 480 ms and 960 ms. 40 of 328 held takes reach 240 ms, nearly all a pause
/// after a word like "So", "Here's" or "Allora", and only those pay for the
/// second alignment.
const LEAD_GAP_MS: i64 = 240;

/// How many consecutive words the aligner's sureness is averaged over.
/// Babble or a made-up word costs the words around it their sureness too,
/// so ten words is long enough that one hard name ("Snyk", "chezmoi") does
/// not drag a clean stretch down, and short enough that 1.5 s of babble does.
const UNSURE_WINDOW: usize = 10;

/// The average sureness, over the least sure `UNSURE_WINDOW` words of a take,
/// below which it is retaken. On 286 measured English takes the least sure
/// clean one read 0.44, and 9 of 11 that recognition heard babble, garble or
/// an inserted word in read 0.42 or less. Fixed before the held-out takes were
/// made: of 102 fresh ones in nine languages it flagged both with a defect
/// (an Italian garble the stacking check let through, a laugh before slide
/// 19) and one of the 100 clean ones. Every take and its reading is in
/// `fixtures/quality/sureness.json`.
const UNSURE_BELOW: f64 = 0.43;

/// How many words placed where a take's audio ends mean it stopped before
/// them. A clean take ends on its last word, so only that word starts in the
/// final aligner step, and only sometimes. The two takes recognition heard
/// stop early — German that ended at "statt sie", Korean that ended five words
/// short — placed three and five words there, exactly the ones not said. Of
/// 354 other chunk takes, in ten languages, and 282 narrations, none placed
/// two. Held out, with the rule fixed first: 109 fresh takes, 60 of them
/// German, flagged none, and recognition heard each reach its last word.
/// Whole takes cut before their last two or three words were caught 241
/// times in 248. Cut before only their last word, 22 times in 124: one late
/// word is how 9 whole takes end, so it is not judged, and a take missing
/// only its last word usually passes. Every take's ending is in
/// `fixtures/quality/tails.json`.
const TAIL_WORDS: usize = 2;

/// One token the aligner was asked to place, with where it put it.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenSpan {
    pub text: String,
    pub start_ms: i64,
    pub end_ms: i64,
    /// How sure the aligner was of this token's place: the lower of the
    /// probabilities it gave its start and its end. `None` when the spans
    /// came from somewhere that does not say.
    pub sure: Option<f32>,
}

/// What the gate measured on one take.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reading {
    /// Tokens that carry a letter or a digit — the ones a voice has to say.
    pub placeable: usize,
    /// Of those, how many sit in a run of two or more consecutive tokens the
    /// aligner gave no time: words stacked where it could not find them.
    pub stacked: usize,
    /// Where the first placeable word starts and the last one ends, in
    /// milliseconds on the take's own clock. `None` with nothing to place.
    pub speech_ms: Option<(i64, i64)>,
    /// The first word was found again later in the take: what the aligner
    /// first gave it was a sound before the speech (`lead_absorbed`).
    pub absorbed_lead: bool,
    /// The average sureness of the take's least sure `UNSURE_WINDOW` words.
    /// `None` when too few words to judge, or no sureness was measured.
    pub least_sure: Option<f64>,
    /// How many of the last words were placed where the take's audio ends:
    /// words the voice stopped before (`dropped_tail`). Zero for a whole take.
    pub dropped_tail: usize,
}

impl Reading {
    pub fn share(&self) -> f64 {
        if self.placeable == 0 {
            0.0
        } else {
            self.stacked as f64 / self.placeable as f64
        }
    }

    /// True when the words could not be found in the audio.
    pub fn is_collapse(&self) -> bool {
        self.placeable >= MIN_TOKENS && self.share() >= STACKED_SHARE
    }

    /// True when some stretch of the take is not clearly its words.
    pub fn is_unsure(&self) -> bool {
        self.least_sure.is_some_and(|s| s < UNSURE_BELOW)
    }

    /// True when the take is speech but not the right speech: retaken, and
    /// kept only if no take is clean.
    fn is_flawed(&self) -> bool {
        self.absorbed_lead || self.is_unsure() || self.dropped_tail > 0
    }
}

impl std::fmt::Display for Reading {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} of {} words stacked at no time ({:.0}%)",
            self.stacked,
            self.placeable,
            self.share() * 100.0
        )
    }
}

/// Read one take's token spans.
///
/// Punctuation-only tokens are left out: `—` or `&` is something the model
/// places badly by design, and a slide full of them is not a garbled one.
pub fn read(tokens: &[TokenSpan]) -> Reading {
    let placeable: Vec<&TokenSpan> = tokens
        .iter()
        .filter(|t| t.text.chars().any(char::is_alphanumeric))
        .collect();
    let zero: Vec<bool> = placeable
        .iter()
        .map(|t| t.end_ms - t.start_ms < UNIT_MS)
        .collect();
    let mut stacked = 0;
    let mut run = 0;
    for z in zero.iter().copied().chain([false]) {
        if z {
            run += 1;
        } else {
            if run >= 2 {
                stacked += run;
            }
            run = 0;
        }
    }
    // Korean is not judged: its clean takes read 0.33 to 0.53, across the
    // whole range a defect reads, so the line would retake half of them.
    let sure: Option<Vec<f32>> = if placeable.iter().any(|t| t.text.chars().any(is_hangul)) {
        None
    } else {
        placeable.iter().map(|t| t.sure).collect()
    };
    Reading {
        placeable: zero.len(),
        stacked,
        speech_ms: match (placeable.first(), placeable.last()) {
            (Some(first), Some(last)) => Some((first.start_ms, last.end_ms)),
            _ => None,
        },
        absorbed_lead: false,
        least_sure: sure
            .filter(|s| s.len() >= MIN_TOKENS)
            .map(|s| least_sure(&s)),
        dropped_tail: 0,
    }
}

/// How many words a take stopped before: its last `TAIL_WORDS` or more
/// words all placed to start within one aligner step of where the audio
/// ends, or after it. `take_ms` is the take's length. Zero means the take
/// ends on its words.
///
/// Qwen3 can emit its end of speech early. The take is then fluent and every
/// word that was said sits where it should, so neither the stacked share nor
/// sureness need see it. The words never said have nowhere to go: the
/// aligner piles them onto the last instant there is, or past it with a
/// length the audio does not have.
pub fn dropped_tail(tokens: &[TokenSpan], take_ms: i64) -> usize {
    let late = tokens
        .iter()
        .filter(|t| t.text.chars().any(char::is_alphanumeric))
        .rev()
        .take_while(|t| t.start_ms >= take_ms - UNIT_MS)
        .count();
    if late >= TAIL_WORDS {
        late
    } else {
        0
    }
}

fn is_hangul(c: char) -> bool {
    matches!(c, '\u{1100}'..='\u{11FF}' | '\u{3130}'..='\u{318F}' | '\u{AC00}'..='\u{D7A3}')
}

/// The lowest average over any `UNSURE_WINDOW` consecutive values, or over
/// all of them when there are fewer.
fn least_sure(sure: &[f32]) -> f64 {
    sure.windows(UNSURE_WINDOW.min(sure.len()))
        .map(|w| w.iter().map(|&v| f64::from(v)).sum::<f64>() / w.len() as f64)
        .fold(f64::INFINITY, f64::min)
}

fn first_two(tokens: &[TokenSpan]) -> Option<(&TokenSpan, &TokenSpan)> {
    let mut placeable = tokens
        .iter()
        .filter(|t| t.text.chars().any(char::is_alphanumeric));
    Some((placeable.next()?, placeable.next()?))
}

/// Where to cut a take to check its first word again, or `None` when its
/// first two words sit too close for the first to have been misplaced.
///
/// Qwen3 sometimes opens a chunk with a sound that is not a word ("Ugh",
/// "Haha"), and the aligner, made to put the first word somewhere, can put it
/// there. The real first word then sits unplaced in what looks like a pause
/// before the second. The cut is where the aligner ended the first word.
pub fn lead_cut_ms(tokens: &[TokenSpan]) -> Option<i64> {
    let (first, second) = first_two(tokens)?;
    (second.start_ms - first.end_ms >= LEAD_GAP_MS).then_some(first.end_ms)
}

/// Whether aligning again from the cut found the first word in the gap.
///
/// `recut` is the same text aligned against the audio after the cut, moved
/// back onto the take's clock. Absorbed, the first word turns up inside the
/// gap: past the silence after the sound it was first given, with a real
/// length, and with the second word where it was. A genuine pause reads
/// otherwise. With the real word cut away, the aligner gives the first word
/// no time, pushes the second word later to make room, or puts it straight
/// back on the tail of where it was ("Allora," and "Então," did this).
///
/// Measured on 328 takes, 40 of them with a gap to check
/// (`fixtures/quality/leads.json`): true for both openings recognition heard
/// as "Ugh" and "Haha", false for all 38 clean ones, in English and nine other
/// languages. Held out, with the rule fixed first: 40 fresh takes of the
/// chunk that opened with "Ugh" flagged only the one that did so again, and 48
/// fresh narrations in ten languages flagged none. Every condition is one or
/// two of the aligner's 80 ms steps, and only three absorbed openings have
/// been seen, so this is a retake, never a failure.
pub fn lead_absorbed(tokens: &[TokenSpan], recut: &[TokenSpan]) -> bool {
    let (Some((first, second)), Some((first_again, second_again))) =
        (first_two(tokens), first_two(recut))
    else {
        return false;
    };
    first_again.start_ms - first.end_ms >= 2 * UNIT_MS
        && first_again.end_ms - first_again.start_ms >= UNIT_MS
        && (second_again.start_ms - second.start_ms).abs() < UNIT_MS
}

/// Cut a take down to its aligned words, keeping the same padding the whole
/// narration is trimmed with (`audio::trim_to_aligned_words`).
///
/// That trim only reaches the narration's two ends. A take for the second or
/// third chunk of a slide can open with sound that is not a word — a chuckle,
/// a run of "hm" — and once concatenated it sits mid-slide where nothing cut
/// it. Measured on the Skill sync text: a second chunk whose first word
/// aligned at 1.1 s, where recognition heard "Hehehe" before it; clean takes
/// place their first word at 0.08 s.
pub fn trim_to_speech(label: &str, pcm: Vec<f32>, reading: &Reading) -> Vec<f32> {
    let Some((start_ms, end_ms)) = reading.speech_ms else {
        return pcm;
    };
    let rate = f64::from(crate::audio::SAMPLE_RATE);
    let start_s = (start_ms as f64 / 1000.0 - crate::audio::TRIM_LEAD_PAD_S).max(0.0);
    let end_s = end_ms as f64 / 1000.0 + crate::audio::TRIM_TAIL_PAD_S;
    let start_i = (start_s * rate).round() as usize;
    let end_i = ((end_s * rate).round() as usize).min(pcm.len());
    if start_i >= end_i {
        // Degenerate timings — leave the audio alone rather than emit nothing.
        return pcm;
    }
    let lead = start_i as f64 / rate;
    let tail = (pcm.len() - end_i) as f64 / rate;
    if lead + tail < crate::audio::TRIM_MIN_CUT_S {
        return pcm;
    }
    if lead >= 0.5 || tail >= 0.5 {
        eprintln!(
            "bisque-voice: {label} had {lead:.2}s before its first word and {tail:.2}s \
             after its last; cut to the words"
        );
    }
    pcm[start_i..end_i].to_vec()
}

/// Something that can judge one synthesized chunk.
pub trait Gate {
    /// `text` is what the engine was asked to say for this chunk; `pcm` is
    /// what it produced, 24 kHz mono. `None` means there was nothing to judge
    /// with, which is not the same as a pass and is never reported as one.
    fn judge(&mut self, text: &str, pcm: &[f32]) -> Result<Option<Reading>>;
}

/// The gate for a run with no aligner: it cannot see a collapse, so it lets
/// every take through and says nothing about it.
pub struct Blind;

impl Gate for Blind {
    fn judge(&mut self, _text: &str, _pcm: &[f32]) -> Result<Option<Reading>> {
        Ok(None)
    }
}

/// Synthesize one chunk until a take is not a collapse, at most `MAX_TAKES`
/// times. `synth` makes one take; `label` names the chunk in messages.
///
/// A take with an absorbed opening, an unsure stretch or a dropped ending is
/// retaken too, within the same bound, but it is still speech: if no clean
/// take follows, the most complete and then the surest of them is kept rather
/// than failing the narration.
pub fn take_until_accepted(
    label: &str,
    text: &str,
    gate: &mut dyn Gate,
    mut synth: impl FnMut() -> Result<Vec<f32>>,
) -> Result<Vec<f32>> {
    let mut last = None;
    let mut fallback: Option<(Vec<f32>, Reading)> = None;
    for take in 1..=MAX_TAKES {
        let pcm = synth()?;
        match gate.judge(text, &pcm)? {
            Some(reading) if reading.is_collapse() => {
                if take < MAX_TAKES {
                    eprintln!(
                        "bisque-voice: {label} came out unintelligible: {reading}. \
                         Synthesizing it again, take {} of {MAX_TAKES}",
                        take + 1
                    );
                }
                last = Some(reading);
            }
            Some(reading) if reading.is_flawed() => {
                if take < MAX_TAKES {
                    let what = if reading.dropped_tail > 0 {
                        format!("stopped before its last {} words", reading.dropped_tail)
                    } else if reading.absorbed_lead {
                        "opened with a sound that is not a word".to_string()
                    } else {
                        "has a stretch that does not sound like its words".to_string()
                    };
                    eprintln!(
                        "bisque-voice: {label} {what}. Synthesizing it again, take {} of \
                         {MAX_TAKES}",
                        take + 1
                    );
                }
                // Keep the flawed take missing the fewest words, then the
                // surest; on a tie, the earlier one.
                let better = |r: &Reading, than: &Reading| {
                    (r.dropped_tail, -r.least_sure.unwrap_or(1.0))
                        < (than.dropped_tail, -than.least_sure.unwrap_or(1.0))
                };
                if fallback
                    .as_ref()
                    .is_none_or(|(_, kept)| better(&reading, kept))
                {
                    fallback = Some((pcm, reading));
                }
            }
            Some(reading) => return Ok(trim_to_speech(label, pcm, &reading)),
            None => return Ok(pcm),
        }
    }
    if let Some((pcm, reading)) = fallback {
        eprintln!(
            "bisque-voice: {label} had a flaw in every take that could be understood; \
             keeping the best of them"
        );
        return Ok(trim_to_speech(label, pcm, &reading));
    }
    let reading = last.expect("the loop runs at least once and records every rejection");
    bail!(
        "{label} came out unintelligible in all {MAX_TAKES} takes; in the last, {reading}. \
         Nothing was written. The text was {text:?}. Run it again to retry; if it \
         fails again, this voice may not be able to say this text."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(text: &str, start_ms: i64, end_ms: i64) -> TokenSpan {
        TokenSpan {
            text: text.to_string(),
            start_ms,
            end_ms,
            sure: None,
        }
    }

    /// A gate that answers from a script, one reading per take.
    struct Scripted(Vec<Option<Reading>>);

    impl Gate for Scripted {
        fn judge(&mut self, _text: &str, _pcm: &[f32]) -> Result<Option<Reading>> {
            Ok(self.0.remove(0))
        }
    }

    const BAD: Reading = Reading {
        placeable: 40,
        stacked: 30,
        speech_ms: Some((0, 1000)),
        absorbed_lead: false,
        least_sure: None,
        dropped_tail: 0,
    };
    const GOOD: Reading = Reading {
        placeable: 40,
        stacked: 0,
        speech_ms: None,
        absorbed_lead: false,
        least_sure: None,
        dropped_tail: 0,
    };

    #[test]
    fn a_clean_first_take_is_synthesized_once() {
        let mut calls = 0;
        let mut gate = Scripted(vec![Some(GOOD)]);
        take_until_accepted("chunk 1/1", "text", &mut gate, || {
            calls += 1;
            Ok(vec![0.1])
        })
        .unwrap();
        assert_eq!(calls, 1);
    }

    #[test]
    fn a_collapsed_take_is_retaken_and_the_clean_one_returned() {
        let mut calls = 0;
        let mut gate = Scripted(vec![Some(BAD), Some(GOOD)]);
        let pcm = take_until_accepted("chunk 1/2", "text", &mut gate, || {
            calls += 1;
            Ok(vec![calls as f32])
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(pcm, vec![2.0], "the accepted take is the second one");
    }

    #[test]
    fn retakes_stop_at_the_bound_and_fail_naming_the_chunk() {
        let mut calls = 0;
        // More rejections scripted than the bound allows: the loop must stop
        // on its own count, not on the script running out.
        let mut gate = Scripted(vec![Some(BAD); MAX_TAKES + 5]);
        let err = take_until_accepted("chunk 2/3", "the words", &mut gate, || {
            calls += 1;
            Ok(vec![0.1])
        })
        .unwrap_err()
        .to_string();
        assert_eq!(calls, MAX_TAKES);
        assert!(err.contains("chunk 2/3"), "{err}");
        assert!(err.contains("Nothing was written"), "{err}");
        assert!(err.contains("30 of 40 words stacked"), "{err}");
    }

    #[test]
    fn a_blind_gate_accepts_without_claiming_to_have_checked() {
        let mut calls = 0;
        take_until_accepted("chunk 1/1", "text", &mut Blind, || {
            calls += 1;
            Ok(vec![0.1])
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(Blind.judge("text", &[0.1]).unwrap(), None);
    }

    #[test]
    fn a_synthesis_error_is_not_retaken() {
        let mut calls = 0;
        let mut gate = Scripted(vec![]);
        let err = take_until_accepted("chunk 1/1", "text", &mut gate, || {
            calls += 1;
            bail!("synthesis failed: out of memory")
        });
        assert!(err.is_err());
        assert_eq!(calls, 1, "an engine error is a failure, not a bad take");
    }

    #[test]
    fn punctuation_only_tokens_are_not_counted() {
        // Every punctuation token at zero length, between placed words: none
        // of it is a stack, because none of it is something to say.
        let tokens = [
            span("We", 0, 160),
            span("—", 160, 160),
            span("…", 160, 160),
            span("ship", 160, 400),
            span("&", 400, 400),
            span("test", 400, 640),
        ];
        assert_eq!(
            read(&tokens),
            Reading {
                placeable: 3,
                stacked: 0,
                speech_ms: Some((0, 640)),
                absorbed_lead: false,
                least_sure: None,
                dropped_tail: 0,
            }
        );
    }

    fn reading_with_speech(start_ms: i64, end_ms: i64) -> Reading {
        Reading {
            placeable: 20,
            stacked: 0,
            speech_ms: Some((start_ms, end_ms)),
            absorbed_lead: false,
            least_sure: None,
            dropped_tail: 0,
        }
    }

    /// Two seconds of samples at 24 kHz, each sample its own index, so a cut
    /// can be read back off the first and last values.
    fn ramp() -> Vec<f32> {
        (0..48_000).map(|i| i as f32).collect()
    }

    #[test]
    fn sound_before_the_first_word_is_cut_with_the_usual_padding() {
        // First word at 1.12 s, last ends at 1.84 s: keep 0.97 s .. 2.0 s.
        let out = trim_to_speech("chunk 2/2", ramp(), &reading_with_speech(1120, 1840));
        assert_eq!(out[0], (0.97f64 * 24_000.0).round() as f32);
        assert_eq!(*out.last().unwrap(), 47_999.0, "the tail pad runs past the end");
    }

    #[test]
    fn a_take_that_starts_on_its_word_is_left_alone() {
        // The clean shape: first word at 0.08 s, pad reaches back past zero,
        // and the tail pad reaches the end. Nothing to cut.
        let out = trim_to_speech("chunk 1/1", ramp(), &reading_with_speech(80, 1720));
        assert_eq!(out.len(), 48_000);
    }

    #[test]
    fn nothing_placeable_means_nothing_is_cut() {
        let reading = Reading {
            placeable: 0,
            stacked: 0,
            speech_ms: None,
            absorbed_lead: false,
            least_sure: None,
            dropped_tail: 0,
        };
        assert_eq!(trim_to_speech("chunk 1/1", ramp(), &reading).len(), 48_000);
    }

    #[test]
    fn an_accepted_take_is_trimmed_before_it_is_returned() {
        let mut gate = Scripted(vec![Some(reading_with_speech(1120, 1840))]);
        let pcm = take_until_accepted("chunk 2/2", "text", &mut gate, || Ok(ramp())).unwrap();
        assert!(pcm.len() < 48_000);
    }

    #[test]
    fn a_lone_zero_length_word_is_not_a_stack() {
        // Short words at 80 ms resolution: "a" and "of" can each land on one
        // instant in clean speech. Apart, they are not a collapse.
        let mut tokens = Vec::new();
        for i in 0..20 {
            let at = i * 300;
            let end = if i % 2 == 0 { at } else { at + 240 };
            tokens.push(span("word", at, end));
        }
        let r = read(&tokens);
        assert_eq!(r.stacked, 0);
        assert!(!r.is_collapse());
    }

    #[test]
    fn consecutive_zero_length_words_are_stacked() {
        let mut tokens = vec![span("one", 0, 240), span("two", 240, 480)];
        for _ in 0..4 {
            tokens.push(span("lost", 480, 480));
        }
        tokens.push(span("three", 480, 720));
        tokens.push(span("four", 720, 720));
        tokens.push(span("five", 720, 720));
        let r = read(&tokens);
        assert_eq!(
            r,
            Reading {
                placeable: 9,
                stacked: 6,
                speech_ms: Some((0, 720)),
                absorbed_lead: false,
                least_sure: None,
                dropped_tail: 0,
            }
        );
        assert!(r.is_collapse());
    }

    /// One measured take: what the aligner returned for real Qwen3 audio,
    /// and what speech recognition heard in it.
    #[derive(serde::Deserialize)]
    struct Take {
        name: String,
        /// `garbled` when recognition heard nonsense, `clean` otherwise.
        label: String,
        tokens: Vec<(String, i64, i64)>,
    }

    fn takes() -> Vec<Take> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/quality/takes.json");
        let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        serde_json::from_slice(&raw).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()))
    }

    fn spans(take: &Take) -> Vec<TokenSpan> {
        take.tokens
            .iter()
            .map(|(text, start_ms, end_ms)| span(text, *start_ms, *end_ms))
            .collect()
    }

    /// Every held take is judged the way recognition heard it: each garbled
    /// one rejected, each clean one — short titles, punctuation-heavy lines,
    /// Chinese, Japanese and the rest — accepted.
    #[test]
    fn held_takes_are_judged_as_recognition_heard_them() {
        let takes = takes();
        assert!(takes.iter().any(|t| t.label == "garbled"));
        assert!(takes.iter().any(|t| t.label == "clean"));
        for take in &takes {
            let reading = read(&spans(take));
            assert_eq!(
                reading.is_collapse(),
                take.label == "garbled",
                "{} ({}) read as {reading}",
                take.name,
                take.label
            );
        }
    }

    fn absorbed() -> Reading {
        Reading {
            absorbed_lead: true,
            ..reading_with_speech(160, 1840)
        }
    }

    #[test]
    fn an_absorbed_opening_is_retaken_and_the_clean_take_returned() {
        let mut calls = 0;
        let mut gate = Scripted(vec![Some(absorbed()), Some(GOOD)]);
        let pcm = take_until_accepted("chunk 1/2", "text", &mut gate, || {
            calls += 1;
            Ok(vec![calls as f32])
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(pcm, vec![2.0]);
    }

    #[test]
    fn an_absorbed_opening_never_fails_the_narration() {
        // Every take opens with a sound: the first is kept, because it is
        // still speech, and the bound still holds.
        let mut calls = 0;
        let mut gate = Scripted(vec![Some(absorbed()); MAX_TAKES + 5]);
        let pcm = take_until_accepted("chunk 1/1", "text", &mut gate, || {
            calls += 1;
            Ok(vec![calls as f32])
        })
        .unwrap();
        assert_eq!(calls, MAX_TAKES);
        assert_eq!(pcm, vec![1.0]);
    }

    #[test]
    fn an_understood_take_is_kept_over_failing_when_the_others_collapse() {
        let mut calls = 0;
        let mut gate = Scripted(vec![Some(BAD), Some(absorbed()), Some(BAD)]);
        let pcm = take_until_accepted("chunk 1/1", "text", &mut gate, || {
            calls += 1;
            Ok(vec![calls as f32])
        })
        .unwrap();
        assert_eq!(calls, MAX_TAKES);
        assert_eq!(pcm, vec![2.0]);
    }

    #[test]
    fn only_a_gap_after_the_first_word_is_checked_again() {
        let close = [span("A", 80, 160), span("skill", 240, 720)];
        assert_eq!(lead_cut_ms(&close), None);
        let apart = [span("A", 160, 640), span("skill", 1600, 2160)];
        assert_eq!(lead_cut_ms(&apart), Some(640));
        // Punctuation is not a word to measure the gap from.
        let quoted = [span("“", 0, 0), span("So", 80, 640), span("push", 1040, 1360)];
        assert_eq!(lead_cut_ms(&quoted), Some(640));
        assert_eq!(lead_cut_ms(&[span("Yes", 80, 880)]), None);
    }

    /// One held opening: the first alignment, the second from the cut (on the
    /// take's clock), and what speech recognition heard at the start.
    #[derive(serde::Deserialize)]
    struct Opening {
        name: String,
        /// `absorbed` when recognition heard a sound before the first word
        /// that the aligner had given that word; `clean` otherwise.
        label: String,
        tokens: Vec<(String, i64, i64)>,
        recut: Vec<(String, i64, i64)>,
    }

    #[test]
    fn held_openings_are_judged_as_recognition_heard_them() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/quality/leads.json");
        let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        let openings: Vec<Opening> =
            serde_json::from_slice(&raw).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()));
        assert!(openings.iter().any(|o| o.label == "absorbed"));
        assert!(openings.iter().any(|o| o.label == "clean"));
        let to_spans = |t: &[(String, i64, i64)]| -> Vec<TokenSpan> {
            t.iter().map(|(w, s, e)| span(w, *s, *e)).collect()
        };
        for o in &openings {
            let tokens = to_spans(&o.tokens);
            assert!(lead_cut_ms(&tokens).is_some(), "{} has no gap to check", o.name);
            assert_eq!(
                lead_absorbed(&tokens, &to_spans(&o.recut)),
                o.label == "absorbed",
                "{} ({})",
                o.name,
                o.label
            );
        }
    }

    fn sure_spans(sure: &[f32]) -> Vec<TokenSpan> {
        sure.iter()
            .enumerate()
            .map(|(i, &v)| TokenSpan {
                sure: Some(v),
                ..span("word", i as i64 * 240, i as i64 * 240 + 160)
            })
            .collect()
    }

    #[test]
    fn the_least_sure_ten_words_decide() {
        // Twenty sure words with one unsure stretch of four: the window that
        // holds all four averages (6 × 0.9 + 4 × 0.05) / 10 = 0.56.
        let mut sure = vec![0.9; 20];
        sure[8..12].fill(0.05);
        let r = read(&sure_spans(&sure));
        assert!((r.least_sure.unwrap() - 0.56).abs() < 1e-6, "{r:?}");
        assert!(!r.is_unsure(), "one hard word does not fail a clean stretch");
        // Babble across seven of ten words is a stretch the aligner lost.
        sure[6..13].fill(0.05);
        assert!(read(&sure_spans(&sure)).is_unsure());
    }

    #[test]
    fn a_short_take_is_averaged_whole() {
        let r = read(&sure_spans(&[0.3; MIN_TOKENS]));
        assert!((r.least_sure.unwrap() - 0.3).abs() < 1e-6);
        assert!(r.is_unsure());
    }

    #[test]
    fn sureness_is_not_judged_on_too_few_words_or_none_measured() {
        assert_eq!(read(&sure_spans(&[0.1; MIN_TOKENS - 1])).least_sure, None);
        // Spans with no sureness (the held fixtures) are not judged unsure.
        let unmeasured: Vec<TokenSpan> = (0..MIN_TOKENS as i64).map(|i| span("a", i * 80, i * 80 + 80)).collect();
        assert_eq!(read(&unmeasured).least_sure, None);
    }

    #[test]
    fn korean_is_not_judged_for_sureness() {
        let mut tokens = sure_spans(&[0.1; MIN_TOKENS]);
        tokens[2].text = "스킬은".into();
        assert_eq!(read(&tokens).least_sure, None);
    }

    #[test]
    fn punctuation_does_not_count_toward_sureness() {
        // The model places "—" badly by design; it is not a word to be sure of.
        let mut tokens = sure_spans(&[0.9; MIN_TOKENS]);
        tokens.insert(3, TokenSpan { sure: Some(0.0), ..span("—", 720, 720) });
        assert!((read(&tokens).least_sure.unwrap() - 0.9).abs() < 1e-6);
    }

    fn unsure(least_sure: f64) -> Reading {
        Reading {
            least_sure: Some(least_sure),
            ..reading_with_speech(80, 1840)
        }
    }

    #[test]
    fn an_unsure_take_is_retaken_and_the_clean_one_returned() {
        let mut calls = 0;
        let mut gate = Scripted(vec![Some(unsure(0.25)), Some(GOOD)]);
        let pcm = take_until_accepted("chunk 1/1", "text", &mut gate, || {
            calls += 1;
            Ok(vec![calls as f32])
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(pcm, vec![2.0]);
    }

    #[test]
    fn when_every_take_is_flawed_the_surest_is_kept() {
        let mut calls = 0;
        let mut gate = Scripted(vec![Some(unsure(0.30)), Some(unsure(0.41)), Some(absorbed())]);
        let pcm = take_until_accepted("chunk 1/1", "text", &mut gate, || {
            calls += 1;
            Ok(vec![calls as f32])
        })
        .unwrap();
        assert_eq!(calls, MAX_TAKES, "the bound holds");
        // The absorbed opening measured no sureness problem, so it is surest.
        assert_eq!(pcm, vec![3.0]);
    }

    #[test]
    fn an_unsure_take_is_kept_over_failing_when_the_others_collapse() {
        let mut gate = Scripted(vec![Some(BAD), Some(unsure(0.2)), Some(BAD)]);
        let mut calls = 0;
        let pcm = take_until_accepted("chunk 1/1", "text", &mut gate, || {
            calls += 1;
            Ok(vec![calls as f32])
        })
        .unwrap();
        assert_eq!(pcm, vec![2.0]);
    }

    /// One measured take: the aligner's sureness of each word, what speech
    /// recognition heard, and where the rule is known to disagree with it.
    #[derive(serde::Deserialize)]
    struct Sureness {
        name: String,
        /// `measured` takes set the line; `held-out` ones were made after.
        set: String,
        /// `clean`, or what recognition heard instead: `garbled`, `babble`,
        /// `inserted`, `absorbed`, or `laugh-before` (a laugh before the first
        /// word, which the trim cuts when the first word is placed after it).
        label: String,
        /// `missed` or `false alarm` where the rule and recognition disagree.
        known: Option<String>,
        words: Vec<(String, f32)>,
    }

    /// Every measured and held-out take is judged as recognition heard it,
    /// except the disagreements the fixture names — and those stay few: no
    /// measured clean take is flagged, and one held-out clean take in 100.
    ///
    /// The `measured` slide narrations (`bv0127_*`, `bv0128_*`, `real-*`) were
    /// aligned whole, as published; the gate aligns each chunk, which is how
    /// every `held-out` take and every `ab-*` and `drift-*` take was aligned.
    #[test]
    fn measured_takes_are_judged_as_recognition_heard_them() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/quality/sureness.json");
        let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        let takes: Vec<Sureness> =
            serde_json::from_slice(&raw).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()));
        let mut disagreements = Vec::new();
        for take in &takes {
            let tokens: Vec<TokenSpan> = take
                .words
                .iter()
                .enumerate()
                .map(|(i, (w, sure))| TokenSpan {
                    sure: Some(*sure),
                    ..span(w, i as i64 * 240, i as i64 * 240 + 160)
                })
                .collect();
            let flagged = read(&tokens).is_unsure();
            let heard_defect = take.label != "clean";
            let expected = match take.known.as_deref() {
                Some("missed") => false,
                Some("false alarm") => true,
                _ => heard_defect,
            };
            assert_eq!(flagged, expected, "{} ({}, {}) ", take.name, take.label, take.set);
            if flagged != heard_defect {
                disagreements.push((take.set.as_str(), take.label.as_str()));
            }
        }
        let clean = |set: &str| {
            takes.iter().filter(|t| t.set == set && t.label == "clean").count()
        };
        let false_alarms = |set: &str| {
            disagreements.iter().filter(|(s, l)| *s == set && *l == "clean").count()
        };
        assert_eq!(false_alarms("measured"), 0, "of {}", clean("measured"));
        assert!(false_alarms("held-out") <= 1, "of {}", clean("held-out"));
    }

    #[test]
    fn words_placed_where_the_audio_ends_were_never_said() {
        // The 0.1.29 German take: "liegen zu lassen" piled onto the take's
        // last 80 ms of 9.44 s.
        let german = [
            span("sie", 8880, 9360),
            span("liegen", 9360, 9440),
            span("zu", 9440, 9440),
            span("lassen", 9440, 9440),
        ];
        assert_eq!(dropped_tail(&german, 9440), 3);
        // Placed past the end, with a length the audio does not have.
        let korean = [
            span("검사했고", 7120, 7920),
            span("그중", 7920, 8000),
            span("발견했습니다", 8960, 11040),
        ];
        assert_eq!(dropped_tail(&korean, 8000), 2);
    }

    #[test]
    fn a_take_that_ends_on_its_last_word_dropped_nothing() {
        let tokens = [
            span("within", 10560, 10880),
            span("a", 10880, 10960),
            span("minute", 10960, 11280),
        ];
        assert_eq!(dropped_tail(&tokens, 11280), 0);
        // One short last word in the final step is a clean ending (Japanese
        // "す", 80 ms), and so is one given no time there.
        let japanese = [span("ま", 8400, 8560), span("す", 8560, 8640)];
        assert_eq!(dropped_tail(&japanese, 8640), 0);
        let code = [span("skill", 32880, 33440), span("code", 33440, 33440)];
        assert_eq!(dropped_tail(&code, 33432), 0);
        // Punctuation after the last word is not a word to place.
        let quoted = [
            span("ship", 400, 880),
            span("”", 880, 880),
            span(".", 880, 880),
        ];
        assert_eq!(dropped_tail(&quoted, 880), 0);
        assert_eq!(dropped_tail(&[], 880), 0);
    }

    fn dropped(words: usize) -> Reading {
        Reading {
            dropped_tail: words,
            ..reading_with_speech(80, 1840)
        }
    }

    #[test]
    fn a_dropped_ending_is_retaken_and_the_whole_take_returned() {
        let mut calls = 0;
        let mut gate = Scripted(vec![Some(dropped(3)), Some(GOOD)]);
        let pcm = take_until_accepted("chunk 1/1", "text", &mut gate, || {
            calls += 1;
            Ok(vec![calls as f32])
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(pcm, vec![2.0]);
    }

    /// One measured ending: the last words the aligner placed, the take's
    /// length, and whether recognition heard the take reach its last word.
    #[derive(serde::Deserialize)]
    struct Ending {
        name: String,
        /// `measured` takes set the rule; `held-out` ones were made after.
        set: String,
        /// `whole`, or `dropped` when recognition heard the take stop early.
        label: String,
        /// For a `dropped` take, how many words recognition did not hear.
        unsaid: Option<usize>,
        take_ms: i64,
        tail: Vec<(String, i64, i64)>,
    }

    /// Every measured and held-out ending is judged as recognition heard it:
    /// each take that stopped early is caught, with the words it did not say
    /// counted, and no whole one is.
    #[test]
    fn measured_endings_are_judged_as_recognition_heard_them() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/quality/tails.json");
        let raw =
            std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        let endings: Vec<Ending> = serde_json::from_slice(&raw)
            .unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()));
        assert!(endings.iter().any(|e| e.label == "dropped"));
        assert!(endings.iter().filter(|e| e.label == "whole").count() > 600);
        for e in &endings {
            let tokens: Vec<TokenSpan> = e.tail.iter().map(|(w, s, t)| span(w, *s, *t)).collect();
            let dropped = dropped_tail(&tokens, e.take_ms);
            assert_eq!(
                dropped > 0,
                e.label == "dropped",
                "{} ({}, {})",
                e.name,
                e.label,
                e.set
            );
            if let Some(unsaid) = e.unsaid {
                assert_eq!(
                    dropped, unsaid,
                    "{}: the words counted are the ones not said",
                    e.name
                );
            }
        }
    }

    /// Which of three flawed takes `take_until_accepted` keeps, by number.
    fn kept(readings: [Reading; MAX_TAKES]) -> f32 {
        let mut calls = 0;
        let mut gate = Scripted(readings.map(Some).to_vec());
        let pcm = take_until_accepted("chunk 1/1", "text", &mut gate, || {
            calls += 1;
            Ok(vec![calls as f32])
        })
        .unwrap();
        pcm[0]
    }

    #[test]
    fn a_take_with_all_its_words_is_kept_over_one_missing_some() {
        assert_eq!(kept([dropped(3), unsure(0.2), dropped(2)]), 2.0);
    }

    #[test]
    fn of_takes_missing_words_the_one_missing_fewest_is_kept() {
        assert_eq!(kept([dropped(3), dropped(2), dropped(4)]), 2.0);
    }

    #[test]
    fn too_few_words_are_never_judged_collapsed() {
        // "Yes. No." — both words stacked at zero length is still two words.
        let tokens = [span("Yes", 0, 0), span("No", 0, 0)];
        let r = read(&tokens);
        assert_eq!(r.stacked, 2);
        assert!(!r.is_collapse());
    }
}
