//! Static resource gate for embedded scripts.
//!
//! Answers, before any parser or interpreter runs: *is this script small and
//! flat enough to even hand to the language?* Byte cap, bracket depth,
//! unary-prefix runs, and conditional count — with strings and comments
//! skipped so that `"(("` inside a literal does not count as nesting.
//!
//! The gate errs on the safe side by design: brackets inside regular
//! expressions and template literals count toward depth, which can refuse a
//! script that a full parser would accept. That trade is correct for a gate
//! that must be cheaper than the parser it protects.
//!
//! Language-agnostic by construction: it reads bytes, knows the C-family
//! bracket and comment conventions (JS, Rust, FormCalc's `{}` blocks), and
//! reports violations as reasons, not panics. Hosts configure the limits to
//! their language and embed via [`ScriptGate::inspect`].
//!
//! # Example
//!
//! ```
//! use script_gate_kit::{ScriptGate, Violation};
//!
//! let gate = ScriptGate::js_defaults();
//! assert!(gate.inspect("if (x) { doThing(); }").is_none());
//!
//! let hostile = format!("{}{{", "{(".repeat(200));
//! let violation = gate.inspect(&hostile).expect("deep nesting refused");
//! assert!(matches!(violation, Violation::TooDeep));
//! ```

/// Why a script was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Violation {
    /// The source exceeds the byte cap.
    TooLong {
        /// Actual size in bytes.
        actual: usize,
        /// The configured cap in bytes.
        limit: usize,
    },
    /// Brackets nest deeper than the cap.
    TooDeep,
    /// A run of unary prefix operators exceeds the cap (a sign of an
    /// obfuscation ladder like `!!!!----x`).
    PrefixRun,
    /// More conditional operators than the cap.
    TooManyConditionals,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLong { actual, limit } => write!(
                f,
                "the script is too long ({} KiB; the limit is {} KiB)",
                actual / 1024,
                limit / 1024
            ),
            Self::TooDeep => f.write_str("the script is nested too deeply to run safely"),
            Self::PrefixRun => {
                f.write_str("the script contains too long a run of prefix operators")
            }
            Self::TooManyConditionals => {
                f.write_str("the script contains too many conditional operators")
            }
        }
    }
}

/// The configured limits. Defaults match the values a hostile-script fuzzer
/// converged on for a form-scripting host (256 KiB source, depth 64, prefix
/// run 64, 4096 conditionals).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScriptGate {
    max_bytes: usize,
    max_depth: usize,
    max_prefix_run: usize,
    max_conditionals: usize,
}

impl Default for ScriptGate {
    fn default() -> Self {
        Self::js_defaults()
    }
}

impl ScriptGate {
    /// The defaults a hostile-script fuzzer converged on for JS-like hosts.
    #[must_use]
    pub const fn js_defaults() -> Self {
        Self {
            max_bytes: 256 * 1024,
            max_depth: 64,
            max_prefix_run: 64,
            max_conditionals: 4096,
        }
    }

    /// Explicit limits (embeddings with smaller budgets).
    #[must_use]
    pub const fn with_limits(
        max_bytes: usize,
        max_depth: usize,
        max_prefix_run: usize,
        max_conditionals: usize,
    ) -> Self {
        Self {
            max_bytes,
            max_depth,
            max_prefix_run,
            max_conditionals,
        }
    }

    /// The byte cap.
    #[must_use]
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Inspects a script. `None` means the script passes the gate and may be
    /// handed to the language; [`Some`] carries the reason it was refused.
    ///
    /// Strings (single- and double-quoted, with backslash escapes and
    /// newline termination), line comments, and block comments are skipped.
    /// Brackets in regexes and template literals count toward depth — the
    /// safe-side trade noted in the module docs.
    #[must_use]
    pub fn inspect(&self, source: &str) -> Option<Violation> {
        if source.len() > self.max_bytes {
            return Some(Violation::TooLong {
                actual: source.len(),
                limit: self.max_bytes,
            });
        }
        let b = source.as_bytes();
        let (mut depth, mut prefix, mut conditionals) = (0usize, 0usize, 0usize);
        let mut i = 0;
        while let Some(&c) = b.get(i) {
            i += 1;
            match c {
                b'\'' | b'"' => {
                    // To the closing quote (or the end of the line, where an
                    // unclosed string ends).
                    while let Some(&d) = b.get(i) {
                        i += 1;
                        if d == b'\\' {
                            i += 1;
                        } else if d == c || d == b'\n' {
                            break;
                        }
                    }
                    prefix = 0;
                }
                b'/' if b.get(i) == Some(&b'/') => {
                    while b.get(i).is_some_and(|d| *d != b'\n') {
                        i += 1;
                    }
                }
                b'/' if b.get(i) == Some(&b'*') => {
                    i += 1;
                    while b.get(i..).is_some_and(|rest| !rest.starts_with(b"*/")) {
                        i += 1;
                    }
                    i += 2;
                }
                b'(' | b'[' | b'{' => {
                    depth += 1;
                    if depth > self.max_depth {
                        return Some(Violation::TooDeep);
                    }
                    prefix = 0;
                }
                b')' | b']' | b'}' => {
                    depth = depth.saturating_sub(1);
                    prefix = 0;
                }
                b'!' | b'~' | b'+' | b'-' => {
                    prefix += 1;
                    if prefix > self.max_prefix_run {
                        return Some(Violation::PrefixRun);
                    }
                }
                b'?' => {
                    conditionals += 1;
                    if conditionals > self.max_conditionals {
                        return Some(Violation::TooManyConditionals);
                    }
                    prefix = 0;
                }
                c if c.is_ascii_whitespace() => {}
                _ => prefix = 0,
            }
        }
        None
    }
}

/// Convenience: inspect against the JS defaults.
#[must_use]
pub fn inspect_js(source: &str) -> Option<Violation> {
    ScriptGate::js_defaults().inspect(source)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
    use super::*;

    #[test]
    fn ordinary_scripts_pass() {
        let gate = ScriptGate::js_defaults();
        assert_eq!(gate.inspect("if (x) { doThing(); }"), None);
        assert_eq!(gate.inspect("var s = \"(((\" ; // not nesting"), None);
        assert_eq!(gate.inspect("/* (( )) */ 1 + 1"), None);
        assert_eq!(gate.inspect(""), None);
    }

    #[test]
    fn oversized_scripts_report_both_numbers() {
        let gate = ScriptGate::with_limits(64, 8, 8, 64);
        match gate.inspect(&"x".repeat(65)) {
            Some(Violation::TooLong { actual, limit }) => {
                assert_eq!(actual, 65);
                assert_eq!(limit, 64);
            }
            other => panic!("expected TooLong, got {other:?}"),
        }
        let msg = gate.inspect(&"x".repeat(65)).unwrap().to_string();
        assert!(msg.contains("too long"), "{msg}");
    }

    #[test]
    fn deep_bracket_nesting_is_refused_even_in_strings_adjacent() -> Result<(), Violation> {
        let gate = ScriptGate::with_limits(1 << 20, 8, 8, 4096);
        let deep = "{".repeat(9);
        assert_eq!(gate.inspect(&deep), Some(Violation::TooDeep));
        let ok = "{".repeat(8);
        assert_eq!(gate.inspect(&ok), None);
        Ok(())
    }

    #[test]
    fn strings_do_not_count_toward_depth() {
        let gate = ScriptGate::js_defaults();
        // 300 open-brackets inside a string literal: skipped, not nested
        let s = format!("var t = \"{}\";", "{".repeat(300));
        assert_eq!(gate.inspect(&s), None);
    }

    #[test]
    fn comments_do_not_count_toward_depth() {
        let gate = ScriptGate::js_defaults();
        let line = format!("// {}\n1", "(".repeat(300));
        assert_eq!(gate.inspect(&line), None);
        let block = format!("/* {} */ 1", "(".repeat(300));
        assert_eq!(gate.inspect(&block), None);
        // an unclosed block comment swallows the rest of the script
        let unclosed = format!("/* {}", "(".repeat(300));
        assert_eq!(gate.inspect(&unclosed), None);
    }

    #[test]
    fn prefix_runs_are_refused() {
        let gate = ScriptGate::with_limits(1 << 20, 64, 8, 4096);
        let ladder = format!("{}x", "!".repeat(9));
        assert_eq!(gate.inspect(&ladder), Some(Violation::PrefixRun));
        let ok = format!("{}x", "!".repeat(8));
        assert_eq!(gate.inspect(&ok), None);
    }

    #[test]
    fn conditional_storms_are_refused() {
        let gate = ScriptGate::with_limits(1 << 20, 64, 8, 8);
        let storm = format!("a ?{} b : c", " ?".repeat(12));
        assert_eq!(gate.inspect(&storm), Some(Violation::TooManyConditionals));
    }

    #[test]
    fn close_brackets_never_underflow_depth() {
        let gate = ScriptGate::js_defaults();
        assert_eq!(gate.inspect("))))"), None, "saturating, not panicking");
    }

    #[test]
    fn js_default_convenience_matches_the_struct() {
        assert_eq!(inspect_js("ok"), None);
        assert_eq!(inspect_js(&"{".repeat(200)), Some(Violation::TooDeep));
    }

    #[test]
    fn violation_display_is_human_readable() {
        let gate = ScriptGate::js_defaults();
        assert!(
            gate.inspect(&"x".repeat(300 * 1024))
                .unwrap()
                .to_string()
                .contains("KiB")
        );
        assert_eq!(
            ScriptGate::js_defaults()
                .inspect(&"{".repeat(200))
                .unwrap()
                .to_string(),
            "the script is nested too deeply to run safely"
        );
    }
}
