// SPDX-License-Identifier: MPL-2.0
//
// Part of Auguth Labs open-source softwares.
// Built for the Rust Programming Language Ecosystem.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Auguth Labs (OPC) Pvt Ltd, India

// ===============================================================================
// ````````````````````````````` KEY-VALUE ARGUMENTS `````````````````````````````
// ===============================================================================

//! Schema-driven parsing and validation of structured macro key arguments.
//!
//! This module provides the building blocks for declaring, parsing, validating,
//! and semantically decoding key-based macro arguments.
//!
//! It separates the problem into four distinct layers:
//!
//! - **Parsing**: [`KeyInfo`], [`KeyValue`], and [`KeyList`] parse raw macro
//!   input into a syntax tree while producing structured diagnostics.
//! - **Schema definition**: [`KeySpec`] and [`crate::key_schema!`] describe the
//!   complete grammar of a key, including its allowed sub-keys and valid
//!   combinations, and generate strongly-typed decoding logic.
//! - **Validation**: [`KeySet`] validates nested key groups by rejecting
//!   unexpected and duplicate sub-keys.
//! - **Value decoding**: [`KeyExpect`] defines how individual sub-key values
//!   are extracted and type-checked.
//!
//! Together these components allow procedural macros to describe argument
//! grammars declaratively while receiving:
//!
//! - strongly typed semantic representations,
//! - deterministic validation,
//! - exact sub-key combination matching, and
//! - rich, schema-aware compiler diagnostics.
//!
//! ## Conceptual model
//!
//! A parsed key has one of the following forms:
//!
//! ```text
//! key
//! key(...)
//! key { ... }
//! key [ ... ]
//! ```
//!
//! Nested forms contain a [`KeyList`], whose elements are validated against a
//! schema before being decoded into a strongly typed specification.
//!
//! ## Defining schemas
//!
//! Most users interact only with [`crate::key_schema!`], which declaratively specifies:
//!
//! - the literal key name,
//! - whether the key may appear without a value,
//! - the allowed sub-keys and their value types,
//! - the exact combinations of sub-keys that are accepted.
//!
//! From this declaration the macro generates:
//!
//! - a specification enum,
//! - a complete [`KeySpec`] implementation,
//! - deterministic decoding logic, and
//! - comprehensive diagnostics.
//!
//! The lower-level helper macros (`__key_spec!`, `__emit_*`, etc.) are internal
//! implementation details and are not intended to be invoked directly.

// ===============================================================================
// ``````````````````````````````````` IMPORTS ```````````````````````````````````
// ===============================================================================

// --- Local crate ---
use crate::{DiagSpan, delim::Delimited, errors::*, lists::*, misc::LINE_SPACE, space::*};

// --- Std Lib ---
use std::{collections::HashMap, marker::PhantomData};

// --- Syn/Quote ---
use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, TokenStreamExt, format_ident};
use syn::{
    Ident,
    parse::{Parse, ParseStream, discouraged::Speculative},
    punctuated::Punctuated,
    token::{Brace, Bracket, Comma, Paren},
};

// ===============================================================================
// ````````````````````````````````` STRUCTURES ``````````````````````````````````
// ===============================================================================

/// Represents a parsed key and its associated value.
///
/// Each entry consists of:
/// - a key [`Ident`]
/// - an optional [`KeyValue`] describing its value
///
/// ## Example
/// ```ignore
/// key(Foo, Bar)
/// ```
///
/// In this case, `key` is the identifier and `value` is a [`IdentList`].
///
/// The generic parameter `T` controls parsing behavior:
/// - [`KeyInfo<()>`] (or simply [`KeyInfo`]): accepts any supported value form
/// - [`KeyInfo<T>`]: restricts parsing to a specific representation
///   (e.g. `KeyInfo<IdentList>`, `KeyInfo<KeyList>`)
///
/// ## Note
/// While parsing raw-arguments,
/// - `()`: supports both value lists and nested key-lists
/// - `{}` / `[]`: nested key-lists only (no simple value lists)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyInfo<T = ()> {
    /// The identifier representing the key.
    pub key: Ident,

    /// The associated value for the key (may be [`KeyValue::None`]).
    pub value: KeyValue<T>,
}

/// Represents the value associated with a parsed delimited key.
///
/// A key may have:
/// - no value (`None`)
/// - a list of expressions, or other low level lists in [`crate::lists`]
/// - a nested key list
///
/// This enum is `#[non_exhaustive]` and may grow with additional
/// value representations in the future.
///
/// The generic parameter `T` controls parsing behavior:
/// - [`KeyValue<()>`] (or simply [`KeyValue`]): accepts any supported value form
/// - [`KeyValue<T>`]: restricts parsing to a specific representation
///   (e.g. `KeyValue<IdentList>`, `KeyValue<KeyList>`)
///
/// ## Note
/// While parsing raw-arguments,
/// - `()`: supports both value lists and nested key-lists
/// - `{}` / `[]`: nested key-lists only (no simple value lists)
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum KeyValue<T = ()> {
    /// No value associated with the key (bare key).
    ///
    /// Example: `key`
    None,

    /// A comma-separated list of expressions.
    ///
    /// Example: `key(variable, VAR, foo(), b"get")`
    Exprs(ExprList),

    /// A nested key list (hierarchical structure).
    ///
    /// Example: `key{subkey, subkey2(..)}`
    Nested(KeyList),

    Group(ValueGroup),

    /// Marker used to enforce a specific value representation via `T`.
    ///
    /// This is not constructed directly and exists purely for type-level control.
    _Marker(PhantomData<T>),
}

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct ValueGroup<T = ()> {
    pub values: Punctuated<KeyValue<T>, Comma>,
}

/// Represents a comma-separated list of parsed keys.
///
/// Each element is a [`KeyInfo`] containing a key and its associated value.
/// This is the top-level structure for key-value argument parsing and
/// supports nesting via [`KeyValue::Nested`].
///
/// ## Example
/// ```ignore
/// key1, key2(Foo), key3{subkey, subkey2(..)}
/// ```
///
/// In this case, `keys` would contain multiple [`KeyInfo`] entries.
///
/// For stricter parsing where a key must contain a nested key-list,
/// use [`KeyInfo<KeyList>`].
///
/// ## Note
/// While parsing raw-arguments,
/// - `()`: supports both value lists and nested key-lists
/// - `{}` / `[]`: nested key-lists only (no simple value lists)
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct KeyList {
    /// The ordered list of parsed key-value entries.
    pub keys: Punctuated<KeyInfo, Comma>,
}

// ===============================================================================
// ``````````````````````````````` TO-TOKENS IMPL ````````````````````````````````
// ===============================================================================

impl<T> ToTokens for ValueGroup<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(&self.values);
    }
}

impl<T> ToTokens for KeyValue<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::None => {}

            Self::Exprs(exprs) => {
                syn::token::Paren::default().surround(tokens, |tokens| {
                    exprs.to_tokens(tokens);
                });
            }

            Self::Nested(keys) => {
                syn::token::Brace::default().surround(tokens, |tokens| {
                    keys.to_tokens(tokens);
                });
            }

            Self::Group(group) => {
                syn::token::Brace::default().surround(tokens, |tokens| {
                    group.to_tokens(tokens);
                });
            }

            Self::_Marker(_) => {}
        }
    }
}

impl<T> ToTokens for KeyInfo<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.key.to_tokens(tokens);
        self.value.to_tokens(tokens);
    }
}

impl ToTokens for KeyList {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(&self.keys);
    }
}

// ===============================================================================
// `````````````````````````````````` SYN-PARSE ``````````````````````````````````
// ===============================================================================

impl<T> Parse for KeyInfo<T>
where
    KeyValue<T>: Parse,
{
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let span = input.span();
        let key = match input.parse::<Ident>() {
            Ok(v) => v,
            Err(_) => {
                return Err(<Ident as ParseDiagnostic>::parse_diagnostic(
                    &DiagSpan::Span(span),
                    None,
                )
                .into());
            }
        };
        let value = input.parse::<KeyValue<T>>()?;
        Ok(KeyInfo { key, value })
    }
}

impl Parse for KeyValue {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if !(input.peek(Paren) || input.peek(Brace) || input.peek(Bracket)) {
            return Ok(KeyValue::None);
        }

        let delimited = Delimited::<()>::parse(input)?;

        match delimited {
            Delimited::Paren(content) => {
                // exprs: key(b"foo", foo(), FOO)
                let fork = content.fork();
                let (v_expr, e_expr) = match fork.parse::<ExprList>() {
                    Ok(v) => (Some(v), None),
                    Err(e) => (None, Some(e)),
                };
                if let Some(v) = v_expr {
                    content.advance_to(&fork);
                    return Ok(KeyValue::Exprs(v));
                };

                // Nested keys: key(a, b(c))
                let fork = content.fork();
                let (v_kl, e_kl) = match fork.parse::<KeyList>() {
                    Ok(v) => (Some(v), None),
                    Err(e) => (None, Some(e)),
                };
                if let Some(v) = v_kl {
                    content.advance_to(&fork);
                    return Ok(KeyValue::Nested(v));
                };

                let mut err = e_expr.unwrap();
                err.combine(e_kl.unwrap());

                Err(err)
            }

            // `{}` and `[]` are reserved exclusively for nested keys or value groups.
            Delimited::Brace(content) | Delimited::Bracket(content) => {
                // Nested keys.
                let fork = content.fork();
                let (v_kl, e_kl) = match fork.parse::<KeyList>() {
                    Ok(v) => (Some(v), None),
                    Err(e) => (None, Some(e)),
                };
                if let Some(v) = v_kl {
                    content.advance_to(&fork);
                    return Ok(KeyValue::Nested(v));
                }

                // Value groups.
                let fork = content.fork();
                let (v_group, e_group) = match fork.parse::<ValueGroup>() {
                    Ok(v) => (Some(v), None),
                    Err(e) => (None, Some(e)),
                };
                if let Some(v) = v_group {
                    content.advance_to(&fork);
                    return Ok(KeyValue::Group(v));
                }

                let mut err = e_kl.clone().unwrap();
                err.combine(e_group.unwrap());
                Err(err)
            }

            Delimited::_Marker(_) => unreachable!(),
        }
    }
}

impl Parse for KeyValue<ExprList> {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if !(input.peek(Paren) || input.peek(Brace) || input.peek(Bracket)) {
            return Ok(KeyValue::None);
        }

        let content = Delimited::<Paren>::parse(input)?;
        let fork = content.fork();
        match fork.parse::<ExprList>() {
            Ok(v) => {
                content.advance_to(&fork);
                return Ok(KeyValue::Exprs(v));
            }
            Err(mut err) => {
                err.combine(
                    <KeyValue<ExprList> as ParseDiagnostic>::parse_diagnostic(
                        &err.clone().into(),
                        None,
                    )
                    .into(),
                );
                Err(err)
            }
        }
    }
}

impl Parse for KeyValue<KeyList> {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if !(input.peek(Paren) || input.peek(Brace) || input.peek(Bracket)) {
            return Ok(KeyValue::None);
        }
        let pre_span = input.span();

        let delimited = Delimited::<()>::parse(input)?;
        match delimited {
            // `()` is reserved exclusively for non-nested value/s.
            // Nested key-lists are not allowed in this delimiter.
            Delimited::Paren(_) => {
                return Err(<KeyValue<KeyList> as ParseDiagnostic>::parse_diagnostic(
                    &DiagSpan::Span(pre_span),
                    None,
                )
                .into());
            }
            Delimited::Brace(content) | Delimited::Bracket(content) => {
                let v = content.parse::<KeyList>()?;
                return Ok(KeyValue::Nested(v));
            }
            Delimited::_Marker(_) => unreachable!(),
        }
    }
}

impl Parse for KeyValue<ValueGroup> {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if !(input.peek(Paren) || input.peek(Brace) || input.peek(Bracket)) {
            return Ok(KeyValue::None);
        }
        let pre_span = input.span();

        let delimited = Delimited::<()>::parse(input)?;
        match delimited {
            // `()` is reserved exclusively for non-nested value/s.
            // Nested value-groups are not allowed in this delimiter.
            Delimited::Paren(_) => {
                Err(<KeyValue<ValueGroup> as ParseDiagnostic>::parse_diagnostic(
                    &DiagSpan::Span(pre_span),
                    None,
                )
                .into())
            }
            Delimited::Brace(content) | Delimited::Bracket(content) => {
                let v = content.parse::<ValueGroup<()>>()?;
                return Ok(KeyValue::Group(v));
            }
            Delimited::_Marker(_) => unreachable!(),
        }
    }
}

impl<T> Parse for ValueGroup<T>
where
    KeyValue<T>: Parse,
    ValueGroup<T>: ParseDiagnostic,
{
    fn parse(input: ParseStream) -> syn::parse::Result<Self> {
        match Punctuated::<KeyValue<T>, Comma>::parse_terminated(input) {
            Ok(v) => Ok(ValueGroup { values: v }),
            Err(mut e) => {
                e.combine(
                    <ValueGroup<T> as ParseDiagnostic>::parse_diagnostic(&e.clone().into(), None)
                        .into(),
                );
                return Err(e);
            }
        }
    }
}

impl Parse for KeyList {
    fn parse(input: ParseStream) -> syn::parse::Result<Self> {
        match Punctuated::<KeyInfo, Comma>::parse_terminated(input) {
            Ok(v) => Ok(KeyList { keys: v }),
            Err(mut e) => {
                e.combine(
                    <KeyList as ParseDiagnostic>::parse_diagnostic(&e.clone().into(), None).into(),
                );
                return Err(e);
            }
        }
    }
}

// ===============================================================================
// `````````````````````````````` PARSE-DIAGNOSTICS ``````````````````````````````
// ===============================================================================

impl ParseDiagnostic for KeyValue {
    const KIND: &'static str = "key's value wrapped in a delimiter";
    const EXPECTED: &'static str =
        "either a bare key or a key with value list wrapped in one of `()`, `{}`, or `[]`";
    const EXAMPLE: &[&'static str] = &[
        "key(<value>, ..)",
        "key[<value>, ..]",
        "key{<value>, ..}",
        "key (no-value key-value)",
    ];
    const DEFAULT_ERROR_INFO: ErrorInfo = KeyParseError::Default.to_error_info();
}

impl ParseDiagnostic for KeyValue<ExprList> {
    const KIND: &'static str = "key-value expression list wrapped in parenthesis `()`";

    const EXPECTED: &'static str = "parenthesis delimited comma-punctuated list of expressions";

    const EXAMPLE: &[&'static str] = &[
        "key(foo, bar, b\"bytes\", make_bytes())",
        "key (no-value key-value)",
    ];

    const DEFAULT_ERROR_INFO: ErrorInfo = KeyParseError::ExprList.to_error_info();
}

impl ParseDiagnostic for KeyList {
    const KIND: &'static str = "comma-seperated keys list";
    const EXPECTED: &'static str =
        "keys list with none or valid values wrapped in either `()`, `{}`, `[]`";
    const EXAMPLE: &[&'static str] = &["\"subkey0, subkey1(..), subkey2{..}, subkey3[..]\""];
    const DEFAULT_ERROR_INFO: ErrorInfo = ParseError::KeyList.to_error_info();
}

impl ParseDiagnostic for KeyValue<KeyList> {
    const KIND: &'static str =
        "key-value as nested list wrapped in either braces `{}` or brackets `{}`";
    const EXPECTED: &'static str = "braces or brackets delimited nested key-list";
    const EXAMPLE: &[&'static str] = &["key{subkey, subkey2(..)}", "key[subkey, subkey2(..)]"];
    const DEFAULT_ERROR_INFO: ErrorInfo = KeyParseError::KeyList.to_error_info();
}

impl<T> ParseDiagnostic for ValueGroup<T> {
    const KIND: &'static str = "comma-separated value groups";
    const EXPECTED: &'static str = "comma-separated parenthesized value groups";
    const EXAMPLE: &[&'static str] = &["\"(<value>, <value>), (<value>, <value>)\""];
    const DEFAULT_ERROR_INFO: ErrorInfo = ParseError::ValueGroup.to_error_info();
}

impl ParseDiagnostic for KeyValue<ValueGroup> {
    const KIND: &'static str =
        "key-value as grouped values wrapped in either braces `{}` or brackets `[]`";
    const EXPECTED: &'static str =
        "braces or brackets delimited comma-separated parenthesized value groups";
    const EXAMPLE: &[&'static str] = &[
        "key{(<value>, <value>), (<value>, <value>)}",
        "key[(<value>, <value>), (<value>,)]",
    ];
    const DEFAULT_ERROR_INFO: ErrorInfo = KeyParseError::ValueGroup.to_error_info();
}

// ===============================================================================
// ``````````````````````````````````` KEY-SET ```````````````````````````````````
// ===============================================================================

/// A validated collection of sub-keys [`KeyList`], belonging to a single parent key.
///
/// `KeySet` represents the *semantic* inside view of a nested key list after parsing:
///
/// ```text
/// {
///     a,
///     b(1, 2),
///     c { d }
/// }
/// ```
///
/// At this stage:
/// - Syntax has already been validated by [`KeyList`] which in itself is a
/// punctuated [`KeyInfo`]
/// - Each sub-key appears **at most once** - non overlapping or duplicated.
/// - All sub-keys are drawn from a known allowed set
///
/// `KeySet` deliberately enforces **exactness**:
/// it rejects unexpected keys and duplicates early, so higher-level
/// caller logic can assume a clean, canonical representation.
///
/// This structure is intentionally minimal:
/// - ordering is discarded
/// - names i.e., keys of underlying [`HashMap`] are normalized to
/// [`String`], although [`KeyInfo`] holds its [`Ident`].
/// - values remain as full [`KeyInfo`] for later typed decoding
#[derive(Debug, Clone)]
pub struct KeySet {
    /// Map from sub-key name -> parsed key information.
    ///
    /// Each key name appears **at most once**.
    /// Duplicate detection is enforced during construction using its
    /// constructor [`KeySet::from_list`].
    pub keys: HashMap<String, KeyInfo>,
}

impl KeySet {
    /// Construct a [`KeySet`] from a parsed [`KeyList`], validating it
    /// against a fixed list of allowed sub-key names.
    ///
    /// Responsibilities of this function:
    /// - Reject **unexpected sub-keys**
    /// - Reject **duplicate sub-keys**
    /// - Preserve the original [`KeyInfo`] for later decoding
    ///
    /// ## Parameters
    ///
    /// - `list`: the parsed list of sub-keys inside a nested group
    /// - `allowed`: the complete set of valid sub-key names for this context
    ///
    /// ## Errors
    ///
    /// Returns a `TokenStream` error if:
    /// - a sub-key name is not present in `allowed`
    /// - the same sub-key appears more than once
    pub fn from_list(list: &KeyList, allowed: &[&Ident]) -> Result<Self, TokenStream> {
        let mut map = HashMap::new();

        // Pre-format the allowed keys for diagnostics.
        // This is used verbatim in error messages to guide the user.
        let expected = {
            let mut collect = Vec::new();
            for allow in allowed {
                collect.push(allow.to_string());
            }
            collect.join(", ")
        };

        // Iterate over parsed sub-keys in source order.
        for key in list.keys.iter() {
            let name = &key.key;

            // Reject any sub-key not explicitly allowed by the schema.
            //
            // This ensures:
            // - no typos silently pass through
            // - decoding logic never sees unknown keys
            if !allowed.contains(&name) {
                return Err(KeyValueErrors::UnexpectedKey {
                    found: key.key.clone(),
                    allowed: expected,
                }
                .into());
            }

            // Reject duplicate sub-keys.
            //
            // This is a structural invariant of `KeySet`:
            // every key may appear at most once.
            if map.contains_key(&name.to_string()) {
                return Err(KeyValueErrors::DuplicateKey {
                    found: key.key.clone(),
                }
                .into());
            }

            // Insert the validated sub-key into the set.
            map.insert(name.to_string(), key.clone());
        }

        Ok(KeySet { keys: map })
    }

    /// Retrieve a sub-key by name from a given [`KeySet`]
    /// and return its [`KeyInfo`].
    ///
    /// Returns `Some(&KeyInfo)` if the key is present,
    /// or `None` otherwise.
    pub fn get(&self, name: &Ident) -> Option<&KeyInfo> {
        self.keys.get(&name.to_string())
    }

    /// Require that a specific sub-key be present in the [`KeySet`]
    /// with optional [`Span`] for error diagnostics.
    ///
    /// This is a convenience helper for schemas that
    /// enforce mandatory sub-keys as opposed to [`KeySet::from_list`].
    ///
    /// ## Errors
    /// Returns a optional spanned diagnostic error as `TokenStream` if the key is missing.
    pub fn require(&self, name: &Ident, span: Option<Span>) -> Result<(), TokenStream> {
        if self.get(name).is_none() {
            return Err(KeyValueErrors::MissingKey {
                span: span.unwrap_or(Span::call_site()),
                require: name.clone(),
            }
            .into());
        }
        Ok(())
    }

    /// Return the number of sub-keys present in this [`KeySet`].
    ///
    /// This is primarily used to enforce **exact matches**
    /// when decoding combinations:
    ///
    /// ```text
    /// matched_required_keys == set.count()
    /// ```
    pub fn count(&self) -> usize {
        self.keys.len()
    }
}

// ===============================================================================
// ````````````````````````````````` KEY-EXPECT ``````````````````````````````````
// ===============================================================================

/// Trait for extracting and diagnosing the *expected value shape* of a key.
///
/// `KeyExpect` defines the contract between:
/// - a parsed [`KeyInfo`] (syntactic form), and
/// - the semantic expectation of a consumer (schema / spec logic).
///
/// Each implementation represents **one exact value form** a key may take,
/// this trait centralizes:
/// - value extraction
/// - validation
/// - uniform diagnostics
///
/// ## Design notes
///
/// - Extraction is *non-destructive* (`Clone`-based)
/// - Failure is always accompanied by a structured error message
/// - Implementations are intentionally simple and declarative
pub trait KeyExpect: Sized {
    /// Short, human-readable description of the expected value form.
    ///
    /// Used directly in diagnostics.
    ///
    /// Example:
    /// - `"a list of identifiers"`
    /// - `"a nested key group"`
    const EXPECTED: &'static str;

    /// Example syntax illustrating the expected form.
    ///
    /// The literal `{key}` is replaced with the actual key name
    /// when constructing diagnostics.
    ///
    /// Example:
    /// - `"{key}(foo, Bar)"`
    /// - `"{key} { subkey(...) }"`
    const EXAMPLE: &'static str;

    /// Optional semantic clarification or constraint.
    ///
    /// This is used to explain *why* a particular form is required
    /// or to clarify subtle distinctions.
    const NOTE: &'static str;

    /// Attempt to extract `Self` from a parsed key.
    ///
    /// Returns `Some(Self)` if the key's value matches the expected form,
    /// or `None` if it does not.
    ///
    /// This function must:
    /// - perform *only* structural matching
    /// - not emit diagnostics
    /// - not mutate the input
    fn extract(key: &KeyInfo) -> Option<Self>;

    /// Extract the expected value from a key or emit a diagnostic error.
    ///
    /// This is the primary entry point used by schema and decoding logic.
    ///
    /// ## Errors
    ///
    /// Returns a `TokenStream` error if:
    /// - the key exists
    /// - but its value does not match the expected form
    #[track_caller]
    fn expect_from(key: &KeyInfo) -> Result<Self, TokenStream> {
        let value = Self::extract(key);
        if let Some(v) = value {
            return Ok(v);
        }

        let mut diag = Self::value_error(&key.key.to_string());
        let span = &mut diag.span;
        *span = DiagSpan::Span(key.key.span());

        Err(diag.into())
    }

    /// Construct a detailed diagnostic message describing the expected value.
    ///
    /// This message is suitable for inclusion in compiler errors and includes:
    /// - a short expectation summary
    /// - a concrete example
    /// - an optional semantic note
    fn value_error(key: &str) -> Diagnostic {
        return KeyValueErrors::InvalidValue {
            key: format_ident!("{}", key),
            expected: Self::EXPECTED.to_string(),
            example: Self::EXAMPLE.replace("{key}", &key.to_string()),
            note: Self::NOTE.to_string(),
        }
        .into();
    }
}

// ===============================================================================
// `````````````````````````````` KEY-EXPECT IMPLS ```````````````````````````````
// ===============================================================================

impl KeyExpect for ExprList {
    const EXPECTED: &'static str = "a list of expressions";
    const EXAMPLE: &'static str = "{key}(foo, bar, b\"baz\", make_bytes())";
    const NOTE: &'static str =
        "expressions may be literals, paths, calls, or other valid Rust expressions";

    fn extract(key: &KeyInfo) -> Option<Self> {
        match &key.value {
            KeyValue::Exprs(v) => Some(v.clone()),
            _ => None,
        }
    }
}

impl KeyExpect for KeyList {
    const EXPECTED: &'static str = "a nested key group";
    const EXAMPLE: &'static str = "{key} { subkey(...) }";
    const NOTE: &'static str = "nested keys must be enclosed in `{}`, `()`, or `[]`";

    fn extract(key: &KeyInfo) -> Option<Self> {
        match &key.value {
            KeyValue::Nested(v) => Some(v.clone()),
            _ => None,
        }
    }
}

impl KeyExpect for () {
    const EXPECTED: &'static str = "no value";
    const EXAMPLE: &'static str = "{key}";
    const NOTE: &'static str = "this key is a flag and must not have arguments";

    fn extract(key: &KeyInfo) -> Option<Self> {
        match &key.value {
            KeyValue::None => Some(()),
            _ => None,
        }
    }
}

impl KeyExpect for ValueGroup {
    const EXPECTED: &'static str = "a group of values";
    const EXAMPLE: &'static str = "{key}{ (foo), (bar, baz) }";
    const NOTE: &'static str = "each group element must be enclosed in parentheses";

    fn extract(key: &KeyInfo) -> Option<Self> {
        match &key.value {
            KeyValue::Group(v) => Some(v.clone()),
            _ => None,
        }
    }
}

impl KeyExpect for ValueGroup<ExprList> {
    const EXPECTED: &'static str = "a group of expression lists";
    const EXAMPLE: &'static str = "{key}{ (foo, bar), (b\"baz\", make_bytes()), (x + y) }";
    const NOTE: &'static str = "each parenthesized group must contain only expressions";

    fn extract(key: &KeyInfo) -> Option<Self> {
        let v = match &key.value {
            KeyValue::Group(v) => v,
            _ => return None,
        };

        let mut collect = Punctuated::<KeyValue<ExprList>, Comma>::new();

        for val in &v.values {
            match val {
                KeyValue::Exprs(expr_list) => {
                    collect.push(KeyValue::Exprs(expr_list.clone()));
                }
                _ => return None,
            }
        }

        Some(Self { values: collect })
    }
}

// ===============================================================================
// ````````````````````````````````` KEY-SCHEMA ``````````````````````````````````
// ===============================================================================

/// Define a schema for a single structured key and generate its decoding logic.
///
/// `key_schema!` is the **user-invokable macro** for declaring how a specific
/// top-level key (e.g. `"exact"`) is parsed from a [`KeyInfo`] and decoded into
/// a strongly-typed enum via [`KeySpec::parse`].
///
/// **Note**: Every `key_schema!` invocation must be unique per module to avoid
/// macro generated hygeine conflicts.
///
/// This macro provides a *declarative schema language* for:
/// - naming a key
/// - declaring its allowed sub-keys and their value types
/// - specifying the exact combinations of sub-keys that are valid
///
/// From this declaration, the macro generates:
///
/// 1. A specification enum (the semantic result)
/// 2. A complete [`KeySpec`] implementation enforcing:
///    - exact sub-key combinations
///    - correct value types
///    - high-quality diagnostics
///
/// ## Syntax
///
/// ```text
/// key_schema! {
///     pub SpecType {
///         key: "literal-key-name",
///         allow_empty: <bool>,
///
///         sub_keys: {
///             sub_key1 : ValueType1,
///             sub_key2 : ValueType2,
///             ...
///         },
///
///         combinations: {
///             Variant1(sub_key1),
///             Variant2(sub_key1, sub_key2),
///             ...
///         }
///     }
/// }
/// ```
///
/// ## Sections
///
/// ### `key`
/// The literal string name of the key as it appears in source code.
///
/// ### `allow_empty`
/// Controls whether the key may appear without a value:
///
/// ```text
/// key
/// ```
///
/// If `true`, this form is accepted and decoded as the `Empty` variant
/// of the generated specification enum.
///  
/// **When `false`, the specification enum does not contain an `Empty` variant.**
///
/// ### `sub_keys`
/// Declares the allowed sub-keys and their value types.
///
/// Any Rust type may be used. A single nested group can be declared
/// with `[ ]`, which is implicitly decoded as:
///
/// ```text
/// T   -> T
/// [T] -> Punctuated<T, Comma>
/// ```
///
/// Only one level of nested grouping is supported.
///
/// ### `combinations`
/// Declares the **exact** sub-key combinations that are valid.
///
/// Each entry:
/// - defines one enum variant
/// - lists the required sub-keys by name
/// - determines the order and types of fields carried by the variant
///
/// Sub-key combinations are matched **exactly**:
/// - all listed sub-keys must be present
/// - no additional sub-keys are permitted
///
/// ## Expansion overview
///
/// For a declaration:
///
/// ```ignore
/// key_schema! {
///     ExactKeySpec {
///         key: "exact",
///         allow_empty: true,
///         sub_keys: {
///             index : IntList,
///             marker : IdentList,
///             counter : [IntList],
///             instance : BStringList,
///         },
///         combinations: {
///             Indexed(index),
///             MarkerInstance(marker, instance),
///             MarkerCounter(marker, counter),
///         }
///     }
/// }
/// ```
///
/// the macro expands (conceptually) to:
///
/// ```ignore
/// pub enum ExactKeySpec {
///     Empty,
///     Indexed(IntList),
///     MarkerInstance(IdentList, BStringList),
///     MarkerCounter(IdentList, Punctuated<IntList, Comma>),
/// }
/// ```
///
/// plus a [`KeySpec`] implementation such that:
///
/// ```ignore
/// ExactKeySpec::parse(&KeyInfo) -> Result<ExactKeySpec, TokenStream>
/// ```
///
/// performs:
/// - key name validation
/// - empty vs nested enforcement
/// - sub-key validation and de-duplication
/// - exact combination matching
/// - typed value decoding
///
/// ## Notes
///
/// - This macro is the **only supported entry point** for defining key schemas.
/// - All lower-level macros (`__emit_*`, `key_spec!`) are internal details.
/// - The generated enum is the authoritative semantic representation
///   of the key's contents.
#[macro_export]
macro_rules! key_schema {
    (
        // Spec type documentation
        $(#[$key_meta:meta])*
        $vis:vis $key_ty:ident {
            key: $key:literal,
            allow_empty: $allow_empty:tt,

            sub_keys: {
                $($sub:ident : $sub_ty:tt),+ $(,)?
            },

            combinations: {
                $(
                    // Variant documentation
                    $(#[$variant_meta:meta])*
                    $variant:ident ( $($field:ident),+ )
                ),+ $(,)?
            }
        }
    ) => {
        // Phase 1: Generate identifier -> index and identifier -> type lookups
        $crate::__emit_subkey_index!(
            @acc []
            0,
            $( $sub : $sub_ty ),+
        );

        $crate::__emit_subkey_ty!(
            @acc [],
            $( $sub : $sub_ty ),+
        );

        $crate::__emit_subkey_parse_decode!(
            @acc [],
            $( $sub : $sub_ty ),+
        );

        $crate::__emit_subkey_value_ty!(
            @acc [],
            $( $sub : $sub_ty ),+
        );

        $crate::__emit_subkey_grouped_ty!(
            @acc [],
            $( $sub : $sub_ty ),+
        );

        // Phase 2: Expand into the internal `key_spec!` macro
        $crate::__key_spec! {

            $(#[$key_meta])*
            $vis $key_ty => {
                key: $key,
                allow_empty: $allow_empty,

                sub_keys: {
                    $( $sub => $sub_ty ),+
                },

                combinations: [
                    $(
                        $(#[$variant_meta])*
                        $variant :
                        [ $( __subkey_index!($field) ),+ ]
                        =>
                        ( $( __subkey_value_ty!($field) ),+ )
                        =>
                        ( $( $field ),+ )
                    ),+
                ]
            }
        }
    };
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// `````````````````````````````````` KEY-SPEC ```````````````````````````````````
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Schema definition trait for a *single top-level key* with structured sub-keys.
///
/// `KeySpec` defines the *semantic decoding contract* for one specific key
/// (e.g. `"exact"`), describing how a parsed [`KeyInfo`] is transformed into a
/// strongly-typed enum-based specification with precise diagnostics.
///
/// `Self` is typically an enum with one variant per combination,
/// plus an optional `Empty` variant.
///
/// This trait is **not implemented manually** if structured enum
/// implementations are generated by the [`crate::key_schema!`] macro,
/// which encodes:
///
/// - the literal key name
/// - the complete set of allowed sub-keys
/// - the exact combinations of sub-keys that are permitted
/// - the concrete Rust types associated with each combination
///
/// The generated implementation uses this information to enforce both
/// *structure* (which sub-keys may appear) and *shape* (which combinations
/// are valid), producing clear, context-rich error messages on failure.
///
/// ## Conceptual model
///
/// A key governed by `KeySpec` has one of the following forms:
///
/// ```text
/// key
/// key { <sub-keys> }
/// ```
///
/// Where `<sub-keys>` must match **exactly one** of the combinations declared
/// in the corresponding [`crate::key_schema!`] invocation.
///
/// Empty values (`key`) are accepted only when `ALLOW_EMPTY == true`.
///
/// ## Responsibilities
///
/// `KeySpec` is responsible for:
/// - validating the top-level key name
/// - enforcing empty vs nested usage rules
/// - rejecting unexpected or duplicate sub-keys
/// - selecting exactly one valid sub-key combination
/// - decoding sub-key values into a strongly typed enum variant
/// - producing high-quality, schema-aware diagnostics
///
/// It intentionally does *not*:
/// - parse raw tokens
/// - interpret individual sub-key value syntax
/// - perform semantic cross-checks between unrelated keys
///
/// Those concerns are handled by [`KeyInfo`], [`KeyExpect`], [`KeySet`],
/// and higher-level logic built on top of the decoded spec.
pub trait KeySpec: Sized {
    /// The literal name of the key (e.g. `"exact"`).
    ///
    /// Used for:
    /// - key matching
    /// - diagnostics
    const KEY: &'static str;

    /// All allowed sub-keys for this key.
    ///
    /// Each entry consists of:
    /// - the sub-key name
    /// - a function producing a diagnostic message describing
    ///   the expected value form
    const SUB_KEYS: &'static [(&'static str, fn(&DiagSpan, Option<ErrorInfo>) -> Diagnostic)];

    /// The set of *exact* sub-key combinations accepted by this key.
    ///
    /// Each combination is represented as a slice of indices into
    /// `SUB_KEYS`.
    ///
    /// A combination matches if and only if:
    /// - all indexed sub-keys are present
    /// - no additional sub-keys are present
    ///
    /// The order of combinations is significant and determines
    /// decoding priority.
    const SUB_KEY_COMBINATIONS: &'static [&'static [usize]];

    /// Whether this key may appear without an associated value.
    ///
    /// If `true`, a bare key (`key`) is accepted and decoded
    /// as [`KeySpec::empty_variant()`].
    ///
    /// If `false`, the key must always contain a nested sub-key group.
    const ALLOW_EMPTY: bool;

    /// Return the list of all allowed sub-key names.
    ///
    /// This is derived mechanically from `SUB_KEYS` and is primarily
    /// used when validating nested key groups.
    fn expected_sub_keys() -> Vec<Ident> {
        let mut keys = Vec::new();
        for (name, _) in Self::SUB_KEYS {
            keys.push(format_ident!("{}", *name));
        }
        keys
    }

    /// Construct a diagnostic error for an unexpected key name.
    #[inline]
    fn invalid_key_err(found: &Ident) -> TokenStream {
        Self::invalid_key_msg(found).into()
    }

    /// Construct a diagnostic error for an invalid key value.
    #[inline]
    fn invalid_value_err(key: &Ident) -> TokenStream {
        Self::invalid_value_msg(key).into()
    }

    /// Format the diagnostic message for an unexpected key name.
    fn invalid_key_msg(found: &Ident) -> Diagnostic {
        return KeyValueErrors::InvalidKey {
            found: found.clone(),
            require: format_ident!("{}", Self::KEY),
        }
        .to_diagnostic();
    }

    /// Format the diagnostic message for an invalid key value.
    #[inline]
    fn invalid_value_msg(key: &Ident) -> Diagnostic {
        Self::help(&key.span())
    }

    /// Parse and decode a key according to this schema.
    ///
    /// This is the **primary entry point** for schema-driven decoding.
    /// It validates the structure of a parsed [`KeyInfo`] and produces
    /// a strongly-typed semantic specification `Self`.
    ///
    /// The parsing process is **deterministic and total**:
    /// given a valid key, exactly one semantic result is produced;
    /// otherwise, a precise diagnostic error is returned.
    ///
    /// ## Validation pipeline
    ///
    /// This function performs the following steps in order:
    ///
    /// 1. **Key name validation**
    ///    - Ensures the parsed key name matches [`KeySpec::KEY`]
    ///
    /// 2. **Empty form handling**
    ///    - If [`KeySpec::ALLOW_EMPTY`] is `true`, accepts a bare key:
    ///
    ///      ```text
    ///      key
    ///      ```
    ///
    ///    - Produces [`KeySpec::empty_variant()`] in this case
    ///
    /// 3. **Nested form enforcement**
    ///    - If the key is not empty (or empty is disallowed),
    ///      requires a nested sub-key group:
    ///
    ///      ```text
    ///      key { ... }
    ///      ```
    ///
    /// 4. **Sub-key validation**
    ///    - Rejects unexpected sub-keys
    ///    - Rejects duplicate sub-keys
    ///    - Produces a validated [`KeySet`]
    ///
    /// 5. **Semantic decoding**
    ///    - Delegates to [`KeySpec::decode_from_set`] to:
    ///      - select exactly one valid sub-key combination
    ///      - decode sub-key values into concrete Rust types
    ///
    /// This function itself does **not** interpret sub-key values.
    /// All value decoding is performed during step (5) by
    /// [`KeySpec::decode_from_set`].
    ///
    /// ## Errors
    ///
    /// Returns a diagnostic error if:
    ///
    /// - the key name does not match [`KeySpec::KEY`]
    /// - the value shape is invalid for the schema
    /// - the key is empty but `ALLOW_EMPTY == false`
    /// - a sub-key is unexpected or duplicated
    /// - no declared combination matches exactly
    /// - a sub-key value fails to decode
    #[track_caller]
    fn parse(key: &KeyInfo) -> Result<Self, TokenStream> {
        // 1. main key name
        if key.key != Self::KEY {
            return Err(Self::invalid_key_err(&key.key));
        }

        // 2. empty case
        if Self::ALLOW_EMPTY {
            if <() as KeyExpect>::expect_from(key).is_ok() {
                return Ok(Self::empty_variant().unwrap());
            }
        }

        // 3. must be nested
        let nested = <KeyList as KeyExpect>::expect_from(key)
            .map_err(|_| Self::invalid_value_err(&key.key))?;

        // 4. validate sub-keys
        let expected = Self::expected_sub_keys();
        let allowed: Vec<&Ident> = expected.iter().collect();
        let set = KeySet::from_list(&nested, &allowed)?;

        // 5. decode
        Self::decode_from_set(&set, &key.key)
    }

    /// Construct the semantic value representing an empty key.
    ///
    /// This method produces the specification value corresponding to
    /// a bare key with no associated value:
    ///
    /// ```text
    /// key
    /// ```
    ///
    /// ## Usage
    ///
    /// This function is **only called** by [`KeySpec::parse`] when:
    ///
    /// - [`KeySpec::ALLOW_EMPTY`] is `true`, and
    /// - the parsed key has no associated value
    /// - hence safely unwrap the optional value.
    ///
    /// The returned value must represent the semantic meaning of an
    /// empty key for this schema.
    ///
    /// ## Notes
    ///
    /// - When `ALLOW_EMPTY == false`, this method is never invoked
    ///   and if invoked returns a `None` value, due to guards in `parse`,
    ///   the empty form is safely rejected during parsing
    fn empty_variant() -> Option<Self>;

    /// Decode a validated set of sub-keys into a semantic specification value.
    ///
    /// This method performs the **final decoding step** of a [`KeySpec::parse`]
    /// invocation, transforming a structurally validated [`KeySet`] into the
    /// strongly-typed specification enum `Self`.
    ///
    /// At this stage, all *structural guarantees* have already been enforced:
    ///
    /// - the top-level key name has been validated
    /// - the value is known to be a nested key group
    /// - all sub-keys are drawn from the allowed set
    /// - no sub-key appears more than once
    ///
    /// What remains is **semantic decoding**.
    ///
    /// ## Responsibilities
    ///
    /// Implementations must:
    ///
    /// - select **exactly one** matching sub-key combination
    /// - reject:
    ///   - missing required sub-keys
    ///   - extra sub-keys
    ///   - partial matches
    /// - decode each sub-key's value into the concrete Rust type
    ///   declared for that combination
    ///
    /// The combination match must be **exact**:
    ///
    /// ```text
    /// required_keys == present_keys
    /// ```
    ///
    /// No implicit defaults, fallbacks, or partial decoding is permitted.
    ///
    /// This design keeps schemas declarative and structural, while allowing
    /// richer value semantics to be opt-in at the type level.
    ///
    /// ## Invariants
    ///
    /// Implementations can assume:
    ///
    /// - `set` contains no duplicate keys
    /// - every key in `set` is allowed by the schema
    /// - `set.count()` accurately reflects the number of present sub-keys
    ///
    /// ## Errors
    ///
    /// Returns a diagnostic error if:
    ///
    /// - no declared combination matches exactly
    /// - more than one combination could match
    /// - a sub-key's value fails to decode
    /// - a permutation is invalid for the decoded value type
    fn decode_from_set(set: &KeySet, key_ident: &Ident) -> Result<Self, TokenStream>;

    /// Generate a detailed help message describing valid key usage.
    ///
    /// This message includes:
    /// - allowed sub-key combinations
    /// - expected value forms for each sub-key
    fn help(span: &Span) -> Diagnostic {
        let msg = match Self::ALLOW_EMPTY {
            true => format!(
                "expected `{0}` maybe without value or  `{0}` {{ <sub-keys> }}",
                Self::KEY,
            ),
            false => format!("expected `{0}` {{ <sub-keys> }}", Self::KEY,),
        };

        let mut diag = Diagnostic {
            error: KeyParseError::InvalidValues {}.into(),
            tags: vec![ErrorTag::Core(ErrorCatalog::Unsupported)],
            msg,
            span: DiagSpan::Span(*span),
            helps: Vec::new(),
            notes: Vec::new(),
        };

        diag.helps.push(Help {
            span: None,
            msg: "Valid sub-key combinations are:".into(),
        });

        // Build a human-readable list of allowed sub-key combinations.
        //
        // Each entry in `SUB_KEY_COMBINATIONS` is a slice of indices into `SUB_KEYS`,
        // representing one *exact* combination of sub-keys that is accepted.
        //
        // Example output:
        //   - `index`
        //   - `marker` + `instance`
        //   - `marker` + `counter`
        for combo in Self::SUB_KEY_COMBINATIONS {
            let combo = combo
                .iter()
                .map(|&i| format!("`{}`", Self::SUB_KEYS[i].0))
                .collect::<Vec<_>>()
                .join(" + ");

            diag.helps.push(Help {
                span: None,
                msg: combo,
            });
        }

        // Build the detailed per-sub-key value expectations.
        //
        // Each entry in `SUB_KEYS` provides:
        // - the sub-key name
        // - a function that generates a diagnostic message describing
        //   the expected value shape for that sub-key
        for (name, expect) in Self::SUB_KEYS {
            let sub = expect(&DiagSpan::Span(*span), None);

            let msg = sub.msg;
            diag.notes.push(Note { msg });

            let mut pushed_help = false;
            for help in sub.helps.iter().filter(|h| h.span.is_none()) {
                if !pushed_help {
                    diag.notes.push(Note {
                        msg: format!("{LINE_SPACE}{LINE_SPACE}sub-key `{}` help:", name),
                    });
                    pushed_help = true
                }
                diag.notes.push(Note {
                    msg: format!("{LINE_SPACE}{LINE_SPACE}{LINE_SPACE}{}", help.msg),
                });
            }

            let mut pushed_note = false;
            for note in &sub.notes {
                if !pushed_note {
                    diag.notes.push(Note {
                        msg: format!("{LINE_SPACE}{LINE_SPACE}sub-key `{}` note:", name),
                    });
                    pushed_note = true
                }
                diag.notes.push(Note {
                    msg: format!("{LINE_SPACE}{LINE_SPACE}{LINE_SPACE}{}", note.msg),
                });
            }
        }

        // Assemble the final help message.
        //
        // This message is included verbatim in diagnostics and describes:
        // - the two valid structural forms of the key (empty or nested)
        // - the exact sub-key combinations that are permitted
        // - the expected value forms for each individual sub-key
        diag
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ````````````````````````` KEY-SCHEMA INTERNAL MACROS ``````````````````````````
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Internal implementation macro utilized by [`crate::key_schema!`].
///
/// It is **not intended to be invoked directly**.
///
/// This macro expands a declarative key schema into:
/// - a strongly-typed specification enum
/// - its [`KeySpec`] implementation enforcing exact sub-key combinations
///
/// All external usage should go through [`crate::key_schema!`], which provides
/// a higher-level, user-facing syntax and generates the required
/// supporting lookup macros.
///
/// This separation keeps the public schema definition ergonomic
/// while allowing `key_spec!` to focus solely on deterministic decoding
/// and diagnostics.
#[macro_export]
macro_rules! __key_spec {
    (
        // Metadata for the marker key type
        $(#[$key_meta:meta])*
        $vis:vis $key_ty:ident => {
            key: $key:literal,
            allow_empty: $allow_empty:tt,

            sub_keys: {
                $($sub:ident => $sub_ty:tt),+ $(,)?
            },

            combinations: [
                $(
                    // Metadata for each enum variant
                    $(#[$variant_meta:meta])*
                    $variant:ident :
                    [ $($idx:expr),+ ]
                    =>
                    ( $($ty:ty),+ )
                    =>
                    ( $($field:ident),+ )
                ),+ $(,)?
            ]
        }
    ) => {
        // Generate the strongly-typed specification enum.
        //
        // - One variant per declared combination
        // - Each variant carries values in the exact order declared
        // - An `Empty` variant is generated **only when `allow_empty: true`**

        $crate::__key_spec!(
            @spec_enum
            vis = $vis,
            allow_empty = $allow_empty,
            key_meta = [$($key_meta)*],
            key_ty = $key_ty,
            key = $key,
            variants = {
                $(
                    $(#[$variant_meta])*
                    $variant ( $( $ty ),+ )
                ),+
            }
        );

        // Generate a inherent impl for the Key type.
        //
        // - includes a helper method for empty_variant
        $crate::__key_spec!(
            @empty_variant_impl
            vis = $vis,
            allow_empty = $allow_empty,
            key_ty = $key_ty,
        );

        // Implement the `KeySpec` contract for this spec type.
        impl $crate::keys::KeySpec for $key_ty
        {
            // Literal key name used for matching and diagnostics
            const KEY: &'static str = $key;

            // Policy flag controlling whether a bare key is accepted
            const ALLOW_EMPTY: bool = $allow_empty;

            // All allowed sub-keys for this key.
            //
            // Each entry pairs:
            // - the sub-key name as a string
            // - a function producing a value-expectation diagnostic
            //
            // The order of this array is *significant*:
            // it defines the indexing scheme used by combinations.
            const SUB_KEYS: &'static [(
                &'static str,
                fn(
                    &$crate::errors::DiagSpan,
                    Option<$crate::errors::ErrorInfo>
                ) -> $crate::errors::Diagnostic,
            )] = &[
                $(
                    (
                        stringify!($sub),
                        <__subkey_grouped_ty!($sub) as $crate::errors::ParseDiagnostic>
                            ::parse_diagnostic
                    ),
                )+
            ];


            // The set of exact sub-key combinations accepted by this key.
            //
            // Each combination is represented as a slice of indices
            // into `SUB_KEYS`.
            //
            // For example:
            //   &[0, 2]
            //
            // means:
            //   - SUB_KEYS[0].0
            //   - SUB_KEYS[2].0
            //
            // must be present *and no others*.
            const SUB_KEY_COMBINATIONS: &'static [&'static [usize]] = &[
                $( &[ $( $idx ),+ ] ),+
            ];

            // Construct the semantic value for an empty key.
            //
            // This is only called when `ALLOW_EMPTY == true`.
            fn empty_variant() -> Option<Self> {
                Self::__empty_variant()
            }

            // Decode the sub-key value.
            //
            // This keeps:
            // - schema definitions declarative
            // - permutation logic reusable and orthogonal
            // - diagnostics consistent across all keys
            fn decode_from_set(
                set: &$crate::keys::KeySet,
                key_ident: &syn::Ident,
            ) -> Result<Self, proc_macro2::TokenStream> {

                // The macro expands the following block once per declared
                // combination in the order listed in `combinations:`.
                //
                // Conceptually, the expansion is equivalent to:
                //
                //   if matches_combo_1 { return Variant1(...) }
                //   else if matches_combo_2 { return Variant2(...) }
                //
                // The first matching combination wins.

                $(
                    {
                        // Tracks whether *all required sub-keys*
                        // for this combination are present.
                        let mut ok = true;

                        // Counts how many required sub-keys
                        // from this combination were found.
                        //
                        // This is later compared with `set.count()`
                        // to ensure *no extra keys* are present.
                        let mut matched = 0;

                        // Iterate over the indices declared for this combination.
                        //
                        // Each `$idx` refers to an entry in `SUB_KEYS`.
                        $(
                            // Resolve the required sub-key name.
                            let name = Self::SUB_KEYS[$idx].0;

                            // Check for presence in the validated KeySet.
                            //
                            // - Present: increment `matched`
                            // - Absent: this combination cannot match
                            if set.get(&quote::format_ident!("{}", name)).is_some() {
                                matched += 1;
                            } else {
                                ok = false;
                            }
                        )+

                        // At this point:
                        //
                        // - `ok == true` means all required sub-keys are present
                        // - `matched` is the number of required keys found
                        //
                        // The final check enforces *exact matching*:
                        //
                        //   matched == set.count()
                        //
                        // This guarantees:
                        // - no required keys are missing
                        // - no extra keys are present
                        if ok && matched == set.count() {

                            // This combination matches exactly.
                            //
                            // Decode each sub-key value using the explicit
                            // type declared in the combination tuple.
                            //
                            // The order of decoding matches the order
                            // of `$idx` and `$ty` in the macro input.
                            return Ok(
                                Self::$variant(
                                    $(
                                        // Derive the ExprList or ValueGroup<ExprList>
                                        // and decode it to the expected variant type
                                        match __subkey_parse_decode!(
                                        $field,
                                        set,
                                        $idx
                                        ) {
                                            Ok(v) => v,
                                            Err(e) => return Err(e),
                                        },

                                    )+
                                )
                            );
                        }

                        // Combination did not match; try the next one.
                    }
                )+

                // No combination matched exactly.
                //
                // This indicates that:
                // - the set of sub-keys is structurally invalid
                // - or does not correspond to any declared combination
                Err(Self::invalid_value_err(key_ident))
            }
        }
    };

    // Specification enum helpers

    // allow_empty = true -> include `Empty`
    (
        @spec_enum
        vis = $vis:vis,
        allow_empty = true,
        key_meta = [$($key_meta:meta)*],
        key_ty = $key_ty:ident,
        key = $key:literal,
        variants = {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident ( $( $ty:ty ),+ )
            ),+
        }
    ) => {
        $(#[$key_meta])*
        #[doc = concat!(
            "Enum type representing the `", $key, "` key schema.\n\n",
            "This enum is generated by [`key_schema!`] and exists \n",
            "to host the [`KeySpec`](", stringify!($crate), "::keys::KeySpec) trait implementation.",
            "",
        )]
        $vis enum $key_ty {

            #[doc = concat!(
                "Represents the `", $key, "` key specified without any sub-keys.\n\n",
                "This variant is generated only when `allow_empty: true` is set in\n",
                "the corresponding `key_schema!` declaration.",
                "",
            )]
            Empty,

            $(
                $(#[$variant_meta])*
                $variant( $( $ty ),+ ),
            )+
        }
    };

    // allow_empty = false -> no `Empty`
    (
        @spec_enum
        vis = $vis:vis,
        allow_empty = false,
        key_meta = [$($key_meta:meta)*],
        key_ty = $key_ty:ident,
        key = $key:literal,
        variants = {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident ( $( $ty:ty ),+ )
            ),+
        }
    ) => {
        $(#[$key_meta])*
        #[doc = concat!(
            "Enum type representing the `", $key, "` key schema.\n\n",
            "This enum is generated by [`key_schema!`] and exists \n",
            "to host the [`KeySpec`](", stringify!($crate), "::keys::KeySpec) trait implementation.",
            "",
        )]
        $vis enum $key_ty {
            $(
                $(#[$variant_meta])*
                $variant( $( $ty ),+ ),
            )+
        }
    };

    // empty_variant helpers

    // allow_empty = true -> `Empty` exists
    (
        @empty_variant_impl
        vis = $vis:vis,
        allow_empty = true,
        key_ty = $key_ty:ident,
    ) => {
        impl $key_ty {
            $vis fn __empty_variant() -> Option<$key_ty> {
                Some($key_ty::Empty)
            }
        }
    };

    // allow_empty = false -> no `Empty`
    (
        @empty_variant_impl
        vis = $vis:vis,
        allow_empty = false,
        key_ty = $key_ty:ident,
    ) => {
        impl $key_ty {
            $vis fn __empty_variant() -> Option<$key_ty> {
                None
            }
        }
    };
}

/// Internal helper macro for [`crate::key_schema!`].
///
/// Generates a lookup macro (`__subkey_index`) that maps sub-key identifiers
/// to stable numeric indices based on their declaration order.
///
/// These indices are later used by [`__key_spec!`] to encode sub-key combinations
/// as index slices, enabling exact-match validation and efficient decoding.
///
/// This macro is an implementation detail and must not be invoked directly.
#[macro_export]
macro_rules! __emit_subkey_index {
    (
        // Terminal rule:
        //
        // All sub-keys have been processed.
        // Emit a macro named `__subkey_index` containing
        // the accumulated identifier -> index mappings.
        @acc [$($out:tt)*]
        $idx:expr,
    ) => {
        macro_rules! __subkey_index {
            $($out)*
        }
    };

    (
        // Recursive rule:
        //
        // Process the next `(ident : type)` pair,
        // assign it the current index, and recurse.
        @acc [$($out:tt)*]
        $idx:expr,
        $head:ident : $ty:tt $(, $rest:ident : $rest_ty:tt)*
    ) => {
        $crate::__emit_subkey_index!(
            @acc
            [
                // Accumulate a mapping from sub-key identifier
                // to its numeric index.
                //
                // This allows later macros to refer to sub-keys
                // symbolically while emitting index-based tables.
                $($out)*
                ($head) => { $idx };
            ]
            // Increment index for the next sub-key
            $idx + 1,
            // Recurse over remaining sub-keys
            $( $rest : $rest_ty ),*
        );
    };
}

#[macro_export]
macro_rules! __emit_subkey_ty {
    (
        @acc [$($out:tt)*],
    ) => {
        macro_rules! __subkey_ty {
            $($out)*
        }
    };

    (
        @acc [$($out:tt)*],
        $head:ident : $ty:tt $(, $rest:ident : $rest_ty:tt)*
    ) => {
        $crate::__emit_subkey_ty!(
            @acc
            [
                $($out)*
                ($head) => { $ty };
            ],
            // Recurse over remaining sub-keys
            $( $rest : $rest_ty ),*
        );
    };
}

#[macro_export]
macro_rules! __decode_parse_subkey {
    // =========================================================================
    // GROUPED: [ExprList]
    // =========================================================================
    (
        [ExprList],
        $set:expr,
        $idx:expr
    ) => {
        <$crate::keys::ValueGroup<$crate::lists::ExprList> as $crate::keys::KeyExpect>::expect_from(
            $set.get(&quote::format_ident!("{}", Self::SUB_KEYS[$idx].0))
                .unwrap(),
        )
    };

    // =========================================================================
    // GROUPED: [T]
    // =========================================================================
    (
        [$ty:ty],
        $set:expr,
        $idx:expr
    ) => {
        <syn::punctuated::Punctuated<$ty, syn::token::Comma>
                                            as TryFrom<
                                                $crate::keys::ValueGroup<$crate::lists::ExprList>
                                            >
                                        >::try_from(
                                            <$crate::keys::ValueGroup<$crate::lists::ExprList>
                                                as $crate::keys::KeyExpect
                                            >::expect_from(
                                                $set.get(
                                                    &quote::format_ident!(
                                                        "{}",
                                                        Self::SUB_KEYS[$idx].0
                                                    )
                                                ).unwrap()
                                            )?
                                        )
    };

    // =========================================================================
    // NORMAL: ExprList
    // =========================================================================
    (
        ExprList,
        $set:expr,
        $idx:expr
    ) => {
        <$crate::lists::ExprList as $crate::keys::KeyExpect>::expect_from(
            $set.get(&quote::format_ident!("{}", Self::SUB_KEYS[$idx].0))
                .unwrap(),
        )
    };

    // =========================================================================
    // NORMAL: T - MUST BE LAST
    // =========================================================================
    (
        $ty:ty,
        $set:expr,
        $idx:expr
    ) => {
        <$ty as TryFrom<$crate::lists::ExprList>>::try_from(
            <$crate::lists::ExprList as $crate::keys::KeyExpect>::expect_from(
                $set.get(&quote::format_ident!("{}", Self::SUB_KEYS[$idx].0))
                    .unwrap(),
            )?,
        )
    };
}

#[macro_export]
macro_rules! __emit_subkey_parse_decode {
    (
        @acc [$($out:tt)*],
    ) => {
        macro_rules! __subkey_parse_decode {
            $($out)*
        }
    };

    (
        @acc [$($out:tt)*],
        $head:ident : $ty:tt $(, $rest:ident : $rest_ty:tt)*
    ) => {
        $crate::__emit_subkey_parse_decode!(
            @acc [
                $($out)*

                ($head, $set:expr, $idx:expr) => {
                    $crate::__decode_parse_subkey!(
                        $ty,
                        $set,
                        $idx
                    )
                };
            ],
            $( $rest : $rest_ty ),*
        );
    };
}

#[macro_export]
macro_rules! __emit_subkey_value_ty {
    (
        @acc [$($out:tt)*],
    ) => {
        macro_rules! __subkey_value_ty {
            $($out)*
        }
    };

    (
        @acc [$($out:tt)*],
        $head:ident : $ty:tt $(, $rest:ident : $rest_ty:tt)*
    ) => {
        $crate::__emit_subkey_value_ty!(
            @acc [
                $($out)*

                ($head) => {
                    $crate::__subkey_value_ty!($ty)
                };
            ],
            $( $rest : $rest_ty ),*
        );
    };
}

#[macro_export]
macro_rules! __subkey_value_ty {
    ([$ty:ty]) => {
        syn::punctuated::Punctuated<$ty, syn::token::Comma>
    };

    ($ty:ty) => {
        $ty
    };
}

#[macro_export]
macro_rules! __emit_subkey_grouped_ty {
    (
        @acc [$($out:tt)*],
    ) => {
        macro_rules! __subkey_grouped_ty {
            $($out)*
        }
    };

    (
        @acc [$($out:tt)*],
        $head:ident : $ty:tt $(, $rest:ident : $rest_ty:tt)*
    ) => {
        $crate::__emit_subkey_grouped_ty!(
            @acc [
                $($out)*

                ($head) => {
                    $crate::__subkey_grouped_ty!($ty)
                };
            ],
            $( $rest : $rest_ty ),*
        );
    };
}

#[macro_export]
macro_rules! __subkey_grouped_ty {
    ([$ty:ty]) => {
        $crate::keys::ValueGroup<$ty>
    };

    ($ty:ty) => {
        $ty
    };
}
