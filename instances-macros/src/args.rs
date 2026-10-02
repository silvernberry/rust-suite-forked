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
// ````````````````````````` PARSED PROC-MACRO ARGUMENTS `````````````````````````
// ===============================================================================

//! Parsers for instances proc-macros user inputs.
//!
//! This module defines the argument parsers used by the instance proc
//! macros. Each parser converts an incoming [`TokenStream`] into a
//! semantically meaningful representation tailored to the macro being
//! expanded.

// ===============================================================================
// ``````````````````````````````````` IMPORTS ```````````````````````````````````
// ===============================================================================

// --- Proc-Suite ---
use proc_suite::{IdentList, IntList, ParseDiagnostic};

// --- Proc Macro Crates ---
use proc_macro::TokenStream;
use syn::{
    Attribute, Error, Expr, ItemImpl, ItemTrait, parse,
    parse::{Parse, ParseStream},
    spanned::Spanned,
    token::Comma,
};

// --- Local Crates ---
use crate::{
    Extraction, ParseBug, ParseError, ProcParseErr,
    access::{
        args::{AccessArgs, BStrInputList, LeafAccess},
        errors::KeysError,
    },
};

// ===============================================================================
// ``````````````````````````````` INSTANCE TRAIT ````````````````````````````````
// ===============================================================================

/// Parsed input for the [`instance_trait`](crate::instance_trait) proc-macro.
///
/// This structure contains:
///
/// - the parsed target trait definition,
/// - and the optional instance-counter selection arguments
///   supplied to the macro.
///
/// Attribute arguments may select instance counters using either:
///
/// - const-generic identifiers,
/// - or positional generic indexes.
///
/// Both forms are normalized into [`InstanceArgs`] during extraction.
#[derive(Clone, Debug)]
pub(crate) struct InstanceTraitArgs {
    pub(crate) item: ItemTrait,
    pub(crate) args: Option<InstanceArgs>,
}

/// Arguments accepted by instance-related attributes on a **trait definition**.
///
/// Attribute arguments of [`instance_trait`](crate::instance_trait) is parsed
/// into suitable  variant of `InstanceArgs`.
///
/// Instance macros applied to traits allow users to select instance counters in
/// one of two ways:
/// - by naming them explicitly using their const-generic **identifiers**, or
/// - by referring to them positionally using **integer indices**.
///
/// This enum normalizes those inputs into a single representation so the
/// expansion logic does not need to care how the user chose to express
/// their intent.
///
/// ## Examples
///
/// ```rust
/// trait Example<const A: u8, const B: u8, const C: u8> {}
/// ```
///
/// Selecting instance counters by **identifier**:
///
/// ```ignore
/// #[_macro_(A, C)]
/// trait Example<const A: u8, const B: u8, const C: u8> {}
/// ```
///
/// Selecting the same instance counters by **positional index**:
///
/// ```ignore
/// #[_macro_(0, 2)]
/// trait Example<const A: u8, const B: u8, const C: u8> {}
/// ```
///
/// Both forms describe the same intent and are normalized by
/// [`InstanceArgs`] into a common representation for downstream expansion.
#[derive(Debug, Clone)]
pub(crate) enum InstanceArgs {
    /// A list of instance names supplied to the macro
    /// (e.g. `#[_macro_(A, B, C)]`).
    Ident(IdentList),

    /// A list of instance indices supplied to the macro
    /// (e.g. `#[_macro_(0, 2)]`).
    Index(IntList),
}

impl Extraction<TokenStream, TokenStream> for InstanceTraitArgs {
    fn raw_extract(
        from: &TokenStream,
        context: &TokenStream,
    ) -> Result<Self, proc_macro2::TokenStream> {
        let item = match parse::<ItemTrait>(from.clone()) {
            Ok(t) => t,
            Err(e) => {
                return Err(<ItemTrait as ParseDiagnostic>::span_parse_error(
                    &e.span(),
                    Some(ParseError::ItemTrait.into()),
                ));
            }
        };

        // Try parsing argument list as identifiers or integers
        let try_idents = parse::<IdentList>(context.clone());
        let try_ints = parse::<IntList>(context.clone());

        let args = match (try_idents, try_ints) {
            (Ok(_), Ok(_)) => {
                // Both present - means no value provided, hence treat as empty/default
                None
            }
            (Ok(idents), Err(_)) => {
                // Identifier list parsed correctly
                if idents.idents.is_empty() {
                    None
                } else {
                    Some(InstanceArgs::Ident(idents))
                }
            }
            (Err(_), Ok(ints)) => {
                // Integer list parsed correctly
                if ints.ints.is_empty() {
                    None
                } else {
                    Some(InstanceArgs::Index(ints))
                }
            }
            (Err(e_ident), Err(e_ints)) => {
                let mut either_or_err: Error = ProcParseErr::IdentsOrInts {
                    args: context.clone().into(),
                }
                .into();
                either_or_err.combine(e_ident);
                either_or_err.combine(e_ints);

                return Err(either_or_err.to_compile_error().into());
            }
        };
        Ok(InstanceTraitArgs { item, args })
    }

    fn validate_extract(
        &self,
        _: &TokenStream,
        _: Option<&TokenStream>,
    ) -> Result<(), proc_macro2::TokenStream> {
        Ok(())
    }
}

// ===============================================================================
// `````````````````````````````` INSTANCE IMPL ``````````````````````````````````
// ===============================================================================

/// Parsed input for the [`instance_impl`](crate::instance_impl) proc-macro
/// which follows [`instance`](crate::instance) proc-macro applied on its
/// associated trait definition.
///
/// This structure contains:
///
/// - the parsed target impl definition,
/// - and the optional instance-counter selection arguments
///   supplied to the macro.
///
/// Attribute arguments may select instance counters using
/// const-generic identifiers, which is normalized into
/// [`IntList`] during extraction.
///
/// ## Examples
///
/// ```ignore
/// #[instance(0, 2)]
/// trait Trait<const A: u8, const B: u8, const C: u8> {}
///
/// // Selecting the same instance counters by **positional index**:
/// #[instance_impl(0, 2)]
/// impl Trait<1, 2, 3> for Struct {}
/// ```
///
/// Or in case if the trait instance is supplied by **identifiers**, still the
/// impl macro requires its positional indexes.
///
/// ```ignore
/// #[instance(A, C)]
/// trait Trait<const A: u8, const B: u8, const C: u8> {}
///
/// #[instance_impl(0, 2)]
/// impl Trait<1, 2, 3> for Struct {}
/// ```
#[derive(Clone, Debug)]
pub(crate) struct InstanceImplArgs {
    pub(crate) item: ItemImpl,
    pub(crate) args: Option<IntList>,
}

impl Extraction<TokenStream, TokenStream> for InstanceImplArgs {
    fn raw_extract(
        from: &TokenStream,
        context: &TokenStream,
    ) -> Result<Self, proc_macro2::TokenStream> {
        let item = match parse::<ItemImpl>(from.clone()) {
            Ok(t) => t,
            Err(e) => {
                return Err(<ItemTrait as ParseDiagnostic>::span_parse_error(
                    &e.span(),
                    Some(ParseError::ItemImpl.into()),
                ));
            }
        };

        // Try parsing argument list as integers
        let try_ints = parse::<IntList>(context.clone());

        let args = match try_ints {
            Ok(l) => {
                if l.ints.is_empty() {
                    None
                } else {
                    Some(l)
                }
            }
            Err(e) => {
                return Err(<IntList as ParseDiagnostic>::span_parse_error(
                    &e.span(),
                    Some(ParseError::IntList.into()),
                ));
            }
        };

        Ok(InstanceImplArgs { item, args })
    }

    fn validate_extract(
        &self,
        _: &TokenStream,
        _: Option<&TokenStream>,
    ) -> Result<(), proc_macro2::TokenStream> {
        Ok(())
    }
}

// ===============================================================================
// ```````````````````````````````` INSTANCE ACCESS ``````````````````````````````
// ===============================================================================

/// Parsed input for the [`instance_access`](crate::instance_access) proc-macro.
///
/// Represents the input used to transform an instance-bound implementation
/// according to a requested access.
///
/// For example, an implementation containing:
///
/// ```ignore
/// #[instance_access(<args>)]
/// impl<T: NodeBound> T::Node
/// where
///     Self: InstanceBound
/// {
///     type Value = Self::Data;
///
///     fn get() -> Self::Value {
///         Self::value()
///     }
/// }
/// ```
///
/// together with access arguments describing a particular instance forms an
/// [`InstanceAccessArgs`] value. The implementation provides the
/// instance-bound associated items, while the access arguments determine which
/// instance representation is required when those items are lowered.
///
/// The implementation is then processed by the instance-access pipeline. Its
/// `Self` references and instance-bound associated accesses are replaced
/// according to the supplied access arguments, and the resulting associated
/// items are emitted as standalone functions, constants, or types.
///
/// For example, an access to:
///
/// ```ignore
/// Self::Value
/// ```
///
/// may be lowered into an explicit instance-bound representation such as:
///
/// ```ignore
/// <<T as NodeBound>::Node as InstanceBound<...>>::Value
/// ```
///
/// Thus, [`InstanceAccessArgs`] preserves both parts of the original macro
/// input required by the transformation: the implementation that defines the
/// instance-bound API and the access specification that determines how that
/// API is resolved.
#[derive(Clone, Debug)]
pub(crate) struct InstanceAccessArgs {
    pub(crate) item: ItemImpl,
    pub(crate) args: AccessArgs,
}

impl Extraction<TokenStream, TokenStream> for InstanceAccessArgs {
    fn raw_extract(
        from: &TokenStream,
        context: &TokenStream,
    ) -> Result<Self, proc_macro2::TokenStream> {
        let args = AccessArgs::checked_extract(&(context.clone().into()), &())?;

        let try_impl = parse::<ItemImpl>(from.clone());

        let item = match try_impl {
            Ok(i) => i,
            Err(_) => todo!("instance get impl parse fail"),
        };

        Ok(InstanceAccessArgs { item, args })
    }

    fn validate_extract(
        &self,
        _: &TokenStream,
        _: Option<&TokenStream>,
    ) -> Result<(), proc_macro2::TokenStream> {
        Ok(())
    }
}

// ===============================================================================
// `````````````````````````````` INSTANCE NODE ``````````````````````````````````
// ===============================================================================

/// Parsed input for the [`instance_node`](crate::instance_node) proc-macro.
///
/// An instance node can originate from either an instance trait or an
/// instance implementation. For example, a subscriber node may be declared
/// within an instance trait (see [`crate::node`]):
///
/// ```ignore
/// #[instance]
/// trait Logger {
///     #[instance_sub(leaf(...))]
///     type Events;
/// }
/// ```
///
/// while a publisher node may be declared within an instance implementation:
///
/// ```ignore
/// #[instance]
/// impl Logger for AppLogger {
///     #[instance_pub(leaf(...))]
///     type Events = AppEvents;
/// }
/// ```
///
/// These two input forms are represented by [`InstanceNodeArgs::Trait`] and
/// [`InstanceNodeArgs::Impl`] respectively. Preserving the original syntax as
/// separate variants allows the instance-node transformation to apply the
/// rules appropriate to the context in which the node is declared.
///
/// The input does not accept additional arguments at this level. Node-specific
/// arguments, when applicable, are extracted from the associated type and its
/// marker attribute annotations during the subsequent node transformation.
#[derive(Clone, Debug)]
pub(crate) enum InstanceNodeArgs {
    Trait(ItemTrait),
    Impl(ItemImpl),
}

impl Extraction<TokenStream, TokenStream> for InstanceNodeArgs {
    fn raw_extract(
        from: &TokenStream,
        context: &TokenStream,
    ) -> Result<Self, proc_macro2::TokenStream> {
        if !context.is_empty() {
            return Err(ProcParseErr::AvoidArguments {
                args: context.clone().into(),
            }
            .into());
        }

        let try_trait = parse::<ItemTrait>(from.clone());
        let try_impl = parse::<ItemImpl>(from.clone());

        let item = match (try_trait, try_impl) {
            (Ok(_), Ok(_)) => return Err(ParseBug::SynParseInconsistent {}.into()),
            (Ok(trait_of), Err(_)) => InstanceNodeArgs::Trait(trait_of),
            (Err(_), Ok(impl_of)) => InstanceNodeArgs::Impl(impl_of),
            (Err(e_ident), Err(e_ints)) => {
                let mut either_or_err: Error = ProcParseErr::JustTraitOrImpl {
                    tokens: from.clone().into(),
                }
                .into();
                either_or_err.combine(e_ident);
                either_or_err.combine(e_ints);

                return Err(either_or_err.to_compile_error().into());
            }
        };
        Ok(item)
    }

    fn validate_extract(
        &self,
        _: &TokenStream,
        _: Option<&TokenStream>,
    ) -> Result<(), proc_macro2::TokenStream> {
        Ok(())
    }
}

// ===============================================================================
// `````````````````````````` INSTANCE DIRECT ACCESS `````````````````````````````
// ===============================================================================

/// Parsed input for the [`instance_direct_access`](crate::instance_direct_access) proc-macro.
///
/// For example, an access request may specify a leaf instance together with
/// an associated item:
///
/// ```ignore
/// #[instance_direct_access(leaf(index(0), ident(value)))]
/// <InstanceType as InstanceTrait<..>>::VALUE
/// ```
///
/// The [`LeafAccess`] supplies the fixed counter indexes and identifiers used
/// to recover the typenum counter for the requested instance. The expression
/// supplies the instance-trait associated item whose path is to be resolved.
///
/// Direct access supports both concrete publisher types and associated
/// subscriber instance types. For example:
///
/// ```ignore
/// <InstanceType as InstanceTrait<..>>::VALUE
///
/// <<Provider as Subscriber>::GetInstance as InstanceTrait<..>>::VALUE
/// ```
///
/// In the first form, the concrete implementing type is used to recover the
/// instance counter directly. In the second form, the associated instance type
/// is resolved through its hidden global lookup before the counter is
/// recovered.
///
/// The parsed expression is subsequently traversed by the direct-access
/// transformation, which preserves supported surrounding expression forms
/// while replacing the instance-trait counter arguments with the resolved
/// typenum counter.
#[derive(Debug, Clone)]
pub(crate) struct InstanceDirectAccessArgs {
    pub(crate) leaf: LeafAccess,
    pub(crate) expr: Expr,
}

impl Parse for InstanceDirectAccessArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let global_span = input.span();
        let attrs = Attribute::parse_outer(input)?;
        if attrs.is_empty() {
            return Err(ProcParseErr::DirectAccessExpected { span: global_span }.into());
        }

        if attrs.len() > 1 {
            return Err(ProcParseErr::DirectAccessExpected { span: global_span }.into());
        }

        let attr = &attrs[0];

        let key_err = Err(KeysError::ExpectedLeaf { span: attr.span() }.into());

        let tokens = match &attr.meta {
            syn::Meta::List(meta) => &meta.tokens,
            _ => return key_err,
        };

        let args = match AccessArgs::checked_extract(tokens, &()) {
            Ok(args) => args,
            Err(_) => return key_err,
        };

        let AccessArgs::Leaf(leaf) = args else {
            return key_err;
        };

        let expr = input.parse::<Expr>()?;

        if !input.is_empty() {
            return Err(ProcParseErr::TrailingTokens { span: input.span() }.into());
        }

        Ok(Self { leaf, expr })
    }
}

impl Extraction<proc_macro2::TokenStream> for InstanceDirectAccessArgs {
    fn raw_extract(
        from: &proc_macro2::TokenStream,
        _: &(),
    ) -> Result<Self, proc_macro2::TokenStream> {
        syn::parse2::<Self>(from.clone()).map_err(|e| e.to_compile_error())
    }

    fn validate_extract(
        &self,
        _: &proc_macro2::TokenStream,
        _: Option<&()>,
    ) -> Result<(), proc_macro2::TokenStream> {
        Ok(())
    }
}

// ===============================================================================
// ````````````````````````````` INSTANCE GETTER `````````````````````````````````
// ===============================================================================

/// Parsed input for the [`instance_get_access`](crate::instance_get_access) proc-macro.
///
/// Represents the complete input required for instance-getter access.
///
/// For example, a getter request may provide an instance-trait associated
/// function together with the dynamic inputs required by the generated getter:
///
/// ```ignore
/// <Provider as InstanceTrait<'a, 0, T>>::Assoc::get(arg), b"users", key
/// ```
///
/// The [`Expr`] identifies the instance-trait associated item to be transformed,
/// while the [`BStrInputList`] supplies the byte-string and identifier inputs
/// that carry the requested instance information through the generated getter.
///
/// For example, the expression:
///
/// ```ignore
/// <Provider as InstanceTrait<'a, 0, T>>::Assoc::get(arg)
/// ```
///
/// is transformed into its standalone getter representation, while the getter
/// inputs are propagated through the transformation and appended to the
/// associated function call as trailing arguments:
///
/// ```ignore
/// get::<'a, Provider, 0, T>(arg, b"users", key)
/// ```
///
/// The same getter inputs are propagated through supported surrounding
/// expression forms such as tuples, references, `return` expressions, and `?`
/// expressions. They are only materialized as function arguments when the
/// transformed associated item is invoked as a function; path-only accesses
/// consume the inputs as part of getter resolution without receiving a
/// function argument list.
#[derive(Debug, Clone)]
pub(crate) struct InstanceGetterArgs {
    pub(crate) expr: Expr,
    pub(crate) input: BStrInputList,
}

impl Parse for InstanceGetterArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let expr = input.parse::<Expr>()?;

        let list = if input.peek(Comma) {
            input.parse::<Comma>()?;
            input.parse::<BStrInputList>()?
        } else {
            BStrInputList::default()
        };

        if !input.is_empty() {
            return Err(ProcParseErr::TrailingTokens { span: input.span() }.into());
        }

        Ok(Self { expr, input: list })
    }
}

impl Extraction<proc_macro2::TokenStream> for InstanceGetterArgs {
    fn raw_extract(
        from: &proc_macro2::TokenStream,
        _: &(),
    ) -> Result<Self, proc_macro2::TokenStream> {
        syn::parse2::<Self>(from.clone()).map_err(|e| e.to_compile_error())
    }

    fn validate_extract(
        &self,
        _: &proc_macro2::TokenStream,
        _: Option<&()>,
    ) -> Result<(), proc_macro2::TokenStream> {
        Ok(())
    }
}
