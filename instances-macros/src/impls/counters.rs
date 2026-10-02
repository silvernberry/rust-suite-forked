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
// ````````````````````````` INSTANCE COUNTER ARGUMENTS ``````````````````````````
// ===============================================================================

//! This module extracts and validates impl-side const-generic arguments
//! used as instance counters.
//!
//! Instance-counter arguments provide the concrete counter values supplied
//! by an implementation for the counter parameters declared on the
//! corresponding instance trait.
//!
//! ## Why this exists
//!
//! Trait-side instance counters are identified exclusively by their
//! position within the trait generic parameter list.
//!
//! For example:
//!
//! ```ignore
//! trait Example<const A: u8, const B: u8> {}
//!
//! impl Example<4, 7> for MyType {}
//! ```
//!
//! Although the trait names its counters `A` and `B`, the implementation
//! does not reference those identifiers.
//!
//! Instead, the implementation supplies values purely by position:
//!
//! ```text
//! index 0 -> 4
//! index 1 -> 7
//! ```
//!
//! Because of this, impl-side counter extraction must resolve concrete
//! const-generic arguments into positional metadata that can be matched
//! against trait-side counter definitions.
//!
//! ## What this module provides
//!
//! This module:
//!
//! - extracts impl-side instance-counter arguments,
//! - validates supported literal const arguments,
//! - normalizes arguments into positional metadata,
//! - and preserves user-defined counter ordering.
//!
//! Extracted arguments are represented as [`CounterArg`] values collected
//! into [`CounterArgs`].
//!
//! ## Extraction Modes
//!
//! Counter arguments can be selected in two ways:
//!
//! ### Index-based extraction
//!
//! ```ignore
//! #[instance(0, 2)]
//! ```
//!
//! Resolves counter arguments by generic argument position within the
//! implemented trait path.
//!
//! Example:
//!
//! ```ignore
//! impl Example<4, T, 7> for MyType {}
//! ```
//!
//! extracts:
//!
//! ```ignore
//! 4, 7
//! ```
//!
//! from positions `0` and `2`.
//!
//! ### Default extraction
//!
//! When no explicit selection is provided, leading contiguous integer
//! const-generic arguments are used automatically.
//!
//! ```ignore
//! impl Example<4, 7, T> for MyType {}
//! ```
//!
//! extracts:
//!
//! ```ignore
//! 4, 7
//! ```
//!
//! Extraction stops at the first non-integer const argument or non-const
//! generic argument.
//!
//! ## Ordering Semantics
//!
//! Counter ordering is semantic and user-controlled.
//!
//! The extracted order does not need to match the textual order of the
//! implemented trait arguments.
//!
//! Example:
//!
//! ```ignore
//! impl Example<4, T, 7> for MyType {}
//!
//! #[instance(2, 0)]
//! ```
//!
//! produces:
//!
//! ```ignore
//! [7, 4]
//! ```
//!
//! even though `4` appears first in the impl.
//!
//! Ordering is preserved exactly as requested by the user because later
//! instance-resolution stages may assign semantic meaning to counter order.
//!
//! ## Supported Counter Arguments
//!
//! Only literal integer const-generic arguments are supported:
//!
//! ```ignore
//! impl Example<4, 7> for MyType {}
//! ```
//!
//! Supported:
//!
//! - integer literals (`4`, `7`, `42`)
//!
//! Rejected:
//!
//! ```ignore
//! impl<const N: usize> Example<N> for MyType {}
//! impl Example<{1 + 2}> for MyType {}
//! impl Example<OTHER_CONST> for MyType {}
//! ```
//!
//! because instance counters must resolve to concrete integer values during
//! extraction.

// ===============================================================================
// ``````````````````````````````````` IMPORTS ```````````````````````````````````
// ===============================================================================

// --- Proc-Suite ---
use proc_suite::{DuplicateCheck, IntList, misc::*};

// --- Proc-Macro crates ---
use proc_macro2::TokenStream;
use syn::{
    Expr, GenericArgument, ItemImpl, Lit, LitInt, parse_quote, punctuated::Punctuated, token::Comma,
};

// --- Local Crate ---
use crate::{
    Extraction, Utilization,
    impls::{
        errors::{CounterArgBugs, CounterArgError, CounterArgExtractionError},
        utils::*,
    },
};

// ===============================================================================
// ``````````````````````````````````` STRUCTS ```````````````````````````````````
// ===============================================================================

/// Describes an **impl-side** const-generic argument used for instance counters.
///
/// Instance traits declare counter spaces through const-generic parameters.
/// Implementations populate those counter spaces by supplying concrete
/// integer const arguments.
///
/// Unlike trait-side counter definitions, impls do not identify counters by
/// name. Instead, each counter value is associated with a trait-declared
/// counter exclusively through its positional index within the implemented
/// trait argument list.
///
/// This type records:
/// - the zero-based generic argument index within the implemented trait path
/// - the literal integer value supplied at that position
///
/// The stored `generic_index` is the only mechanism used to associate an
/// impl-supplied counter value with a trait-declared instance counter.
///
/// ## Example
///
/// ```ignore
/// trait Example<T, const A: u8, U, const B: u8> {}
///
/// impl<T, U> Example<T, 4, U, 7> for MyType {}
/// ```
///
/// In the implementation above:
///
/// - `4` has `generic_index == 1`
/// - `7` has `generic_index == 3`
///
/// These indices correspond directly to the trait-side counter parameters,
/// allowing the instance system to bind values to counters without relying
/// on parameter identifiers.
///
/// Renaming `A` or `B` in the trait definition has no effect on instance
/// resolution as long as the positional indices remain unchanged.
#[derive(Clone, Debug)]
pub(crate) struct CounterArg {
    /// The zero-based index of the trait const-generic parameter this argument
    /// corresponds to.
    ///
    /// This must match the `generic_index` recorded on the trait side and is the
    /// sole selector used to bind this value to a specific instance counter.
    pub(super) generic_index: usize,

    /// The literal integer value supplied by the `impl`.
    ///
    /// This represents the actual instance-counter value and is used directly
    /// during metadata generation, affiliate-counter resolution, hash
    /// construction, and last-instance enforcement.
    pub(super) const_lit: LitInt,
}

/// A vector of [`CounterArg`] values describing all **instance-counter**
/// const-generic arguments supplied by an implementation.
///
/// Each [`CounterArg`] represents a single const-generic argument from the
/// implemented trait path and is identified exclusively by its
/// zero-based [`CounterArg::generic_index`].
///
/// ## Ordering semantics
///
/// The order of this vector is **intentional and user-defined**.
/// It does **not** need to be ascending by `generic_index`, nor does it
/// necessarily mirror the textual order of the trait arguments appearing
/// in the implementation.
///
/// When instance-counter arguments are interleaved with other generic
/// arguments, this vector preserves the order in which the user intends
/// those counters to participate in instance resolution.
///
/// Expansion logic must therefore rely on `generic_index` for stable
/// trait–impl association and treat the vector order as semantically
/// meaningful.
///
/// ## Examples
///
/// ```ignore
/// trait MyTrait<const A: u8, T, const B: u8, const C: u8> {}
///
/// impl<T> MyTrait<4, T, 7, 9> for MyType {}
/// ```
///
/// If the user selects counters in the order `B, A`, the collected
/// arguments will be:
///
/// ```ignore
/// CounterArgs [
///   CounterArg { generic_index: 2, const_lit: 7 },
///   CounterArg { generic_index: 0, const_lit: 4 },
/// ]
/// ```
///
/// Likewise, when counters are interleaved with other arguments:
///
/// ```ignore
/// trait MyTrait<T, const X: u8, U, const Y: u8> {}
///
/// impl<T, U> MyTrait<T, 3, U, 8> for MyType {}
/// ```
///
/// And the user specifies the instance order `Y, X`, the resulting vector
/// preserves that intent:
///
/// ```ignore
/// CounterArgs [
///   CounterArg { generic_index: 3, const_lit: 8 },
///   CounterArg { generic_index: 1, const_lit: 3 },
/// ]
/// ```
pub(crate) type CounterArgs = Vec<CounterArg>;

/// Borrowed slice view of extracted impl-side instance-counter arguments.
pub(crate) type CounterArgsSlice<'a> = &'a [CounterArg];

// ===============================================================================
// ```````````````````````` COUNTERS ARGUMENTS EXTRACTION ````````````````````````
// ===============================================================================

impl Extraction<ItemImpl, Option<&IntList>> for CounterArgs {
    fn raw_extract(from: &ItemImpl, context: &Option<&IntList>) -> Result<Self, TokenStream> {
        match context {
            Some(ints) => <Self as Extraction<ItemImpl, IntList>>::checked_extract(from, ints),
            None => <Self as Extraction<ItemImpl>>::checked_extract(from, &()),
        }
    }

    fn validate_extract(
        &self,
        from: &ItemImpl,
        context: Option<&Option<&IntList>>,
    ) -> Result<(), TokenStream> {
        let Some(args) = context else {
            return <Self as Extraction<ItemImpl>>::validate_extract(&self, from, None);
        };
        match args {
            Some(ints) => {
                <Self as Extraction<ItemImpl, IntList>>::validate_extract(&self, from, Some(ints))
            }
            None => <Self as Extraction<ItemImpl>>::validate_extract(&self, from, None),
        }
    }
}

impl Extraction<ItemImpl, IntList> for CounterArgs {
    fn validate_context(context: &IntList) -> Result<(), TokenStream> {
        Ok(context.duplicate_check(Some(CounterArgError::DuplicateCounterIndexes.into()))?)
    }

    fn raw_extract(from: &ItemImpl, context: &IntList) -> Result<Self, TokenStream> {
        if context.ints.is_empty() {
            return Err(CounterArgExtractionError::EmptyCountersIndexes {
                ints: context.clone(),
            }
            .into());
        }
        let mut collect = Vec::new();
        let args = ImplTraitGenArgs::checked_utilize(from, &())?.0;

        let trait_ident = ImplTraitIdent::checked_utilize(from, &())?.0.clone();
        for int in &context.ints {
            let index = parse_pos_usize(int)?;
            let Some(arg) = args.get(index) else {
                let trait_generics = ImplTraitGenArgs::checked_utilize(from, &())?.0.clone();
                return Err(CounterArgExtractionError::OutOfBoundsGenericIndex {
                    gen_idx: index,
                    given_idx: int.clone(),
                    trait_ident,
                    trait_generics,
                }
                .into());
            };
            let GenericArgument::Const(c) = arg else {
                return Err(CounterArgExtractionError::ArgumentNotConstGeneric {
                    gen_idx: index,
                    given_idx: int.clone(),
                    trait_ident,
                    found: arg.clone(),
                }
                .into());
            };

            let Expr::Lit(l) = c else {
                return Err(CounterArgExtractionError::ArgumentNotConstGenericExpr {
                    gen_idx: index,
                    given_idx: int.clone(),
                    trait_ident,
                    found: c.clone(),
                }
                .into());
            };
            let Lit::Int(i) = &l.lit else {
                return Err(
                    CounterArgExtractionError::ArgumentNotConstGenericExprLitInt {
                        gen_idx: index,
                        given_idx: int.clone(),
                        trait_ident,
                        found: l.lit.clone(),
                    }
                    .into(),
                );
            };
            collect.push(CounterArg {
                generic_index: index,
                const_lit: i.clone(),
            });
        }

        if collect.is_empty() {
            return Err(
                CounterArgExtractionError::EmptyUtilizedCounterArgsFromList {
                    ints: context.clone(),
                }
                .into(),
            );
        }

        Ok(collect)
    }

    fn validate_extract(
        &self,
        from: &ItemImpl,
        context: Option<&IntList>,
    ) -> Result<(), TokenStream> {
        let Some(ints) = context else {
            return <Self as Extraction<ItemImpl>>::validate_extract(&self, from, None);
        };
        let args = ImplTraitGenArgs::checked_utilize(from, &())?.0;

        for int in &ints.ints {
            let idx = parse_pos_usize(int)?;
            let mut found = false;
            for (i, _) in args.iter().enumerate() {
                if i == idx {
                    found = true
                }
            }
            if !found {
                return Err(
                    CounterArgBugs::CounterIndexesExtractionViaGenArgsInconsistent {}.into(),
                );
            }
            found = false;
            for c in self {
                let i = c.generic_index;
                if i == idx {
                    found = true
                }
            }
            if !found {
                return Err(CounterArgBugs::CounterIndexesExtractionInconsistent {}.into());
            }
        }

        if self.len() > args.len() {
            return Err(CounterArgBugs::ExtractedCounterArgsLenAboveActualGenArgs {}.into());
        };

        for counter in self {
            let idx = counter.generic_index;
            let Some(arg) = args.get(idx) else {
                return Err(CounterArgBugs::ExtractedArgumentOutOfBoundsGenericIndex {}.into());
            };
            let GenericArgument::Const(c) = arg else {
                return Err(CounterArgBugs::ExtractedArgumentNotConstGeneric {}.into());
            };

            let Expr::Lit(l) = c else {
                return Err(CounterArgBugs::ExtractedArgumentNotConstGenericExpr {}.into());
            };
            let Lit::Int(i) = &l.lit else {
                return Err(CounterArgBugs::ExtractedArgumentNotConstGenericExprLitInt {}.into());
            };
            if *i != counter.const_lit {
                return Err(CounterArgBugs::ExtractedCounterArgumentIsNotSame {}.into());
            }
        }

        Ok(())
    }
}

impl Extraction<ItemImpl> for CounterArgs {
    fn raw_extract(from: &ItemImpl, _: &()) -> Result<Self, TokenStream> {
        let mut collect = Vec::new();
        let args = ImplTraitGenArgs::checked_utilize(from, &())?.0;
        let trait_ident = ImplTraitIdent::checked_utilize(from, &())?.0.clone();
        if args.is_empty() {
            return Err(CounterArgExtractionError::TraitArgsNeedsLeadConstGenerics {
                trait_generics: args.clone(),
                trait_ident,
            }
            .into());
        }

        for (index, arg) in args.iter().enumerate() {
            let GenericArgument::Const(c) = arg else {
                break;
            };

            let Expr::Lit(l) = c else {
                break;
            };
            let Lit::Int(i) = &l.lit else {
                break;
            };
            collect.push(CounterArg {
                generic_index: index,
                const_lit: i.clone(),
            });
        }
        if collect.is_empty() {
            let path = ImplTraitPath::checked_utilize(from, &())?.0;
            return Err(
                CounterArgExtractionError::CannotDeriveCounterArgsImplicitly {
                    trait_ident,
                    trait_path: path.clone(),
                }
                .into(),
            );
        }
        Ok(collect)
    }

    fn validate_extract(&self, from: &ItemImpl, _: Option<&()>) -> Result<(), TokenStream> {
        let may_ints = self.iter().map(|c| &c.generic_index);
        let mut ints = Punctuated::<LitInt, Comma>::new();
        for (i, int) in may_ints.enumerate() {
            if i != *int {
                return Err(CounterArgBugs::ExtractionViaLeadConstGenericsInconsistent {}.into());
            }
            ints.push(parse_quote!(#i));
        }

        <Self as Extraction<ItemImpl, IntList>>::validate_extract(
            &self,
            from,
            Some(&IntList { ints: ints }),
        )?;
        Ok(())
    }
}
