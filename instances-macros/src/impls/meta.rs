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
// ``````````````````````` INSTANCE IMPL META (VALIDATION) ```````````````````````
// ===============================================================================

//! Generates implementation-side metadata and validation checkpoints
//! used by the instance-trait proc-macro pipeline.
//!
//! This module is the implementation-side counterpart of
//! [`crate::traits::meta`].
//!
//! While trait-side expansion reflects instance-counter metadata into
//! hidden associated items, implementation-side expansion reconstructs
//! corresponding metadata and validates that both sides remain
//! synchronized.
//!
//! ## Example
//!
//! Given:
//!
//! ```ignore
//! #[trait_instance(A, C)]
//! trait Example<
//!     const A: u8,
//!     const B: usize,
//!     const C: u8,
//! > {
//!     type InstanceCounter;
//! }
//!
//! #[instance_impl(0, 2)]
//! impl Example<1, 10, 3> for MyType {
//!     type InstanceCounter = u8;
//! }
//! ```
//!
//! After typenum transformation performed by
//! [`InstanceTraitTypeNumCounters`](crate::traits::typenum::InstanceTraitTypeNumCounters)
//! and
//! [`InstanceImplTypeNumCounters`](crate::impls::typenum::InstanceImplTypeNumCounters):
//!
//! ```ignore
//! trait Example<__TypeNumCounters, const B: usize> {
//!     type InstanceCounter;
//! }
//!
//! impl Example<(U1, U3), 10> for MyType {
//!     type InstanceCounter = u8;
//! }
//! ```
//!
//! Trait-side metadata expansion generates:
//!
//! ```ignore
//! trait Example<__TypeNumCounters, const B: usize> {
//!     #[doc(hidden)]
//!     const __COUNTERS_GENERICS_META:
//!         [(usize, &'static str); 2] = [
//!             (0, "A"),
//!             (2, "C"),
//!         ];
//!
//!     #[doc(hidden)]
//!     const __COUNTERS_GENERIC_INDEXES_META:
//!         &'static [usize] = &[0, 2];
//!
//!     #[doc(hidden)]
//!     const __COUNTERS_LEN_META: usize = 2;
//!
//!     #[doc(hidden)]
//!     const __COUNTER_INDEX_0: u8;
//!
//!     #[doc(hidden)]
//!     const __COUNTER_INDEX_2: u8;
//!
//!     #[doc(hidden)]
//!     type GlobalTerminalAssoc: TerminalAccess<Self>;
//!
//!     #[doc(hidden)]
//!     type GlobalTerminalExact:
//!         Exact<
//!             <Self::GlobalTerminalAssoc as TerminalAccess<Self>>::Terminal
//!         >;
//!
//!     #[doc(hidden)]
//!     type SelfTerminalAssoc:
//!         Example<Self::GlobalTerminalExact, 10>;
//! }
//! ```
//!
//! This module then generates implementation-side metadata:
//!
//! ```ignore
//! impl Example<(U1, U3), 10> for MyType {
//!     type InstanceCounter = u8;
//!
//!     #[doc(hidden)]
//!     const __COUNTERS_GENERIC_INDEXES_META:
//!         &'static [usize] = &[0, 2];
//!
//!     #[doc(hidden)]
//!     const __COUNTERS_LEN_META: usize = 2;
//!
//!     #[doc(hidden)]
//!     const __COUNTER_INDEX_0: u8 = 1;
//!
//!     #[doc(hidden)]
//!     const __COUNTER_INDEX_2: u8 = 3;
//!
//!     #[doc(hidden)]
//!     const __COUNTERS_GENERICS_CHECKER: () = {
//!         /* validates generic metadata */
//!     };
//!
//!     #[doc(hidden)]
//!     type GlobalTerminalAssoc = instances::Global;
//!
//!     #[doc(hidden)]
//!     type GlobalTerminalExact =
//!         <instances::Global as TerminalAccess<Self>>::Terminal;
//!
//!     #[doc(hidden)]
//!     type SelfTerminalAssoc = MyType;
//! }
//! ```
//!
//! where:
//!
//! - [`ImplCountersGenericsIndexesMeta`] reflects the generic indexes
//!   of extracted implementation-side counters (`&[0, 2]`).
//!
//! - [`ImplCountersLenMeta`] reflects the total number of extracted
//!   implementation-side counters (`2`).
//!
//! - [`ImplOriginalCounterConst`] reconstructs the original counter
//!   values (`1` and `3`) of type from `type InstanceCounter = <type>`
//!   after typenum transformation has erased the original generic arguments.
//!
//! - [`ImplCountersGenericsChecker`] validates that implementation-side
//!   counter metadata matches the metadata reflected by
//!   [`CountersGenericsMeta`],
//!   [`CountersGenericsIndexesMeta`],
//!   and [`CountersLenMeta`].
//!
//! - [`GlobalTerminalAssocs`] provides the global instance, its exact terminal
//!   counter, and the implementation itself as hidden associated types,
//!   supplying the impl-side meta information required by sum types.
//!
//! Together, trait-side reflection and implementation-side validation,
//! including the global and terminal-instance boundaries provided by
//! [`GlobalTerminalAssocs`], form a deterministic synchronization protocol
//! that allows instance semantics to remain available even after typenum
//! transformation has erased the original counter generics.
//!
//! The generated metadata is consumed exclusively by proc-macro
//! expansion and is not intended for direct use by library users.

// ===============================================================================
// ``````````````````````````````````` IMPORTS ```````````````````````````````````
// ===============================================================================

// --- Local-crate ---
use crate::{
    Extension, Extraction, Insertion, Instance, Transformation, Utilization,
    impls::{
        counters::*,
        errors::{MetaErrors, MetaImplItemsBugs},
        utils::*,
    },
    traits::{
        affiliates::{BackAffiliateCounters, BackAffiliateInstance},
        meta::*,
        sum::{GlobalTerminalAssoc, GlobalTerminalExact, SelfTerminalAssoc},
    },
};

// --- Proc-macro Utilties ---
use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::{Expr, ImplItem, ImplItemConst, ImplItemType, ItemImpl, LitInt, Type, parse_quote};

// --- Proc Suite ---
use proc_suite::{SupportCrate, misc::*};

// ===============================================================================
// `````````````````````````````` INSTANCE IMPL META `````````````````````````````
// ===============================================================================

/// Aggregates and executes all impl-side metadata generation and
/// validation phases used by the instance-trait proc-macro pipeline.
///
/// This phase acts as the implementation-side orchestration layer
/// responsible for applying:
///
/// - impl-side counter generic-index reflection via
///   [`ImplCountersGenericsIndexesMeta`],
/// - impl-side counter-count reflection via [`ImplCountersLenMeta`],
/// - counter generic synchronization via
///   [`ImplCountersGenericsChecker`],
/// - and original counter reconstruction via
///   [`ImplOriginalCounterConst`].
/// - and impl-side global and terminal-instance boundaries via
///   [`GlobalTerminalAssocs`].
///
/// Conceptually, this phase converts extracted implementation-side
/// counter metadata into:
///
/// - validation checkpoints,
/// - reflected implementation metadata,
/// - synchronization contracts,
/// - and reconstructed counter constants.
/// - and global/terminal-instance boundaries used by sum types.
///
/// Together with the trait-side metadata phase
/// ([`InstanceTraitMeta`]),
/// this forms the complete reflection and validation layer of the
/// instance-trait proc-macro pipeline.
///
/// This phase executes after typenum transformation performed by
/// [`InstanceImplTypeNumCounters`](crate::impls::typenum::InstanceImplTypeNumCounters)
/// and validates metadata reflected by
/// [`InstanceTraitMeta`].
///
/// The [`GlobalTerminalAssocs`] phase exposes the global instance, its exact
/// terminal counter, and the implementation itself as associated types,
/// providing the impl-side meta information required by the trait-side
/// sum-type machinery.
///
/// Each phase remains independently composable and validatable through
/// the proc-macro pipeline trait system.
#[derive(Debug, Clone)]
pub(crate) struct InstanceImplMeta;

impl<'a> Transformation<ItemImpl, CounterArgsSlice<'a>> for InstanceImplMeta {
    fn raw_transform(
        &self,
        transform: &mut ItemImpl,
        context: &CounterArgsSlice,
    ) -> Result<(), TokenStream> {
        ImplCountersGenericsIndexesMeta::checked_extend(
            &ImplCountersGenericsIndexesMeta,
            transform,
            context,
        )?;
        ImplCountersLenMeta::checked_extend(&ImplCountersLenMeta, transform, context)?;
        ImplOriginalCounterConst::checked_extend(&ImplOriginalCounterConst, transform, context)?;
        ImplCountersGenericsChecker::checked_extend(
            &ImplCountersGenericsChecker,
            transform,
            context,
        )?;
        GlobalTerminalAssocs::checked_transform(&GlobalTerminalAssocs, transform, &())?;
        Ok(())
    }

    fn validate_transform(
        &self,
        transform: &ItemImpl,
        context: Option<&CounterArgsSlice<'a>>,
    ) -> Result<(), TokenStream> {
        ImplCountersGenericsIndexesMeta::validate_extend(
            &ImplCountersGenericsIndexesMeta,
            None,
            transform,
            context,
        )?;
        ImplCountersLenMeta::validate_extend(&ImplCountersLenMeta, None, transform, context)?;
        ImplOriginalCounterConst::validate_extend(
            &ImplOriginalCounterConst,
            None,
            transform,
            context,
        )?;
        ImplCountersGenericsChecker::validate_extend(
            &ImplCountersGenericsChecker,
            None,
            transform,
            context,
        )?;
        GlobalTerminalAssocs::validate_transform(&GlobalTerminalAssocs, transform, None)?;
        Ok(())
    }
}

// ===============================================================================
// ``````````````````````` COUNTERS GENERICS INDEXES META ````````````````````````
// ===============================================================================

/// Extends an instance implementation with hidden metadata describing
/// the generic indexes of all extracted instance counters.
///
/// Although [`CountersGenericsMeta`] reflects this information on the
/// trait side, implementation-side validation requires its own view of
/// the extracted counter positions.
///
/// This phase generates a hidden associated constant containing the
/// generic indexes of all extracted instance counters as a slice.
///
/// ## Example
///
/// Given:
///
/// ```ignore
/// #[instance_impl(0, 2)]
/// impl Example<1, 2, 3> for MyType {}
/// ```
///
/// where extraction determines:
///
/// ```text
/// 1 -> generic index 0
/// 3 -> generic index 2
/// ```
///
/// this extension generates:
///
/// ```ignore
/// impl Example<1, 2, 3> for MyType {
///     #[doc(hidden)]
///     const __COUNTERS_GENERIC_INDEXES_META:
///         &'static [usize] = &[0, 2];
/// }
/// ```
///
/// This metadata is later consumed by
/// [`ImplCountersGenericsChecker`] to validate that:
///
/// - the implementation identified the correct instance counters,
/// - the counter ordering matches trait-side expectations,
/// - and impl-side expansion remains synchronized with reflected
///   trait metadata.
///
/// The generated identifier is derived from
/// [`CountersGenericsIndexesMeta`] so that both trait-side and impl-side
/// expansion refer to the same metadata contract.
#[derive(Debug, Clone)]
pub(super) struct ImplCountersGenericsIndexesMeta;

impl<'a> Insertion<ItemImpl, CounterArgsSlice<'a>> for ImplItemConst {
    fn raw_insert(&self, to: &mut ItemImpl, _: &CounterArgsSlice) -> Result<(), TokenStream> {
        to.items.push(ImplItem::Const(self.clone()));
        Ok(())
    }

    fn validate_inserted(
        _of: Option<&Self>,
        _to: &ItemImpl,
        _context: Option<&CounterArgsSlice<'a>>,
    ) -> Result<(), TokenStream> {
        Ok(())
    }
}

impl<'a> Extension<ImplItemConst, ItemImpl, CounterArgsSlice<'a>>
    for ImplCountersGenericsIndexesMeta
{
    fn raw_extend(
        &self,
        _towards: &ItemImpl,
        context: &CounterArgsSlice<'a>,
    ) -> Result<ImplItemConst, TokenStream> {
        let ident = gen_const_ident::<CountersGenericsIndexesMeta>();
        let ty = parse_quote!(&'static [usize]);
        let expr = counter_generics_indexes_meta_expr(context)?;

        let item = ImplItemConst {
            attrs: proc_suite::internal_code(),
            ident,
            expr,
            ty,
            vis: syn::Visibility::Inherited,
            defaultness: None,
            const_token: Default::default(),
            generics: Default::default(),
            colon_token: Default::default(),
            eq_token: Default::default(),
            semi_token: Default::default(),
        };
        Ok(item)
    }

    fn validate_extend(
        &self,
        item: Option<&ImplItemConst>,
        towards: &ItemImpl,
        context: Option<&CounterArgsSlice<'a>>,
    ) -> Result<(), TokenStream> {
        let extracted;
        let counters = match context {
            Some(c) => c,
            None => {
                extracted = CounterArgs::checked_extract(towards, &None)?;
                extracted.as_slice().as_ref().as_ref()
            }
        };

        let ident = gen_const_ident::<CountersGenericsIndexesMeta>();
        let ty: Type = parse_quote!(&'static [usize]);
        let expr = counter_generics_indexes_meta_expr(counters)?;

        validate_impl_const! {
            item: item,
            towards: towards,
            ident: ident,
            ty: ty,
            expr: expr,
            errors: {
                not_found: MetaImplItemsBugs::ImplCountersGenericsIndexesMetaNotFound {},
                wrong_ident: MetaImplItemsBugs::ImplCountersGenericsIndexesMetaWrongIdent {},
                wrong_ty: MetaImplItemsBugs::ImplCountersGenericsIndexesMetaInvalidType {},
                has_generics: MetaImplItemsBugs::ImplCountersGenericsIndexesMetaHasGenerics {},
                invalid_expr: MetaImplItemsBugs::ImplCountersGenericsIndexesMetaInvalidExpr {},
            }
        }
    }
}

/// Builds the impl-side counter-generic-index metadata expression.
///
/// Example:
///
/// ```ignore
/// &[0, 2]
/// ```
fn counter_generics_indexes_meta_expr<'a>(
    counters: CounterArgsSlice<'a>,
) -> Result<Expr, TokenStream> {
    let mut collect = Vec::new();
    for c in counters.iter() {
        let gen_idx = c.generic_index;
        let expr: LitInt = parse_quote!(#gen_idx);
        collect.push(expr);
    }
    if collect.is_empty() {
        return Err(
            MetaImplItemsBugs::CounterArgsAreEmptyToProvideForGenericsIndexesMeta {}.into(),
        );
    }
    let expr = parse_quote!(
        &[#(#collect),*]
    );
    Ok(expr)
}

// ===============================================================================
// ```````````````````````````` COUNTERS LENGTH META `````````````````````````````
// ===============================================================================

/// Extends an instance implementation with hidden metadata representing
/// the total number of extracted instance counters.
///
/// Since [`ImplCountersGenericsIndexesMeta`] reflects all extracted
/// counter generic indexes as a slice, this phase provides the
/// corresponding expected slice length.
///
/// In particular, [`ImplCountersGenericsChecker`] uses this metadata
/// together with [`ImplCountersGenericsIndexesMeta`] and the reflected
/// trait-side metadata [`CountersGenericsMeta`] to validate:
///
/// - that the implementation identified the correct number of instance
///   counters,
/// - that counter metadata remains structurally consistent,
/// - and that impl-side expansion participates in the expected pipeline.
///
/// ## Example
///
/// Given:
///
/// ```ignore
/// #[instance_impl(0, 2)]
/// impl Example<1, 2, 3> for MyType {}
/// ```
///
/// where extraction determines:
///
/// ```text
/// 1 -> counter
/// 3 -> counter
/// ```
///
/// this extension generates:
///
/// ```ignore
/// impl Example<1, 2, 3> for MyType {
///     #[doc(hidden)]
///     const __COUNTERS_LEN_META: usize = 2;
/// }
/// ```
///
/// This metadata is later consumed by
/// [`ImplCountersGenericsChecker`] to validate counter-count
/// consistency against trait-side expectations.
///
/// The generated identifier is derived from [`CountersLenMeta`] so that
/// both trait-side and impl-side expansion refer to the same metadata
/// contract.
#[derive(Debug, Clone)]
pub(super) struct ImplCountersLenMeta;

impl<'a> Extension<ImplItemConst, ItemImpl, CounterArgsSlice<'a>> for ImplCountersLenMeta {
    fn raw_extend(
        &self,
        _: &ItemImpl,
        context: &CounterArgsSlice<'a>,
    ) -> Result<ImplItemConst, TokenStream> {
        let ident = gen_const_ident::<CountersLenMeta>();
        let ty = parse_quote!(usize);
        let len = context.len();
        let expr = parse_quote!(#len);

        let item = ImplItemConst {
            attrs: proc_suite::internal_code(),
            ident,
            expr,
            ty,
            vis: syn::Visibility::Inherited,
            defaultness: None,
            const_token: Default::default(),
            generics: Default::default(),
            colon_token: Default::default(),
            eq_token: Default::default(),
            semi_token: Default::default(),
        };
        Ok(item)
    }

    fn validate_extend(
        &self,
        item: Option<&ImplItemConst>,
        towards: &ItemImpl,
        context: Option<&CounterArgsSlice<'a>>,
    ) -> Result<(), TokenStream> {
        let extracted;
        let counters = match context {
            Some(c) => c,
            None => {
                extracted = CounterArgs::checked_extract(towards, &None)?;
                extracted.as_slice().as_ref().as_ref()
            }
        };

        let ident = gen_const_ident::<CountersLenMeta>();
        let ty: Type = parse_quote!(usize);
        let len = counters.len();
        let expr: Expr = parse_quote!(#len);

        validate_impl_const! {
            item: item,
            towards: towards,
            ident: ident,
            ty: ty,
            expr: expr,
            errors: {
                not_found: MetaImplItemsBugs::ImplCountersLenMetaNotFound {},
                wrong_ident: MetaImplItemsBugs::ImplCountersLenMetaWrongIdent {},
                wrong_ty: MetaImplItemsBugs::ImplCountersLenMetaInvalidType {},
                has_generics: MetaImplItemsBugs::ImplCountersLenMetaHasGenerics {},
                invalid_expr: MetaImplItemsBugs::ImplCountersLenMetaInvalidExpr {},
            }
        }
    }
}

// ===============================================================================
// `````````````````````````` COUNTERS GENERICS CHECKER ``````````````````````````
// ===============================================================================

/// Extends an instance implementation with a hidden associated constant
/// used as a structural validation checkpoint for instance-counter
/// generic metadata.
///
/// This phase acts as the implementation-side counterpart of
/// [`CountersGenericsChecker`].
///
/// This phase acts as a synchronization/checkpoint node between:
///
/// - [`CountersGenericsMeta`],
/// - [`ImplCountersGenericsIndexesMeta`],
/// - [`ImplCountersLenMeta`],
///
/// The generated constant validates:
///
/// - that the number of impl-side counters matches the expectations
///   reflected by [`CountersLenMeta`],
/// - that impl-side counter generic indexes match the metadata reflected
///   by [`CountersGenericsMeta`] and
///   [`CountersGenericsIndexesMeta`],
/// - that all predecessor instances satisfy the same validation rules,
/// - and that the implementation participates in the expected expansion
///   pipeline.
///
/// For non-zeroth instances, validation recursively evaluates the
/// predecessor's [`CountersGenericsChecker`] through the back-affiliate
/// chain.
///
/// This guarantees that generic-index consistency holds across the
/// entire instance lineage when the terminal instance is invoked
/// rather than only the current instance.
///
/// ## Example
///
/// Trait-side metadata:
///
/// ```ignore
/// trait Example<__TypeNumCounters> {
///     #[doc(hidden)]
///     const __COUNTERS_GENERICS_META:
///         [(usize, &'static str); 2] = [
///             (0, "A"),
///             (2, "B"),
///         ];
///
///     #[doc(hidden)]
///     const __COUNTERS_GENERIC_INDEXES_META:
///         &'static [usize];
///
///     #[doc(hidden)]
///     const __COUNTERS_LEN_META: usize;
///
///     #[doc(hidden)]
///     const __COUNTERS_GENERICS_CHECKER: ();
/// }
/// ```
///
/// Implementation-side expansion:
///
/// ```ignore
/// impl Example<(U1, U3)> for MyType {
///     #[doc(hidden)]
///     const __COUNTERS_GENERIC_INDEXES_META:
///         &'static [usize] = &[0, 2];
///
///     #[doc(hidden)]
///     const __COUNTERS_LEN_META: usize = 2;
///
///     #[doc(hidden)]
///     const __COUNTERS_GENERICS_CHECKER: () = {
///         if trait_generics_meta.len() != impl_len_meta {
///             panic!(/* generated length diagnostic */)
///         } else {
///             if !counter_generics_checker(
///                 &trait_generics_meta,
///                 impl_indexes_meta,
///             ) {
///                 panic!(/* generated generic-index diagnostic */)
///             } else {
///                 ()
///             }
///         }
///     };
/// }
/// ```
///
/// This effectively acts as a hidden synchronization/checkpoint node
/// between trait-side reflection and impl-side validation phases.
#[derive(Debug, Clone)]
pub(super) struct ImplCountersGenericsChecker;

impl Insertion<ItemImpl> for ImplItemConst {
    fn raw_insert(&self, to: &mut ItemImpl, _: &()) -> Result<(), TokenStream> {
        to.items.push(ImplItem::Const(self.clone()));
        Ok(())
    }

    fn validate_inserted(
        _of: Option<&Self>,
        _to: &ItemImpl,
        _: Option<&()>,
    ) -> Result<(), TokenStream> {
        Ok(())
    }
}

impl<'a> Extension<ImplItemConst, ItemImpl, CounterArgsSlice<'a>> for ImplCountersGenericsChecker {
    fn raw_extend(
        &self,
        towards: &ItemImpl,
        context: &CounterArgsSlice<'a>,
    ) -> Result<ImplItemConst, TokenStream> {
        let ident = gen_const_ident::<CountersGenericsChecker>();
        let ty = parse_quote!(());
        let expr = counter_generics_checker(towards, context)?;
        let item = ImplItemConst {
            attrs: proc_suite::internal_code(),
            ident,
            expr,
            ty,
            vis: syn::Visibility::Inherited,
            defaultness: None,
            const_token: Default::default(),
            generics: Default::default(),
            colon_token: Default::default(),
            eq_token: Default::default(),
            semi_token: Default::default(),
        };
        Ok(item)
    }

    fn validate_extend(
        &self,
        item: Option<&ImplItemConst>,
        towards: &ItemImpl,
        context: Option<&CounterArgsSlice<'a>>,
    ) -> Result<(), TokenStream> {
        let extracted;
        let counters = match context {
            Some(c) => c,
            None => {
                extracted = CounterArgs::checked_extract(towards, &None)?;
                extracted.as_slice().as_ref().as_ref()
            }
        };

        let ident = gen_const_ident::<CountersGenericsChecker>();
        let ty: Type = parse_quote!(());
        let expr = counter_generics_checker(towards, counters)?;

        validate_impl_const! {
            item: item,
            towards: towards,
            ident: ident,
            ty: ty,
            expr: expr,
            errors: {
                not_found: MetaImplItemsBugs::ImplCountersGenericsCheckerNotFound {},
                wrong_ident: MetaImplItemsBugs::ImplCountersGenericsCheckerWrongIdent {},
                wrong_ty: MetaImplItemsBugs::ImplCountersGenericsCheckerInvalidType {},
                has_generics: MetaImplItemsBugs::ImplCountersGenericsCheckerHasGenerics {},
                invalid_expr: MetaImplItemsBugs::ImplCountersGenericsCheckerInvalidExpr {},
            }
        }
    }
}

/// Builds the impl-side counter-generic validation expression.
///
/// Example:
///
/// ```ignore
/// {
///     if trait_generics_meta.len() != impl_len_meta {
///         panic!(...)
///     } else {
///         if !counter_generics_checker(
///             &trait_generics_meta : &[(usize, &'static str)],
///             impl_indexes_meta: &[usize],
///         ) {
///             panic!(...)
///         } else {
///             let _ = back_affiliate_checker();
///             ()
///         }
///     }
/// }
/// ```
fn counter_generics_checker<'a>(
    impl_of: &ItemImpl,
    counters: CounterArgsSlice<'a>,
) -> Result<Expr, TokenStream> {
    let crate_of = Instance::support_crate();

    let trait_generics_meta =
        AssocTyExpr::checked_extract(&(impl_of, &gen_const_ident::<CountersGenericsMeta>()), &())?
            .0;

    let impl_len_meta =
        AssocTyExpr::checked_extract(&(impl_of, &gen_const_ident::<CountersLenMeta>()), &())?.0;

    let impl_indexes_meta = AssocTyExpr::checked_extract(
        &(impl_of, &gen_const_ident::<CountersGenericsIndexesMeta>()),
        &(),
    )?
    .0;

    let self_ident = gen_const_ident::<CountersGenericsChecker>();
    let back_checker =
        AssocOfAffiliateTyExpr::<BackAffiliateInstance, BackAffiliateCounters>::checked_extract(
            &(impl_of, &self_ident),
            &counters,
        )?
        .0;

    let trait_ident = ImplTraitIdent::checked_utilize(impl_of, &())?.0;

    let err = MetaErrors::InvalidCounterIndexes {
        trait_of: trait_ident.clone(),
    }
    .to_string();
    if is_zeroth_instance(counters)? {
        let expr = parse_quote!(
            {
                if #trait_generics_meta.len() != #impl_len_meta {
                    let msg = #err;
                    panic!("{}", msg)
                } else {
                    if !#crate_of::counter_generics_checker(
                        &#trait_generics_meta,
                        #impl_indexes_meta
                    ) {
                        let msg = #err;
                        panic!("{}", msg)
                    } else {
                        ()
                    }
                }
            }
        );
        return Ok(expr);
    }
    let expr = parse_quote!(
        {
            if #trait_generics_meta.len() != #impl_len_meta {
                let msg = #err;
                panic!("{}", msg)
            } else {
                if !#crate_of::counter_generics_checker(
                    &#trait_generics_meta,
                    #impl_indexes_meta
                ) {
                    let msg = #err;
                    panic!("{}", msg)
                } else {
                    let _ = #back_checker;
                    ()
                }
            }
        }
    );
    Ok(expr)
}

// ===============================================================================
// `````````````````````````` ORIGINAL COUNTER CONSTANT ``````````````````````````
// ===============================================================================

/// Extends an instance implementation with hidden associated constants
/// representing the original extracted counter values.
///
/// This phase acts as the implementation-side counterpart of
/// [`OriginalCounterConst`].
///
/// After
/// [`InstanceTraitTypeNumCounters`](crate::traits::typenum::InstanceTraitTypeNumCounters)
/// transformation, the original counter const-generics no longer exist
/// on the trait definition.
///
/// Lkewise, after
/// [`InstanceImplTypeNumCounters`](crate::impls::typenum::InstanceImplTypeNumCounters)
/// transformation, instance-counter arguments are encoded into a
/// synthesized type-level counter carrier and no longer exist as
/// ordinary trait generic arguments.
///
/// Since later expansion stages may still need access to the original
/// counter values, this phase reconstructs each extracted counter as a
/// hidden associated constant whose identifier matches the trait-side
/// reflection generated by [`OriginalCounterConst`].
///
/// ## Example
///
/// Before typenum transformation:
///
/// ```ignore
/// #[instance_impl(0, 2)]
/// impl Example<1, T, 3> for MyType {
///     type InstanceCounter = u8;
/// }
/// ```
///
/// After typenum transformation:
///
/// ```ignore
/// impl Example<(U1, U3), T> for MyType {
///     type InstanceCounter = u8;
/// }
/// ```
///
/// After metadata expansion:
///
/// ```ignore
/// impl Example<(U1, U3), T> for MyType {
///     type InstanceCounter = u8;
///
///     #[doc(hidden)]
///     const __COUNTER_INDEX_0: u8 = 1;
///
///     #[doc(hidden)]
///     const __COUNTER_INDEX_2: u8 = 3;
/// }
/// ```
///
/// These generated constants mirror the trait-side constants generated
/// by [`OriginalCounterConst`] and restore access to the original
/// counter values after typenum transformation performed by
/// [`InstanceTraitTypeNumCounters`](crate::traits::typenum::InstanceTraitTypeNumCounters)
/// and
/// [`InstanceImplTypeNumCounters`](crate::impls::typenum::InstanceImplTypeNumCounters).
///
/// The generated identifiers are derived from
/// [`OriginalCounterConst`] using the corresponding generic indexes,
/// ensuring that trait-side and impl-side expansion refer to the same
/// reflected counter representation.
///
/// Together, [`OriginalCounterConst`] and
/// [`ImplOriginalCounterConst`] provide a stable associated-constant
/// representation of original counter values across the entire
/// instance-trait expansion pipeline.
#[derive(Debug, Clone)]
pub(super) struct ImplOriginalCounterConst;

impl<'a> Insertion<ItemImpl, CounterArgsSlice<'a>> for Vec<ImplItemConst> {
    fn raw_insert(&self, to: &mut ItemImpl, _: &CounterArgsSlice) -> Result<(), TokenStream> {
        for c in self {
            to.items.push(ImplItem::Const(c.clone()));
        }
        Ok(())
    }

    fn validate_inserted(
        _of: Option<&Self>,
        _to: &ItemImpl,
        _context: Option<&CounterArgsSlice<'a>>,
    ) -> Result<(), TokenStream> {
        Ok(())
    }
}

impl<'a> Extension<Vec<ImplItemConst>, ItemImpl, CounterArgsSlice<'a>>
    for ImplOriginalCounterConst
{
    fn raw_extend(
        &self,
        towards: &ItemImpl,
        context: &CounterArgsSlice<'a>,
    ) -> Result<Vec<ImplItemConst>, TokenStream> {
        let mut collect = Vec::new();
        let ty = ImplCountersUTy::checked_utilize(towards, &())?.0;
        for c in context.iter() {
            let gen_idx = &c.generic_index;
            let lit = &c.const_lit;

            let ident = gen_const_ident_with_suffix::<OriginalCounterConst>(Some(
                gen_idx.to_string().as_bytes(),
            ));
            let expr = parse_quote!(#lit);

            let item = ImplItemConst {
                attrs: proc_suite::internal_code(),
                ident,
                expr,
                ty: ty.clone(),
                vis: syn::Visibility::Inherited,
                defaultness: None,
                const_token: Default::default(),
                generics: Default::default(),
                colon_token: Default::default(),
                eq_token: Default::default(),
                semi_token: Default::default(),
            };
            collect.push(item);
        }
        Ok(collect)
    }

    fn validate_extend(
        &self,
        item: Option<&Vec<ImplItemConst>>,
        towards: &ItemImpl,
        context: Option<&CounterArgsSlice<'a>>,
    ) -> Result<(), TokenStream> {
        let extracted;
        let counters = match context {
            Some(c) => c,
            None => {
                extracted = CounterArgs::checked_extract(towards, &None)?;
                extracted.as_slice().as_ref().as_ref()
            }
        };

        let ty = ImplCountersUTy::checked_utilize(towards, &())?.0;
        for c in counters.iter() {
            let gen_idx = &c.generic_index;
            let lit = &c.const_lit;

            let ident = gen_const_ident_with_suffix::<OriginalCounterConst>(Some(
                gen_idx.to_string().as_bytes(),
            ));

            let expr: Expr = parse_quote!(#lit);
            validate_impl_consts! {
                items: item,
                towards: towards,
                ident: ident,
                ty: ty,
                expr: expr,
                errors: {
                    not_found: MetaImplItemsBugs::ImplOriginalCounterConstNotFound {},
                    wrong_ident: MetaImplItemsBugs::ImplOriginalCounterConstWrongIdent {},
                    wrong_ty: MetaImplItemsBugs::ImplOriginalCounterConstInvalidType {},
                    has_generics: MetaImplItemsBugs::ImplOriginalCounterConstHasGenerics {},
                    invalid_expr: MetaImplItemsBugs::ImplOriginalCounterConstInvalidExpr {},
                }
            }
        }
        Ok(())
    }
}

// ===============================================================================
// `````````````````````````` CUMULATED CONST CHECKER ``````````````````````````
// ===============================================================================

/// Provides the default implementation of [`CumulatedConstChecker`].
///
/// All ordinary instance implementations receive a trivial checker:
///
/// ```text
/// ()
/// ```
///
/// and therefore perform no validation through this constant alone.
///
/// ## Why this exists
///
/// [`CumulatedConstChecker`] acts as the root evaluation entry point for
/// compile-time validation, but only terminal instances can determine
/// when the complete affiliate graph has been constructed.
///
/// Consequently, non-terminal instances intentionally emit a no-op
/// implementation.
///
/// ## Terminal Replacement
///
/// During terminal-instance expansion, the generated placeholder is
/// replaced with an accumulated checker expression that references all
/// required validation phases.
///
/// Conceptually:
///
/// ```text
/// non-terminal:
///     CumulatedConstChecker = ()
///
/// terminal:
///     CumulatedConstChecker = {
///         CheckerA;
///         CheckerB;
///         CheckerC;
///         ...
///     }
/// ```
///
/// This allows validation to remain lazy during ordinary instance
/// expansion while still providing a single evaluation point once a
/// terminal boundary is reached.
///
/// As a result, forcing evaluation of a terminal instance's
/// [`CumulatedConstChecker`] can recursively validate the complete
/// affiliate lineage and all accumulated checker phases.
#[derive(Debug, Clone)]
pub(super) struct ImplCumulatedConstChecker;

impl<'a> Extension<ImplItemConst, ItemImpl> for ImplCumulatedConstChecker {
    fn raw_extend(&self, _: &ItemImpl, _: &()) -> Result<ImplItemConst, TokenStream> {
        let ident = gen_const_ident::<CumulatedConstChecker>();
        let ty = parse_quote!(());
        let expr: Expr = parse_quote!(());
        let item = ImplItemConst {
            attrs: proc_suite::internal_code(),
            ident,
            expr,
            ty,
            vis: syn::Visibility::Inherited,
            defaultness: None,
            const_token: Default::default(),
            generics: Default::default(),
            colon_token: Default::default(),
            eq_token: Default::default(),
            semi_token: Default::default(),
        };
        Ok(item)
    }

    fn validate_extend(
        &self,
        item: Option<&ImplItemConst>,
        towards: &ItemImpl,
        _: Option<&()>,
    ) -> Result<(), TokenStream> {
        let ident = gen_const_ident::<CumulatedConstChecker>();
        let ty: Type = parse_quote!(());
        let expr: Expr = parse_quote!(());

        validate_impl_const! {
            item: item,
            towards: towards,
            ident: ident,
            ty: ty,
            expr: expr,
            errors: {
                not_found: MetaImplItemsBugs::ImplCumulatedConstCheckerNotFound {},
                wrong_ident: MetaImplItemsBugs::ImplCumulatedConstCheckerWrongIdent {},
                wrong_ty: MetaImplItemsBugs::ImplCumulatedConstCheckerNotUnitType {},
                has_generics: MetaImplItemsBugs::ImplCumulatedConstCheckerHasGenerics {},
                invalid_expr: MetaImplItemsBugs::ImplCumulatedConstCheckerInvalidExpr {},
            }
        }
    }
}

// ===============================================================================
// ```````````````````````````` GLOBAL TERMINAL ASSOCS ```````````````````````````
// ===============================================================================

/// Provides the meta-level terminal boundaries required by the instance
/// implementation's sum-type machinery.
///
/// An instance implementation represents one concrete member of the
/// instance-trait family. This transformation exposes, as associated types,
/// the information needed to relate that implementation to the global
/// instance and its terminal instance:
///
/// ```ignore
/// impl Example<1> for MyType {
///     type GlobalTerminalAssoc = Global;
///
///     type GlobalTerminalExact =
///         <Global as TerminalAccess<Self>>::Terminal;
///
///     type SelfTerminalAssoc = Self;
/// }
/// ```
///
/// These associated types are meta information about the implementation:
/// [`GlobalTerminalAssoc`] identifies the global instance, [`GlobalTerminalExact`]
/// identifies its exact terminal counter, and [`SelfTerminalAssoc`] identifies
/// the concrete implementation itself.
///
/// This is what connects the implementation side to sum types. A `#[sum]`
/// associated type is given `From`/`Into` bounds to the corresponding
/// associated type of `SelfTerminalAssoc`, making the different instance
/// implementations behave like members of an enum-like type boundary:
///
/// ```text
/// Example<1>::Value \
/// Example<2>::Value ---terminal instance::Value
/// Example<3>::Value /
/// ```
#[derive(Debug, Clone)]
pub(super) struct GlobalTerminalAssocs;

impl Transformation<ItemImpl> for GlobalTerminalAssocs {
    fn raw_transform(
        &self,
        transform: &mut ItemImpl,
        _: &(),
    ) -> Result<(), proc_macro2::TokenStream> {
        let crate_of = Instance::support_crate();
        transform.items.push(syn::ImplItem::Type(ImplItemType {
            attrs: proc_suite::internal_code(),
            ident: gen_type_ident::<GlobalTerminalAssoc>(),
            ty: parse_quote!(#crate_of::Global),
            vis: syn::Visibility::Inherited,
            defaultness: None,
            type_token: Default::default(),
            generics: Default::default(),
            eq_token: Default::default(),
            semi_token: Default::default(),
        }));
        let self_ty = &*transform.self_ty;
        transform.items.push(syn::ImplItem::Type(ImplItemType {
            attrs: proc_suite::internal_code(),
            ident: gen_type_ident::<GlobalTerminalExact>(),
            ty: parse_quote!(<#crate_of::Global as #crate_of::TerminalAccess<#self_ty>>::Terminal),
            vis: syn::Visibility::Inherited,
            defaultness: None,
            type_token: Default::default(),
            generics: Default::default(),
            eq_token: Default::default(),
            semi_token: Default::default(),
        }));
        transform.items.push(syn::ImplItem::Type(ImplItemType {
            attrs: proc_suite::internal_code(),
            ident: gen_type_ident::<SelfTerminalAssoc>(),
            ty: parse_quote!(#self_ty),
            vis: syn::Visibility::Inherited,
            defaultness: None,
            type_token: Default::default(),
            generics: Default::default(),
            eq_token: Default::default(),
            semi_token: Default::default(),
        }));
        Ok(())
    }

    fn validate_transform(
        &self,
        transform: &ItemImpl,
        _: Option<&()>,
    ) -> Result<(), proc_macro2::TokenStream> {
        let crate_of = Instance::support_crate();
        let self_ty = &*transform.self_ty;

        let global_assoc = gen_type_ident::<GlobalTerminalAssoc>();
        let global_terminal = gen_type_ident::<GlobalTerminalExact>();
        let self_terminal = gen_type_ident::<SelfTerminalAssoc>();

        let expected_global: Type = parse_quote!(
            #crate_of::Global
        );

        let expected_terminal: Type = parse_quote!(
            <#crate_of::Global as #crate_of::TerminalAccess<#self_ty>>::Terminal
        );

        let expected_self: Type = parse_quote!(
            #self_ty
        );

        let mut found_global = false;
        let mut found_terminal = false;
        let mut found_self = false;

        for item in &transform.items {
            let ImplItem::Type(item) = item else {
                continue;
            };

            if item.ident == global_assoc {
                found_global = true;

                if item.generics != Default::default() {
                    return Err(MetaImplItemsBugs::GlobalAssocWrongType {}.into());
                }

                if item.ty != expected_global {
                    return Err(MetaImplItemsBugs::GlobalAssocWrongType {}.into());
                }
            } else if item.ident == global_terminal {
                found_terminal = true;

                if item.generics != Default::default() {
                    return Err(MetaImplItemsBugs::GlobalTerminalExactWrongType {}.into());
                }

                if item.ty != expected_terminal {
                    return Err(MetaImplItemsBugs::GlobalTerminalExactWrongType {}.into());
                }
            } else if item.ident == self_terminal {
                found_self = true;

                if item.generics != Default::default() {
                    return Err(MetaImplItemsBugs::SelfTerminalAssocWrongType {}.into());
                }

                if item.ty != expected_self {
                    return Err(MetaImplItemsBugs::SelfTerminalAssocWrongType {}.into());
                }
            }
        }

        if !found_global {
            return Err(MetaImplItemsBugs::GlobalAssocNotFound {}.into());
        }

        if !found_terminal {
            return Err(MetaImplItemsBugs::GlobalTerminalExactNotFound {}.into());
        }

        if !found_self {
            return Err(MetaImplItemsBugs::SelfTerminalAssocNotFound {}.into());
        }

        Ok(())
    }
}
