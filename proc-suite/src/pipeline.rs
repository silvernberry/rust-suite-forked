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
// ```````````````````````````` PROC-MACRO PIPELINES `````````````````````````````
// ===============================================================================

//! Provides structured pipeline primitives for building proc-macro workflows.
//!
//! The pipeline organizes syntax processing into explicit stages for extraction,
//! utilization, extension, transformation, and insertion via declaring usable
//! traits.
//!
//! See [`SupportCrate`] for support-crate resolution and delegation semantics.

// ===============================================================================
// ``````````````````````````````````` IMPORTS ```````````````````````````````````
// ===============================================================================

// --- Syn Crate ---
use syn::parse_quote;

// --- Local Crate ---
use crate::syns::{AttributePush, PathQualifierReplace, PushAttribute, ReplacePathQualifier};

// ===============================================================================
// ```````````````````````````````` PROC-PIPELINE ````````````````````````````````
// ===============================================================================

/// Generates a structured proc-macro pipeline as a set of local traits.
///
/// Procedural macros operate on syntax represented as `TokenStream`s.
/// Input is parsed into structured syntax, analyzed, transformed, and
/// ultimately emitted back as a `TokenStream`.
///
/// This macro organizes that process into explicit pipeline stages:
///
/// `text
/// Extract -> Extend -> Transform -> Insert
///      \
///       -> Utilize
/// `
///
/// - **Extract**: derives owned, pipeline-friendly structures from syntax.
/// - **Utilize**: derives borrowed, lifetime-carrying views over existing syntax.
/// - **Extend**: conditionally derives and inserts new structures.
/// - **Transform**: mutates existing structures in place.
/// - **Insert**: integrates data into a target representation.
///
/// Procedural macros are fundamentally syntactic transformations. Some
/// workflows benefit from extracting syntax into owned intermediate
/// representations, while others can operate directly on borrowed views
/// of existing syntax.
///
/// The stages can therefore be composed according to the transformation:
///
/// - `extract -> extend -> insert`
/// - `extract -> transform`
/// - `utilize -> transform`
/// - `utilize -> extend -> insert`
/// - or any combination of stages.
///
/// Each stage supports two execution modes:
///
/// - **checked**: performs pre- and post-validation.
/// - **unchecked**: performs minimal execution.
///
/// ## Syntax
///
/// ```ignore
/// proc_pipeline! {
///     struct MacroGlobalIdent {
///         extract: ExtractTraitIdent,
///         insert: InsertTraitIdent,
///         extend: ExtendTraitIdent,
///         transform: TransformTraitIdent,
///         utilize: UtilizeTraitIdent,
///         support: "support_crate",
///         delegate: ("delegate_crate", "delegate_macro"), // optional
///     }
/// }
/// ```
///
/// `support` specifies the support crate used by generated code.
/// `delegate` **optionally** specifies the delegated support crate and its
/// delegate attribute macro.
///
/// See [`SupportCrate`] for the semantics of support-crate resolution
/// and delegation.
///
/// ## Generated Traits
///
/// The macro expands into traits declared at the invocation site
/// (`pub(crate)`). This allows implementations for foreign types, such as
/// `syn` syntax nodes, while avoiding orphan-rule restrictions by keeping
/// the traits local to the crate.
///
/// ## Purpose
///
/// Provides a predictable and composable structure for proc-macro logic,
/// replacing ad-hoc transformations with explicit pipeline stages and
/// defined ownership boundaries between extracted and utilizable
/// representations.
#[macro_export]
macro_rules! proc_pipeline {

    (@delegate $Delegate:literal, $DelegateProcMacro:literal) => {
        fn delegate_crate() -> Option<syn::Ident> {
            Some(::quote::format_ident!($Delegate))
        }

        fn delegate_macro() -> Option<syn::Ident> {
            Some(::quote::format_ident!($DelegateProcMacro))
        }
    };

    (@delegate) => {};

    (
        $(#[$Docs:meta])*
        $Vis:vis struct $Type:ident {
            extract: $Extract:ident,
            insert: $Insert:ident,
            extend: $Extend:ident,
            transform: $Transform:ident,
            utilize: $Utilize:ident,
            support: $Support:literal
            $(, delegate: ($Delegate:literal, $DelegateProcMacro:literal))?
            $(,)?
        }
    ) => {

        $(#[$Docs])*
        $Vis struct $Type;

        impl $crate::pipeline::SupportCrate for $Type {
            fn support_crate() -> syn::Ident {
                quote::format_ident!($Support)
            }

            proc_pipeline!(@delegate
                $($Delegate, $DelegateProcMacro)?
            );
        }

        /// Defines extraction of structured data from an input source.
        ///
        /// This trait represents the initial stage of a structured proc-macro
        /// pipeline, where relevant information is derived from syntax or other
        /// inputs and prepared for further processing.
        ///
        /// `From` typically represents the syntax node (or source) from which
        /// `Self` is derived, under the given `Context`.
        ///
        /// `Context` provides optional external state, configuration, or borrowed
        /// supporting data used during extraction.
        ///
        /// Since `Self` is typically owned (`'static`), `Context` is commonly used
        /// to provide temporary borrowed data required during derivation or validation.
        ///
        /// This trait is usually implemented for types that hold the extracted
        /// data used in later processing stages.
        ///
        /// Provides two execution paths:
        /// - [`Self::checked_extract`]: includes validation before and after extraction
        /// - [`Self::unchecked_extract`]: performs extraction with minimal checks
        ///
        /// Implementors control input validation, extraction logic, and
        /// post-extraction verification.
        pub(crate) trait $Extract<From, Context = ()>
        where
            Self: Sized + 'static + Clone + std::fmt::Debug,
        {
            /// Performs extraction with full validation.
            ///
            /// Validates the source and context, executes extraction,
            /// and verifies the resulting value.
            fn checked_extract(
                from: &From,
                context: &Context,
            ) -> Result<Self, proc_macro2::TokenStream> {
                Self::validate_from(from)?;
                Self::validate_context(context)?;
                let result = Self::raw_extract(from, context)?;
                result.validate_extract(from, Some(context))?;
                Ok(result)
            }

            /// Performs extraction with minimal validation.
            ///
            /// Skips post-extraction validation, returning the raw result.
            fn unchecked_extract(
                from: &From,
                context: &Context,
            ) -> Result<Self, proc_macro2::TokenStream> {
                Self::validate_from(from)?;
                Self::validate_context(context)?;
                Self::raw_extract(from, context)
            }

            /// Validates the input source before extraction.
            ///
            /// Override to enforce preconditions on `From`.
            ///
            /// Default is no-op.
            fn validate_from(_from: &From) -> Result<(), proc_macro2::TokenStream> {
                Ok(())
            }

            /// Validates the extraction context before extraction.
            ///
            /// Override to enforce constraints on `Context`.
            ///
            /// Default is no-op.
            fn validate_context(_context: &Context) -> Result<(), proc_macro2::TokenStream> {
                Ok(())
            }

            /// Performs the core extraction logic.
            ///
            /// Converts `From` into `Self` using the given `Context`.
            fn raw_extract(from: &From, context: &Context) -> Result<Self, proc_macro2::TokenStream>;

            /// Validates the extracted result.
            ///
            /// `context` is optional so this validation hook may be reused
            /// independently from the checked extraction pipeline, such as:
            ///
            /// - standalone tests,
            /// - partial pipeline execution,
            /// - manual validation,
            /// - or external validation utilities.
            fn validate_extract(
                &self,
                from: &From,
                context: Option<&Context>,
            ) -> Result<(), proc_macro2::TokenStream>;
        }

        /// Defines insertion of `Self` into a target type (`Into`) using a given context.
        ///
        /// This represents the mutation stage of a structured proc-macro pipeline,
        /// where `Self` is integrated into `Into` as part of its final structure.
        ///
        /// `Into` is typically a syntax node or target representation, while
        /// `Context` provides external rules or configuration required during
        /// insertion (such as naming policy, validation rules, feature flags,
        /// or expansion constraints).
        ///
        /// `Context` provides optional external state, configuration, or borrowed
        /// supporting data used during insertion.
        ///
        /// Since `Self` is typically owned (`'static`), `Context` is commonly used
        /// to provide temporary borrowed data required during derivation or validation.
        ///
        /// `Self` may represent:
        /// - a generated syntax node,
        /// - extracted metadata being re-inserted,
        /// - or contextual data projected into the target representation.
        ///
        /// Provides two execution paths:
        /// - [`Self::checked_insert`] performs full validation before and after insertion
        /// - [`Self::unchecked_insert`] performs insertion with minimal checks
        ///
        /// This keeps insertion predictable, validated, and composable within the
        /// larger proc-macro pipeline.
        pub(crate) trait $Insert<Into, Context = ()>
        where
            Self: Sized + 'static + Clone + std::fmt::Debug,
        {
            /// Performs insertion with full validation.
            ///
            /// Validates the insertion context and target, applies the insertion,
            /// and verifies the final inserted state.
            fn checked_insert(
                &self,
                to: &mut Into,
                context: &Context,
            ) -> Result<(), proc_macro2::TokenStream> {
                self.validate_context(context)?;
                self.validate_into(to, context)?;
                self.raw_insert(to, context)?;
                Self::validate_inserted(Some(self), to, Some(context))
            }

            /// Performs insertion with minimal validation.
            ///
            /// Validates preconditions and applies insertion without checking
            /// the final inserted state.
            fn unchecked_insert(
                &self,
                to: &mut Into,
                context: &Context,
            ) -> Result<(), proc_macro2::TokenStream> {
                self.validate_context(context)?;
                self.validate_into(to, context)?;
                self.raw_insert(to, context)
            }

            /// Validates the insertion context before insertion.
            ///
            /// Override to enforce constraints on the external insertion context.
            ///
            /// Default implementation performs no validation.
            fn validate_context(&self, _context: &Context) -> Result<(), proc_macro2::TokenStream> {
                Ok(())
            }

            /// Validates the target before insertion.
            ///
            /// Override to ensure the target is in a valid state to receive
            /// the insertion.
            ///
            /// Default implementation performs no validation.
            fn validate_into(
                &self,
                _to: &Into,
                _context: &Context,
            ) -> Result<(), proc_macro2::TokenStream> {
                Ok(())
            }

            /// Performs the core insertion logic.
            ///
            /// Integrates `Self` into `Into`, mutating the target representation
            /// using the provided context.
            fn raw_insert(
                &self,
                to: &mut Into,
                context: &Context,
            ) -> Result<(), proc_macro2::TokenStream>;

            /// Validates the target after insertion.
            /// to ensure the resulting structure remains valid.
            ///
            /// `context` and `inserted`` are optional so this validation hook may be reused
            /// independently from the checked extraction pipeline, such as:
            ///
            /// - standalone tests,
            /// - partial pipeline execution,
            /// - manual validation,
            /// - or external validation utilities.
            fn validate_inserted(
                _of: Option<&Self>,
                _to: &Into,
                _context: Option<&Context>,
            ) -> Result<(), proc_macro2::TokenStream>;
        }

        /// Defines extension of a target (`Towards`) within the proc-macro pipeline.
        ///
        /// This stage sits between extraction and insertion:
        #[doc = concat!(
                    "- extraction derives structured data via [`", stringify!($Extract), "`],\n",
                    "- insertion integrates generated data via [`", stringify!($Insert), "`],"
                )]
        /// - extension uses extracted or contextual data (`Self`) to derive a new
        ///   `Type`, which is then inserted into `Towards`.
        ///
        /// `Self` usually represents validated extracted data or contextual state
        /// that decides *what* should be added.
        ///
        /// `Context` provides optional external state, configuration, or borrowed
        /// supporting data used during extension.
        ///
        /// Since `Self` is typically owned (`'static`), `Context` is commonly used
        /// to provide temporary borrowed data required during derivation or validation.
        ///
        /// The produced `Type` must implement
        #[doc = concat!("[`", stringify!($Insert), "`],")]
        /// allowing it to be inserted into the target.
        ///
        /// Provides two execution paths:
        /// - [`Self::checked_extend`] performs validation before derivation and insertion
        /// - [`Self::unchecked_extend`] performs extension with minimal checks
        ///
        /// This enables controlled structural growth of syntax nodes and target
        /// representations during macro expansion.
        pub(crate) trait $Extend<Type, Towards, Context = ()>
        where
            Self: Sized + 'static + Clone + std::fmt::Debug,
            Type: $Insert<Towards, Context> + Sized,
        {
            /// Performs extension with full validation.
            ///
            /// Validates the context and target, derives the item,
            /// validates the derived result, and inserts it using
            #[doc = concat!("[`", stringify!($Insert), "`].")]
            fn checked_extend(
                &self,
                insert_to: &mut Towards,
                context: &Context,
            ) -> Result<(), proc_macro2::TokenStream> {
                self.validate_context(context)?;
                self.validate_towards(insert_to, context)?;
                let item = self.raw_extend(insert_to, context)?;
                item.checked_insert(insert_to, context)?;
                self.validate_extend(Some(&item), insert_to, Some(context))?;
                Ok(())
            }

            /// Performs extension with minimal validation.
            ///
            /// Validates preconditions, derives the item, and inserts it
            /// without validating the derived value.
            fn unchecked_extend(
                &self,
                insert_to: &mut Towards,
                context: &Context,
            ) -> Result<(), proc_macro2::TokenStream> {
                self.validate_context(context)?;
                self.validate_towards(insert_to, context)?;
                let item = self.raw_extend(insert_to, context)?;
                item.unchecked_insert(insert_to, context)?;
                Ok(())
            }

            /// Validates the extension context before extension.
            ///
            /// Override to enforce constraints on the external extension context.
            ///
            /// Default implementation performs no validation.
            fn validate_context(&self, _context: &Context) -> Result<(), proc_macro2::TokenStream> {
                Ok(())
            }

            /// Validates the target before extension.
            ///
            /// Override to ensure the target is in a valid state to receive
            /// the derived insertion.
            ///
            /// Default implementation performs no validation.
            fn validate_towards(
                &self,
                _towards: &Towards,
                _context: &Context,
            ) -> Result<(), proc_macro2::TokenStream> {
                Ok(())
            }

            /// Produces the item to be inserted into the target.
            ///
            /// Derives a new `Type` from `Self` using the current state
            /// of `Towards` and the provided context.
            ///
            /// The returned value is later inserted via
            #[doc = concat!("[`", stringify!($Insert), "`].")]
            fn raw_extend(
                &self,
                towards: &Towards,
                context: &Context,
            ) -> Result<Type, proc_macro2::TokenStream>;

            /// Validates the extended result
            ///
            /// `context` and `item` are optional so this validation hook may be reused
            /// independently from the checked extraction pipeline, such as:
            ///
            /// - standalone tests,
            /// - partial pipeline execution,
            /// - manual validation,
            /// - or external validation utilities.
            fn validate_extend(
                &self,
                item: Option<&Type>,
                towards: &Towards,
                context: Option<&Context>,
            ) -> Result<(), proc_macro2::TokenStream>;
        }

        /// Defines transformation of an existing value (`Type`) in place.
        ///
        /// Unlike extension (which derives and inserts new data), transformation
        /// operates directly on the target, modifying or projecting its internal
        /// representation.
        #[doc = concat!(
                    "See [`", stringify!($Extend), "`] for extension-based workflows."
                )]
        ///
        /// `Self` acts as contextual or extracted data used to drive how the
        /// transformation is applied.
        ///
        /// `Context` provides optional external state, configuration, or borrowed
        /// supporting data used during transformation.
        ///
        /// Since `Self` is typically owned (`'static`), `Context` is commonly used
        /// to provide temporary borrowed data required during derivation or validation.
        ///
        /// Provides two execution paths:
        /// - [`Self::checked_transform`]: applies transformation with post-validation
        /// - [`Self::unchecked_transform`]: applies transformation with minimal checks
        ///
        /// This is typically used for rewriting, normalizing, or updating a
        /// structure without introducing new elements via insertion.
        pub(crate) trait $Transform<Type, Context = ()>
        where
            Self: Sized + 'static + Clone + std::fmt::Debug,
        {
            /// Performs transformation with full validation.
            ///
            /// Checks preconditions, applies the transformation,
            /// and validates the resulting state.
            fn checked_transform(&self, transform: &mut Type, context: &Context) -> Result<(), proc_macro2::TokenStream> {
                self.validate_type(transform)?;
                self.validate_context(context)?;
                self.raw_transform(transform, context)?;
                self.validate_transform(transform, Some(context))?;
                Ok(())
            }

            /// Performs transformation with minimal validation.
            ///
            /// Applies the transformation without validating the result.
            fn unchecked_transform(&self, transform: &mut Type, context: &Context) -> Result<(), proc_macro2::TokenStream> {
                self.validate_type(transform)?;
                self.validate_context(context)?;
                self.raw_transform(transform, context)?;
                Ok(())
            }


            /// Validates the transformation context.
            ///
            /// Default implementation performs no validation.
            fn validate_context(&self, _context: &Context) -> Result<(), proc_macro2::TokenStream> {
                Ok(())
            }

            /// Validates the target before transformation.
            ///
            /// Override to ensure the target is in a valid state prior
            /// to mutation.
            ///
            /// Default implementation performs no validation.
            fn validate_type(
                &self,
                _type: &Type,
            ) -> Result<(), proc_macro2::TokenStream> {
                Ok(())
            }

            /// Performs the core transformation logic.
            ///
            /// Mutates the target in place using `Self` as context,
            /// updating its internal representation.
            fn raw_transform(&self, transform: &mut Type, context: &Context) -> Result<(), proc_macro2::TokenStream>;

            /// Validates the transformed state.
            ///
            /// `context` and `type` are optional so this validation hook may be reused
            /// independently from the checked extraction pipeline, such as:
            ///
            /// - standalone tests,
            /// - partial pipeline execution,
            /// - manual validation,
            /// - or external validation utilities.
            fn validate_transform(&self, transform: &Type, context: Option<&Context>) -> Result<(), proc_macro2::TokenStream>;
        }


        /// Defines derivation of data from an input source without requiring
        /// ownership of the resulting representation.
        ///
        /// Procedural macros are fundamentally syntactic transformations.
        /// While some workflows benefit from extracting syntax into owned,
        /// pipeline-friendly structures, many transformations primarily operate
        /// on existing syntax trees and reuse most of their original structure.
        ///
        /// In such cases, constructing a fully owned extraction model may be
        /// unnecessary. Borrowing from the source syntax can more naturally
        /// express the relationship between derived data and the structures from
        /// which it originates while avoiding allocation of owned intermediates.
        ///
        /// Unlike
        #[doc = concat!("[`", stringify!($Extract), "`],")]
        /// this trait is designed for lifetime-carrying representations.
        /// Implementors are permitted to borrow from the source and context,
        /// allowing `Self` to act as a lightweight utilizable view over existing
        /// syntax rather than an owned pipeline artifact.
        ///
        /// `From` typically represents the syntax node (or source) from which
        /// `Self` is derived under the given `Context`.
        ///
        /// `Context` provides optional external state, configuration, or borrowed
        /// supporting data used during derivation.
        ///
        /// The lifetime `'a` represents the lifetime of data borrowed from the
        /// source and context. Implementations may use it to construct temporary
        /// projections, views, or metadata representations tied to the lifetime
        /// of the originating structures.
        ///
        /// This trait is commonly used when:
        ///
        /// - inspecting existing syntax without constructing owned intermediates,
        /// - borrowing directly from syntax trees,
        /// - building temporary projections or views,
        /// - sharing references to contextual information,
        /// - or driving transformations from existing structures.
        ///
        /// Provides two execution paths:
        ///
        /// - [`Self::checked_utilize`]: includes validation before and after derivation
        /// - [`Self::unchecked_utilize`]: performs derivation with minimal checks
        ///
        /// Implementors control source validation, derivation logic, and
        /// post-derivation verification.
        ///
        /// See
        #[doc = concat!("[`", stringify!($Extract), "`]")]
        /// when an owned (`'static`) pipeline artifact is required.
        pub(crate) trait $Utilize<'a, From, Context = ()>
        where
            Self: Sized + std::fmt::Debug + 'a,
        {

            /// Performs derivation of a utilizable value with full validation.
            ///
            /// Validates the source and context, constructs the utilizable value,
            /// and verifies the resulting representation.
            fn checked_utilize(
                from: &'a From,
                context: &'a Context,
            ) -> Result<Self, proc_macro2::TokenStream> {
                Self::validate_from(from)?;
                Self::validate_context(context)?;
                let result = Self::raw_utilize(from, context)?;
                result.validate_utilize(from, Some(context))?;
                Ok(result)
            }


            /// Performs derivation of a utilizable value with minimal validation.
            ///
            /// Skips post-derivation validation, returning the raw result.
            fn unchecked_utilize(
                from: &'a From,
                context: &'a Context,
            ) -> Result<Self, proc_macro2::TokenStream> {
                Self::validate_from(from)?;
                Self::validate_context(context)?;
                Self::raw_utilize(from, context)
            }


            /// Validates the input source before construction of the utilizable value.
            ///
            /// Override to enforce preconditions on `From`.
            ///
            /// Default implementation performs no validation.
            fn validate_from(_from: &From) -> Result<(), proc_macro2::TokenStream> {
                Ok(())
            }


            /// Validates the utilization context before construction of the
            /// utilizable value.
            ///
            /// Override to enforce constraints on `Context`.
            ///
            /// Default implementation performs no validation.
            fn validate_context(_context: &Context) -> Result<(), proc_macro2::TokenStream> {
                Ok(())
            }

            /// Performs the core utilization logic.
            ///
            /// Constructs a utilizable representation from `From` using the given
            /// `Context`.
            ///
            /// Unlike owned extraction pipelines, implementations may return
            /// borrowed, lifetime-carrying, or otherwise non-`'static` views over
            /// the source representation.
            fn raw_utilize(from: &'a From, context: &'a Context) -> Result<Self, proc_macro2::TokenStream>;

            /// Validates the utilizable result.
            ///
            /// `context` is optional so this validation hook may be reused
            /// independently from the checked utilization pipeline, such as:
            ///
            /// - standalone tests,
            /// - partial pipeline execution,
            /// - manual validation,
            /// - or external validation utilities.
            fn validate_utilize(
                &self,
                from: &From,
                context: Option<&Context>,
            ) -> Result<(), proc_macro2::TokenStream>;
        }

    };
}

/// Describes the crate-resolution and delegation model used by a proc-macro
/// pipeline.
///
/// # Why a support crate is necessary
///
/// A procedural macro is implemented in a proc-macro crate, but the code
/// produced by that macro is ultimately compiled in the user's crate. The
/// generated code may need to refer to traits, types, helper macros, or other
/// items that belong to the library associated with the proc macro.
///
/// Those items should be accessed through a normal library crate rather than
/// by making generated code depend directly on the proc-macro implementation
/// crate. The normal library crate acts as the public support layer:
///
/// ```text
/// user crate
///     |
///     |--- depends on -> support crate
///                           |
///                           |--- depends on -> proc-macro crate
/// ```
///
/// For example, suppose a library provides a proc macro and also provides a
/// trait required by the code generated by that macro:
///
/// ```text
/// my_library
///     |--- public support items
///     |     |--- Trait
///     |     |--- Type
///     |
///     |--- re-exports -> my_library_macro
/// ```
///
/// A user then depends on `my_library`, not directly on
/// `my_library_macro`. If the proc macro generates an implementation of
/// `Trait`, the generated code should therefore refer to the trait through
/// the public library:
///
/// ```ignore
/// my_library::Trait
/// ```
///
/// rather than directly referring to the implementation crate.
///
/// The support crate can re-export the proc-macro entry points and the items
/// required by generated code. This gives the generated code one stable,
/// user-visible crate through which it can access everything belonging to the
/// macro's public interface.
///
/// Consequently, generated code commonly needs to construct paths whose first
/// segment is the identifier returned by [`Self::support_crate`]:
///
/// ```ignore
/// #support_crate::Trait
/// #support_crate::Type
/// #support_crate::helper_macro!
/// ```
///
/// For example, if `support_crate()` returns the identifier `my_library`, the
/// generated paths become conceptually:
///
/// ```ignore
/// my_library::Trait
/// my_library::Type
/// my_library::helper_macro!
/// ```
///
/// [`Self::support_crate`] provides the identifier used as the first path
/// segment for these generated paths. Keeping this information in the
/// pipeline makes crate resolution explicit and allows every part of the
/// pipeline to construct paths consistently.
///
/// # Why delegation is necessary
///
/// A proc macro can itself be used as a building block by another proc macro.
/// In that situation, the inner proc macro still has its own support crate,
/// but the user may depend only on the outer macro's support crate.
///
/// The inner support crate can re-export the outer support crate's public
/// items. This means the generated code from the outer macro must sometimes
/// refer to its own outer support crate rather than directly to the inner macro's
/// original support crate.
///
/// For example, consider two layers of proc-macro composition:
///
/// ```text
/// user crate
///     |
///     |--- depends on -> inner support crate
///                           |
///                           |--- re-exports inner support items
///                           |
///                           |--- uses -> inner proc macro
/// ```
///
/// The inner proc macro may generate:
///
/// ```ignore
/// inner_support::Trait
/// ```
///
/// but the composed (nested) macro needs that reference to resolve through the outer
/// support crate:
///
/// ```ignore
/// outer_support::Trait
/// ```
///
/// The inner macro does not necessarily know that it is being used through
/// the outer layer. Delegation provides the mechanism by which the inner
/// pipeline can communicate that new crate-resolution context to the outer
/// generated syntax.
///
/// This creates a simple crate-identifier substitution:
///
/// ```text
/// delegated support crate
///         |
///         | delegation
///         |
/// support crate
/// ```
///
/// [`Self::delegate_crate`] identifies the crate whose references should be
/// replaced by the inner macro's [`Self::support_crate`] when such delegation
/// is required.
///
/// The replacement is performed by [`Self::delegate_inner`], which operates
/// directly on a [`syn`] syntax representation and replaces occurrences of the
/// delegated crate identifier with the support-crate identifier.
///
/// For example, conceptually:
///
/// before delegation:
/// ```ignore
/// outer_support::Trait
/// outer_support::Type
/// ```
/// after delegation:
/// ```ignore
/// inner_support::Trait
/// inner_support::Type
/// ```
///
/// Delegation is optional. A pipeline that is not being embedded inside
/// another proc-macro pipeline does not need to perform any replacement, so
/// the default implementation returns `None`.
///
/// # Why a delegate macro is required for attributes
///
/// Function-like and attribute procedural macros differ in how their input
/// and generated output participate in macro expansion.
///
/// With a function-like proc macro, an outer proc macro can construct another
/// macro invocation around its input. The inner macro invocation can
/// therefore be represented explicitly in the tokens being passed through
/// the expansion process. This makes it possible for the outer pipeline to
/// arrange for the inner expansion and subsequently rewrite the resulting
/// syntax.
///
/// For example, an outer macro can conceptually produce:
///
/// ```ignore
/// // here, we describe outer and inner as structures not
/// // dependencies
///
/// outer_macro! {
///     { "a user given literal expr example" }
/// }
///
/// outer_macro_of_inner! {
///     inner_macro! {
///         { "a user given literal expr example" }
///     }
/// }
///
/// outer_macro_of_inner_input!{
///      { "a inner macro transformed literal" }
/// }
/// ```
///
/// The inner macro expansion can then become part of the syntax that the
/// outer pipeline can work with.
///
/// Attribute macros have a different expansion relationship. An outer
/// attribute macro cannot inspect syntax that will only be generated later
/// by an inner attribute macro. In particular, an outer transformation cannot
/// retroactively modify the output of an inner attribute expansion merely by
/// rewriting the syntax it originally received.
///
/// For example, consider:
///
/// ```ignore
/// #[outer]
/// #[inner]
/// impl Sum for Marker { ... };
/// ```
///
/// The outer attribute receives the item as it exists at that point in the
/// expansion process. It cannot simply rewrite the syntax that `inner` will
/// produce later, because that generated syntax does not yet exist from the
/// outer macro's point of view.
///
/// Therefore, attribute delegation requires a second mechanism: a delegate
/// attribute that is attached to the input (pushed to the bottom)
/// which invokes after the inner attribute macro expands.
///
/// Conceptually, the expansion proceeds as:
///
/// ```text
/// original item
///             |
///       outer macro expands - adds delegate attribute
///             |
///       inner macro expands
///             |
///       delegate attribute runs
///             |
///       support-crate references are rewritten
/// ```
///
/// For example, the pipeline can arrange the attributes conceptually as:
///
/// ```ignore
/// #[inner_support::outer_macro] // since outer macros are re-expoted to inner support crate
/// #[inner_support::delegate_macro]
/// impl Sum for Marker { ... };
/// ```
///
/// The inner macro expands first and may produce:
///
/// ```ignore
/// outer_support::Trait
/// outer_support::Type
/// ```
///
/// The delegate attribute then receives that resulting syntax and rewrites
/// those references to:
///
/// ```ignore
/// inner_support::Trait
/// inner_support::Type
/// ```
///
/// The important point is that the delegate attribute does not perform the
/// inner transformation itself. Its purpose is to run at the correct point
/// in the expansion process so that the syntax produced by the re-exported
/// outer macro is available for path rewriting.
///
/// [`Self::delegate_macro`] provides the identifier of this delegate
/// attribute. The delegate macro is expected to perform the support-crate
/// replacement after the outer macro has produced its syntax.
///
/// [`Self::delegate_attribute`] attaches this delegate attribute to a syntax
/// item. The attribute is placed at the end of the existing attribute list so
/// that it can operate after the relevant outer macro expansion.
///
/// Both the delegate crate and delegate macro are optional. If either is not
/// configured, no delegate attribute is added.
///
/// # Direct delegation
///
/// [`Self::delegate_item`] is the direct form of delegation. It is used when
/// the current macro has access to the inner syntax (of outer macro) that needs
/// to be rewritten directly instead of passing an delegate attribute.
///
/// Especially in scenarios where attributes are not available to be passed for
/// macro expansion.
///
/// The operation is:
///
/// ```text
/// delegated-crate identifier
///         |
///         | replacement
/// support-crate identifier
/// ```
///
/// For example:
///
/// ```ignore
/// // input syntax:
/// outer_support::Trait
/// outer_support::Type
///
/// // replacement:
/// // outer_support -> inner_support
///
/// // resulting syntax:
/// inner_support::Trait
/// inner_support::Type
/// ```
///
/// The implementation is expressed in terms of [`PathQualifierReplace`] so that the
/// mechanism is independent of the particular `syn` syntax node being
/// processed.
///
/// This allows the same operation to be applied to different syntax
/// representations without making `SupportCrate` responsible for knowing
/// their concrete structure.
///
/// # Attribute delegation
///
/// [`Self::delegate_attribute`] is the deferred form of delegation. Instead
/// of immediately rewriting syntax, it places a delegate attribute onto the
/// syntax so that another macro expansion can occur first.
///
/// For example, if the syntax currently contains:
///
/// ```ignore
/// #[inner_support::outer_macro]
/// struct Marker;
/// ```
///
/// delegation can append:
///
/// ```ignore
/// #[inner_support::outer_macro]
/// #[inner_support::delegate_macro]
/// struct Marker;
/// ```
///
/// After the inner macro has expanded, the delegate macro receives the
/// resulting syntax and can perform the equivalent of:
///
/// ```text
/// outer_support -> inner_support
/// ```
///
/// This distinction is important because the two mechanisms solve different
/// expansion-order problems:
///
/// - direct delegation rewrites syntax that is already available;
/// - attribute delegation schedules the rewrite for syntax that will become
///   available only after an outer attribute macro expands.
///
/// # Using `delegate_inner` from a proc-macro attribute
///
/// [`Self::delegate_inner`] is intended to be called by the delegate attribute macro
/// provided by the proc-macro crate itself.
///
/// The delegate attribute is the bridge between the proc-macro expansion
/// mechanism and [`SupportCrate`]. The attribute receives the syntax produced
/// by the inner macro, parses that syntax into a `syn` representation, and
/// then asks the pipeline to perform the support-crate substitution.
///
/// For example, a proc-macro crate may expose a delegate attribute:
///
/// ```ignore
/// #[proc_macro_attribute]
/// pub fn delegate(attr: TokenStream, item: TokenStream) -> TokenStream {
///     let mut item: syn::Item = syn::parse_macro_input!(item);
///
///     Pipeline::delegate_inner(&mut item);
///
///     quote::quote!(#item).into()
/// }
/// ```
///
/// The important operation here is:
///
/// ```ignore
/// Pipeline::delegate_inner(&mut item);
/// ```
///
/// The delegate attribute does not need to know how delegate-crate identifiers
/// are represented or how they should be replaced towards support-crate.
/// That responsibility belongs to [`SupportCrate`].
///
/// For example, suppose the pipeline is configured with:
///
/// ```text
/// support crate  = inner_support
/// delegate crate = outer_support
/// ```
///
/// and the outer proc macro produces:
///
/// ```ignore
/// impl outer_support::Trait for Value {
///     fn method(&self) {}
/// }
/// ```
///
/// When the delegate attribute receives this generated item, it invokes:
///
/// ```ignore
/// Pipeline::delegate_inner(&mut item);
/// ```
///
/// which performs the configured identifier replacement:
///
/// ```text
/// outer_support
///       │
///       │ delegate_inner
///       │
/// inner_support
/// ```
///
/// resulting in:
///
/// ```ignore
/// impl inner_support::Trait for Value {
///     fn method(&self) {}
/// }
/// ```
///
/// This means the delegate attribute itself remains generic. It does not need
/// to contain a hard-coded replacement such as `outer_support -> inner_support`.
/// The pipeline supplies both identifiers through [`Self::support_crate`] and
/// [`Self::delegate_crate`].
///
/// The same mechanism can be used for any syntax representation implementing
/// [`PathQualifierReplace`]. For example, the delegate attribute may receive a
/// [`syn::Item`], [`syn::DeriveInput`], or another supported syntax node and
/// pass it directly to [`Self::delegate_inner`].
///
/// The complete responsibility is therefore divided into three parts:
///
/// ```text
/// proc-macro attribute
///        │
///        │ parses generated syntax
/// delegate_inner
///        │
///        │ obtains support_crate + delegate_crate
/// PathQualifierReplace
///        │
/// rewritten syntax
/// ```
///
/// This is the direct delegation path: the delegate attribute runs after the
/// inner macro has produced its syntax, and `delegate_inner` immediately
/// rewrites that already-available syntax.
pub trait SupportCrate {
    /// Returns the identifier of the normal support crate.
    ///
    /// Generated paths use this identifier to access the public support layer
    /// associated with the proc-macro pipeline.
    ///
    /// See [`SupportCrate`] documentation to understand the semantics.
    fn support_crate() -> syn::Ident;

    /// Returns the crate identifier whose references should be replaced by the
    /// normal support crate when this pipeline is being delegated through another
    /// proc macro.
    ///
    /// `None` means that no crate substitution is required.
    ///
    /// See [`SupportCrate`] documentation to understand the semantics.
    fn delegate_crate() -> Option<syn::Ident> {
        None
    }

    /// Returns the identifier of the attribute macro used to perform deferred
    /// support-crate delegation.
    ///
    /// This is used when an outer attribute macro generates syntax that cannot
    /// be rewritten by the inner macro directly.
    ///
    /// `None` means that no delegate attribute is available.
    ///
    /// See [`SupportCrate`] documentation to understand the semantics.
    fn delegate_macro() -> Option<syn::Ident> {
        None
    }

    /// Replaces references to the delegated support crate with the normal
    /// support crate in an already available syntax representation.
    ///
    /// If no delegated crate is configured, the syntax is left unchanged.
    ///
    /// ```ignore
    /// replace_idents(
    ///     from: delegate_crate
    ///     to: support_crate
    /// )
    /// ```
    ///
    /// See [`SupportCrate`] documentation to understand the semantics.
    fn delegate_inner<T: PathQualifierReplace>(input: &mut T) {
        let Some(from) = Self::delegate_crate() else {
            return;
        };

        let to = Self::support_crate();

        input.replace_path_qualifiers(&mut ReplacePathQualifier { from, to });
    }

    /// Appends the configured delegate attribute to a syntax representation.
    ///
    /// The delegate attribute is intentionally added rather than performing
    /// the replacement immediately. This allows an outer attribute macro to
    /// expand first, after which the delegate attribute can rewrite the
    /// delegate-crate references present in the generated syntax.
    ///
    /// If delegation has not been configured, this operation does nothing.
    ///
    /// See [`SupportCrate`] documentation to understand the semantics.
    fn delegate_attribute<T: PushAttribute>(input: &mut T) {
        let support_crate = Self::support_crate();

        let Some(delegate_macro) = Self::delegate_macro() else {
            return;
        };

        input.push_attribute(&mut AttributePush {
            attr: parse_quote!(#[#support_crate::#delegate_macro]),
        });
    }

    /// Performs direct delegation for an item.
    ///
    /// Item delegation uses the same identifier-replacement mechanism as
    /// [`Self::delegate_inner`], because the item's syntax is available for
    /// direct inspection at the point where delegation is performed.
    fn delegate_item<T: PathQualifierReplace>(input: &mut T) {
        Self::delegate_inner(input);
    }
}
