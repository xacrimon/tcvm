use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, quote, quote_spanned};
use syn::{
    parse::{Parse, ParseStream},
    spanned::Spanned,
    visit_mut::VisitMut,
};
use synstructure::{AddBounds, decl_derive};

fn collect_derive(s: synstructure::Structure) -> TokenStream {
    fn find_collect_meta(attrs: &[syn::Attribute]) -> syn::Result<Option<&syn::Attribute>> {
        let mut found = None;
        for attr in attrs {
            if attr.path().is_ident("collect") && found.replace(attr).is_some() {
                return Err(syn::parse::Error::new_spanned(
                    attr.path(),
                    "Cannot specify multiple `#[collect]` attributes! Consider merging them.",
                ));
            }
        }

        Ok(found)
    }

    // Deriving `Collect` must be done with care, because an implementation of `Drop` is not
    // necessarily safe for `Collect` types. This derive macro has three available modes to ensure
    // that this is safe:
    //   1) Require that the type be 'static with `#[collect(require_static)]`.
    //   2) Prohibit a `Drop` impl on the type with `#[collect(no_drop)]`
    //   3) Allow a custom `Drop` impl that might be unsafe with `#[collect(unsafe_drop)]`. Such
    //      `Drop` impls must *not* access garbage collected pointers during `Drop::drop`.
    #[derive(PartialEq)]
    enum Mode {
        RequireStatic,
        NoDrop,
        UnsafeDrop,
    }

    let mut mode = None;
    let mut override_bound = None;
    let mut gc_lifetime = None;

    fn usage_error(meta: &syn::meta::ParseNestedMeta, msg: &str) -> syn::parse::Error {
        meta.error(format_args!(
            "{msg}. `#[collect(...)]` requires one mode (`require_static`, `no_drop`, or `unsafe_drop`) and optionally `bound = \"...\"`."
        ))
    }

    let result = match find_collect_meta(&s.ast().attrs) {
        Ok(Some(attr)) => attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("bound") {
                if override_bound.is_some() {
                    return Err(usage_error(&meta, "multiple bounds specified"));
                }

                let lit: syn::LitStr = meta.value()?.parse()?;
                override_bound = Some(lit);
                return Ok(());
            }

            if meta.path.is_ident("gc_lifetime") {
                if gc_lifetime.is_some() {
                    return Err(usage_error(&meta, "multiple `'gc` lifetimes specified"));
                }

                let lit: syn::Lifetime = meta.value()?.parse()?;
                gc_lifetime = Some(lit);
                return Ok(());
            }

            meta.input.parse::<syn::parse::Nothing>()?;

            if mode.is_some() {
                return Err(usage_error(&meta, "multiple modes specified"));
            } else if meta.path.is_ident("require_static") {
                mode = Some(Mode::RequireStatic);
            } else if meta.path.is_ident("no_drop") {
                mode = Some(Mode::NoDrop);
            } else if meta.path.is_ident("unsafe_drop") {
                mode = Some(Mode::UnsafeDrop);
            } else {
                return Err(usage_error(&meta, "unknown option"));
            }
            Ok(())
        }),
        Ok(None) => Ok(()),
        Err(err) => Err(err),
    };

    if let Err(err) = result {
        return err.to_compile_error();
    }

    let Some(mode) = mode else {
        panic!(
            "{}",
            "deriving `Collect` requires a `#[collect(...)]` attribute"
        );
    };

    let where_clause = if mode == Mode::RequireStatic {
        quote!(where Self: 'static)
    } else {
        override_bound
            .as_ref()
            .map(|x| {
                x.parse()
                    .expect("`#[collect]` failed to parse explicit trait bound expression")
            })
            .unwrap_or_else(|| quote!())
    };

    let mut errors = vec![];

    let collect_impl = if mode == Mode::RequireStatic {
        let mut impl_struct = s.clone();
        impl_struct.add_bounds(AddBounds::None);
        impl_struct.gen_impl(quote! {
            gen unsafe impl<'gc> ::tcvm::dmm::Collect<'gc> for @Self #where_clause {
                const NEEDS_TRACE: bool = false;
            }
        })
    } else {
        let mut impl_struct = s.clone();

        let mut needs_trace_expr = TokenStream::new();
        quote!(false).to_tokens(&mut needs_trace_expr);

        let mut static_bindings = vec![];

        // Ignore all bindings that have `#[collect(require_static)]` For each binding with
        // `#[collect(require_static)]`, we push a bound of the form `FieldType: 'static` to
        // `static_bindings`, which will be added to the genererated `Collect` impl. The presence of
        // the bound guarantees that the field cannot hold any `Gc` pointers, so it's safe to ignore
        // that field in `needs_trace` and `trace`
        impl_struct.filter(|b| match find_collect_meta(&b.ast().attrs) {
            Ok(Some(attr)) => {
                let mut static_binding = false;
                let result = attr.parse_nested_meta(|meta| {
                    if meta.input.is_empty() && meta.path.is_ident("require_static") {
                        static_binding = true;
                        static_bindings.push(b.ast().ty.clone());
                        Ok(())
                    } else {
                        Err(meta.error("Only `#[collect(require_static)]` is supported on a field"))
                    }
                });
                errors.extend(result.err());
                !static_binding
            }
            Ok(None) => true,
            Err(err) => {
                errors.push(err);
                true
            }
        });

        for static_binding in static_bindings {
            impl_struct.add_where_predicate(syn::parse_quote! { #static_binding: 'static });
        }

        // `#[collect(require_static)]` only makes sense on fields, not enum variants. Emit an error
        // if it is used in the wrong place
        if let syn::Data::Enum(..) = impl_struct.ast().data {
            for v in impl_struct.variants() {
                for attr in v.ast().attrs {
                    if attr.path().is_ident("collect") {
                        errors.push(syn::parse::Error::new_spanned(
                            attr.path(),
                            "`#[collect]` is not suppported on enum variants",
                        ));
                    }
                }
            }
        }

        // We've already called `impl_struct.filter`, so we we won't try to include `NEEDS_TRACE`
        // for the types of fields that have `#[collect(require_static)]`
        for v in impl_struct.variants() {
            for b in v.bindings() {
                let ty = &b.ast().ty;
                // Resolving the span at the call site makes rustc emit a 'the error originates a
                // derive macro note' We only use this span on tokens that need to resolve to items
                // (e.g. `gc_arena::Collect`), so this won't cause any hygiene issues
                let call_span = b.ast().span().resolved_at(Span::call_site());
                quote_spanned!(call_span=>
                    || <#ty as ::tcvm::dmm::Collect>::NEEDS_TRACE
                )
                .to_tokens(&mut needs_trace_expr);
            }
        }
        // Likewise, this will skip any fields that have `#[collect(require_static)]`
        let trace_body = impl_struct.each(|bi| {
            // See the above handling of `NEEDS_TRACE` for an explanation of this
            let call_span = bi.ast().span().resolved_at(Span::call_site());
            quote_spanned!(call_span=>
                {
                    // Use a temporary variable to ensure that all tokens in the call to
                    // `gc_arena::Collect::trace` have the same hygiene information. If we used
                    // #bi directly, then we would have a mix of hygiene contexts, which would
                    // cause rustc to produce sub-optimal error messagse due to its inability to
                    // merge the spans. This is purely for diagnostic purposes, and has no effect
                    // on correctness
                    let bi = #bi;
                    cc.trace(bi);
                }
            )
        });

        // If we have no configured `'gc` lifetime and the type has a *single* generic lifetime, use
        // that one.
        if gc_lifetime.is_none() {
            let mut all_lifetimes =
                impl_struct
                    .ast()
                    .generics
                    .params
                    .iter()
                    .filter_map(|p| match p {
                        syn::GenericParam::Lifetime(lt) => Some(lt),
                        _ => None,
                    });

            if let Some(lt) = all_lifetimes.next() {
                if all_lifetimes.next().is_none() {
                    gc_lifetime = Some(lt.lifetime.clone());
                } else {
                    panic!(
                        "deriving `Collect` on a type with multiple lifetime parameters requires a `#[collect(gc_lifetime = ...)]` attribute"
                    );
                }
            }
        };

        if override_bound.is_some() {
            impl_struct.add_bounds(AddBounds::None);
        } else {
            impl_struct.add_bounds(AddBounds::Generics);
        };

        if let Some(gc_lifetime) = gc_lifetime {
            impl_struct.gen_impl(quote! {
                gen unsafe impl ::tcvm::dmm::Collect<#gc_lifetime> for @Self #where_clause {
                    const NEEDS_TRACE: bool = #needs_trace_expr;

                    #[inline]
                    fn trace<Trace: ::tcvm::dmm::collect::Trace<#gc_lifetime>>(&self, cc: &mut Trace) {
                        match *self { #trace_body }
                    }
                }
            })
        } else {
            impl_struct.gen_impl(quote! {
                gen unsafe impl<'gc> ::tcvm::dmm::Collect<'gc> for @Self #where_clause {
                    const NEEDS_TRACE: bool = #needs_trace_expr;

                    #[inline]
                    fn trace<Trace: ::tcvm::dmm::collect::Trace<'gc>>(&self, cc: &mut Trace) {
                        match *self { #trace_body }
                    }
                }
            })
        }
    };

    let drop_impl = if mode == Mode::NoDrop {
        let mut drop_struct = s.clone();
        drop_struct.add_bounds(AddBounds::None).gen_impl(quote! {
            gen impl ::tcvm::dmm::__MustNotImplDrop for @Self {}
        })
    } else {
        quote!()
    };

    let errors = errors.into_iter().map(|e| e.to_compile_error());
    quote! {
        #collect_impl
        #drop_impl
        #(#errors)*
    }
}

decl_derive! {
    [Collect, attributes(collect)] =>
    /// Derives the `Collect` trait needed to trace a gc type.
    ///
    /// To derive `Collect`, an additional attribute is required on the struct/enum called
    /// `collect`. This has several optional arguments, but the only required argument is the derive
    /// strategy. This can be one of
    ///
    /// - `#[collect(require_static)]` - Adds a `'static` bound, which allows for a no-op trace
    ///   implementation. This is the ideal choice where possible.
    /// - `#[collect(no_drop)]` - The typical safe tracing derive strategy which only has to add a
    ///   requirement that your struct/enum does not have a custom implementation of `Drop`.
    /// - `#[collect(unsafe_drop)]` - The most versatile tracing derive strategy which allows a
    ///   custom drop implementation. However, this strategy can lead to unsoundness if care is not
    ///   taken (see the above explanation of `Drop` interactions).
    ///
    /// The `collect` attribute also accepts a number of optional configuration settings:
    ///
    /// - `#[collect(bound = "<code>")]` - Replaces the default generated `where` clause with the
    ///   given code. This can be an empty string to add no `where` clause, or otherwise must start
    ///   with `"where"`, e.g., `#[collect(bound = "where T: Collect")]`. Note that this option is
    ///   ignored for `require_static` mode since the only bound it produces is `Self: 'static`.
    ///   Also note that providing an explicit bound in this way is safe, and only changes the trait
    ///   bounds used to enable the implementation of `Collect`.
    ///
    /// - `#[collect(gc_lifetime = "<lifetime>")]` - the `Collect` trait requires a `'gc` lifetime
    ///   parameter. If there is no lifetime parameter on the type, then `Collect` will be
    ///   implemented for all `'gc` lifetimes. If there is one lifetime on the type, this is assumed
    ///   to be the `'gc` lifetime. In the very unusual case that there are two or more lifetime
    ///   parameters, you must specify *which* lifetime should be used as the `'gc` lifetime.
    ///
    /// Options may be passed to the `collect` attribute together, e.g.,
    /// `#[collect(no_drop, bound = "")]`.
    ///
    /// The `collect` attribute may also be used on any field of an enum or struct, however the
    /// only allowed usage is to specify the strategy as `require_static` (no other strategies are
    /// allowed, and no optional settings can be specified). This will add a `'static` bound to the
    /// type of the field (regardless of an explicit `bound` setting) in exchange for not having
    /// to trace into the given field (the ideal choice where possible). Note that if the entire
    /// struct/enum is marked with `require_static` then this is unnecessary.
    collect_derive
}

// Not public API; implementation detail of `gc_arena::Rootable!`.
// Replaces all `'_` lifetimes in a type by the specified named lifetime.
// Syntax: `__unelide_lifetimes!('lt; SomeType)`.
#[doc(hidden)]
#[proc_macro]
pub fn __unelide_lifetimes(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    struct Input {
        lt: syn::Lifetime,
        ty: syn::Type,
    }

    impl Parse for Input {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let lt: syn::Lifetime = input.parse()?;
            let _: syn::Token!(;) = input.parse()?;
            let ty: syn::Type = input.parse()?;
            Ok(Self { lt, ty })
        }
    }

    struct UnelideLifetimes(syn::Lifetime);

    impl VisitMut for UnelideLifetimes {
        fn visit_lifetime_mut(&mut self, i: &mut syn::Lifetime) {
            if i.ident == "_" {
                *i = self.0.clone();
            }
        }
    }

    let mut input = syn::parse_macro_input!(input as Input);
    UnelideLifetimes(input.lt).visit_type_mut(&mut input.ty);
    input.ty.to_token_stream().into()
}
