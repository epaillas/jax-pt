# Prediction Flow

This page documents the current execution flow for `jaxpt` power-spectrum
prediction calls and the Taylor-emulator build path that sits on top of the
theory layer.

The canonical entrypoint is now the theory layer in `jaxpt.theories`, typically
`GalaxyPowerSpectrumMultipolesTheory(...)` or
`ClassPTGalaxyPowerSpectrumMultipolesTheory(...)`.

```mermaid
flowchart TD
    A["GalaxyPowerSpectrumMultipolesTheory(...) or ClassPTGalaxyPowerSpectrumMultipolesTheory(...)"] --> B["theory(parameters)"]
    B --> C["template.resolve(cosmology_overrides)"]
    C -->|"jaxpt backend"| D["compute_basis(linear_input, settings, k?)"]
    C -->|"CLASS-PT backend"| J["predict_classpt_multipoles(...)"]

    subgraph Jaxpt["Jaxpt basis construction"]
        D --> D1{"k override?"}
        D1 -->|"yes"| D2["project basis terms to output k"]
        D1 -->|"no"| D3["use input k grid directly"]
        D2 --> F["compute_tree_level_basis(linear_input, settings, output_k)"]
        D3 --> F["compute_tree_level_basis(linear_input, settings)"]

        F --> F1["compute_real_tree_matter(...)"]
        F --> F2["compute_counterterm_shape(...)"]
        F --> F3["compute_real_loop_terms(...)"]
        F --> F4["compute_linear_rsd_terms(...)"]
        F --> F5["compute_counterterm_multipoles(...)"]
        F --> F6["compute_rsd_loop_terms(...)"]

        subgraph JaxptLoops["Current jaxpt loop implementation"]
            F3 --> G0{"settings.loop_order"}
            G0 -->|"tree"| G1["return zero-valued real-space loop terms"]
            G0 -->|"one_loop"| G2["compute_fftlog_realspace_terms(...)"]
            G2 --> G3["FFTLog coefficient extraction\njnp.fft.fft(...)"]
            G2 --> G4["analytic real-space kernels\nM13, M22, J, IFG2"]
            G2 --> G5["vectorized spectral contractions"]
            F6 --> G6["compute_fftlog_rsd_terms(...)"]
        end

        F1 --> H["make_basis(...)"]
        F2 --> H
        F3 --> H
        F4 --> H
        F5 --> H
        F6 --> H
        H --> I["BasisSpectra"]
    end

    I --> K["galaxy_multipoles(basis, nuisance_params)"]
    J --> K["MultipolePrediction\nP0(k), P2(k), P4(k)"]
```

Tree-level and one-loop predictions share the same basis-construction pipeline, with `settings.loop_order` switching the loop stages between zeros and the analytic FFTLog one-loop terms.

## Emulator Flow

The multipole-emulator entrypoint is the training script
[`scripts/build_taylor_emulator.py`](/Users/epaillas/code/jax-pt/scripts/build_taylor_emulator.py)
or the lower-level helper `build_multipole_emulator(...)`.

```mermaid
flowchart TD
    A["build_taylor_emulator.py"] --> B["PowerSpectrumTemplate(...)"]
    B --> C["GalaxyPowerSpectrumMultipolesTheory(...)"]
    C --> D["build_multipole_emulator(theory, ...)"]
    D --> E["TaylorEmulator(...)"]
    E --> F{"hashed file exists?"}
    F -->|"yes"| G["load taylor_<hash>.npz"]
    F -->|"no"| H["finite-difference theory evaluations"]
    H --> I["progress callback / CLI progress bar"]
    I --> J["save taylor_<hash>.npz"]
    G --> K["predict(parameters)"]
    J --> K["predict(parameters)"]
```

During evaluation, the emulator accepts any valid theory parameter name but
raises if a valid non-emulated parameter is varied away from its fiducial
value.

## Jaxpt Real-Space Flow

The jaxpt real-space prediction path reuses the same basis-construction stage and then assembles
the observable in `jaxpt/bias.py`.

```mermaid
flowchart TD
    A["LinearPowerInput"] --> B["compute_basis(linear_input, settings, k?)"]
    B --> C["BasisSpectra"]
    C --> D{"observable"}
    D -->|"matter"| E["matter_real_spectrum(basis, cs)"]
    D -->|"galaxy"| F["galaxy_real_spectrum(basis, b1, b2, bG2, bGamma3, cs, cs0, Pshot)"]
    E --> G["real-space P_mm(k)"]
    F --> H["real-space P_gg(k)"]
```

The assembly layer is shared between jaxpt and synthetic test bases. `CLASS-PT` remains the
validation oracle, but it is no longer a basis-construction branch in the public API.
