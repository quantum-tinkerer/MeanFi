# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2024-05-31

### Fixed
- Ensure {autolink}`~meanfi.params.rparams.tb_to_rparams` and {autolink}`~meanfi.params.rparams.rparams_to_tb` use minimal parametrisation of the tight-binding dictionary.

### Added
- Density matrix cost for the mean-field solver.
- Functionality {autolink}`~meanfi.kwant_helper.utils.tb_to_builder` to create `kwant` systems with the tight-binding dictionaries.
- Functionality {autolink}`~meanfi.kwant_helper.utils.tb_to_kfunc` to create k-dependent function from tight-binding format.

### Changed
- Rewrote {autolink}`~meanfi.kwant_helper.utils.builder_to_tb` to avoid potential bugs and remove the use of `copy`.

## [1.0.0] - 2024-05-09

- First release of _MeanFi_.
