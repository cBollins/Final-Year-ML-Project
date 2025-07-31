# Introduction

In preparation for this project tour, I must encourage the reader to fully familiarise themselves with this README especially since the final report `Final_Report_u2201924.pdf` was equally a comment on the physics of neutrino interactions and imaging detectors as it was the machine learning approaches deployed to classify them. While reading the full report gives a full picture, it is likely more worthwhile to instead engage with the `Project_Overview.ipynb` notebook provided, which will strip all particle physics theory away from the report, and provide a more accessible and comfortable picture.

For the extra brave, the Jupyter environment has been collected and placed into `/UG_Project_Raw`.

## Motivation

The universe is matter-dominated. That is, around 70% of the mass-energy in the cosmos is in the mass of both baryonic and non-baryonic (dark) matter. Antimatter is similar to matter; while the mass of an antimatter particle is the same as its matter counterpart, properties like electric charge and spin are inverse. If matter and antimatter particles intersect, a process called annihilation occurs, converting all mass into energy with 100% efficiency.

The early universe was significantly hot, enabling a process called baryogenesis. Heat energy was dense enough to spontaneously create matter-antimatter quark pairs, which typically annihilated back into energy. This symmetry was seemingly broken though, only enough to results in 1 matter quark for every $~2\times10^9$ annihilated pairs, resulting in a matter dominated universe. This symmetry break can begin to be explained by detecting CP (charge-parity) violation, and we think that in the neutrino sector, detecting significant characteristic differences between the behaviour of matter and antimatter neutrinos could bring us a step closer to CP violation. Neutrino oscillations is the phenomenon that observes a non-zero probability that a neutrino will change its flavour over its travel from source to target. Detecting different oscillation rate between $\nu$ and $\bar{\nu}$ is where this project takes the lead, and aids the classification of interactions in order to see the disappearance/reappearance.

## Imaging Detector

Imaging detectors are a recent innovation in the neutrino physics sector. Liquid Argon Time Projection Chambers (LArTPCs) provide the highest resolution output of any neutrino detector in history, DUNE is a large LArTPC currently under construction. Clearer images of neutrino interactions give machine learning models more information to extract features and classify the interaction.

The pasticles within an event leaves behind a trace of ionisation electrons, which are pulled across to three wire planes by an electric field, conventionally named `U, V, W` planes. The entirety of the analysis is performed on the `W` plane, also known as the collection plane, giving the best signal/noise ratio. In short, the samples that features are extracted from are a top-down projection of the event onto the $xz$ plane.

## Dataset

The reverse-engineered reconstruction data are stored in `.ROOT` files. Each reconstructed particle is indexed in the dataframe, and can be grouped via a key `event number`. A typical event contains 5-12 particles.

### Features + Truth values

1. `reco_hits`, The spatial co-ordinates on the collection plane, the entire image of the interaction/event.
2. `neutrino_vtx`, Co-ordinates of where the reconstruction detects the neutrino hitting the Argon nucleus.
3. `particle_vtx`, Co-ordinates where the particle decayed from its parent. The candidate particle is selected by the particle vertex closest to the neutrino vertex.
4. `is_nue, is_numu, is_cc`, Truth records of the event classification. We are looking to classify events into: `is_nue, is_numu` or `is_nc == !is_cc`. 
