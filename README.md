# Introduction

In preparation for this project tour, I must encourage the reader to **fully** familiarise themselves with this README especially since the final report `Final_Report_u2201924.pdf` was equally a comment on the physics of neutrino interactions and imaging detectors as it was the machine learning approaches deployed to classify them. While reading the full report gives a full picture, it is likely more worthwhile to instead engage with this and the `Project_Overview.ipynb` notebook provided, which will strip all particle physics theory away from the report, and provide a more accessible and comfortable picture.

For the extra brave, the Jupyter environment has been collected from the Noteable cloud and placed into `/UG_Project_Raw`. The .ROOT files have been put into the .gitignore since they are either corrupt or too large. Examples of the data will be given though.

## Motivation

The universe is matter-dominated. That is, around 70% of the mass-energy density in the cosmos is both baryonic and non-baryonic (dark) **matter**. Antimatter is similar to matter; while the mass of an antimatter particle is the same as its matter counterpart, properties like electric charge and spin are inverse. If matter and antimatter particles collide, a process called annihilation occurs, converting all mass into energy with 100% efficiency.

The early universe was significantly hot, enabling a process called baryogenesis. Heat energy was dense enough to spontaneously create matter-antimatter quark pairs, which typically annihilated back into energy. This symmetry was seemingly broken, only enough to produce 1 matter quark for every $~10^9$ annihilated pairs. These leftover quarks ultimately construct each nucleon in the universe today. Detecting **CP violation** (charge-parity) in the neutrino sector could lead discussions on why the symmetry was disrupted during baryogenesis. 

### Neutrino Oscillations

The physical phenomenon that observes a non-zero probability that a neutrino will **change** its flavour over its travel from source to target. Flavour of course being the associated lepton, we have $\nu_e, \nu_\mu, \nu_\tau$, and corresponding antimatter neutrinos of the same flavours. Detecting different oscillation rate between $\nu$ and $\bar{\nu}$ is where this project takes the reigns, and aids the classification of interactions in order to see the disappearance/reappearance.

$$
P(\nu_\alpha \longrightarrow \nu_\beta) = \sin^2(2\theta)\sin^2 \left( 1.27 \Delta m^2 \frac{L\text{ (km)}}{E \text{(GeV)}} \right)
$$

This project assumes the 'two flavour approximation' where only oscillations between $\nu_e \longleftrightarrow \nu_\mu$ are considered, for two distinct reasons:
1. While $\nu_\mu \longrightarrow \nu_\tau$ is the more common oscillation, the beam is not likely to create a tau neutrino energetic enough to create a daughter tau lepton in the far detector, it will likely create an event similar to a neutral current, which would introduce a confusion topology with an existing class.
2. Even if we do get a characteristic tau lepton, it will be far too similar to a muon.

In summary, the matter in this universe is here due to a mechanism causing CP violation during baryogenesis. We think that observing significant differences between $\nu$ oscillations compared to $\bar{\nu}$ oscillations is one of the ways to open up ideas about this mechanism.

## Imaging Detector

Imaging detectors are a recent innovation in the neutrino physics sector. Liquid Argon Time Projection Chambers (LArTPCs) provide the highest resolution output of any neutrino detector in history, DUNE is a large LArTPC currently under construction. Clearer images of neutrino interactions give machine learning models more information to extract features and classify the interaction.

The pasticles within an event leaves behind a trace of ionisation electrons, which are pulled across to three wire planes by an electric field, conventionally named `U, V, W` planes. The entirety of the analysis is performed on the `W` plane, also known as the collection plane, giving the best signal/noise ratio. In short, the samples that features are extracted from are a top-down projection of the event onto the $xz$ plane.

### Beam Production

The neutrino beam at is created at Fermilab, Illinois, and will be fired 1300km through the Earth's crust in the direction of these LArTPCs in DUNE. This is done by firing protons at a target material, which released charged particles ($\pi, K$ primarily). These particles are steered by a magnetic horn into a fine beam, they decay shortyl after and release muon neutrinos $\nu_\mu$. Reversing magnetic polarity will steer the oppositely charged pions and kaons into the beam, which will decay into muon antineutrinos. This is important, as the motivation is to detect differences between $\nu_\mu \longrightarrow \nu_e$ and $\bar{\nu_\mu} \longrightarrow \bar{\nu_e}$ oscillations.

### Near and Far detectors

In order to observe neutrino oscillations, we need to first observe the **beam composition** before the neutrinos have had a chance to do so. This is why the setup is composed of two detectors, the near detector at the start of the beam, and the far detector at DUNE. Neutrino flux is much higher at the near detector, which is why no imaging data is collected there. Each image would be highly perturbed by other elementary particles from other interactions. The far detector is where we get the most clarity in the images, and where, given an ideal classifier, $\nu_e$ appearance can be observed.

A full picture is given on the homepage of the [DUNE website](https://www.dunescience.org/).

## Dataset

The reverse-engineered reconstruction data are stored in `.ROOT` files. Each reconstructed particle is indexed in the dataframe, and can be grouped via a key `event number`. A typical event contains 5-12 particles.

### Features + Truth values

1. `reco_hits`, The spatial co-ordinates on the collection plane, the entire image of the interaction/event. When the ionisation electrons are absorbed by the wire planes, they create a Gaussian pulse. $w$ position is the position of the wire in the detector, and $x$ is calculated in terms of time and the mean drift velocity of the electric field. This is called 'electron drift reconstruction'.
2. `neutrino_vtx`, Co-ordinates of where the reconstruction detects the neutrino hitting the Argon nucleus.
3. `particle_vtx`, Co-ordinates where the particle decayed from its parent. The candidate particle is selected by the particle vertex closest to the neutrino vertex.
4. `is_nue, is_numu, is_cc`, Truth records of the event classification. We are looking to classify events into: `is_nue, is_numu` or `is_nc == !is_cc`.
5. `reco_adcs`, When the collection plane (`W` wires) collect the ionisation electrons, the pulse can be integrated and give an ADC (analogue to digital converter). This feature is proportional to the energy of the event at each hit.

### Cheated and Pandora

There are two types of files given by DUNE, these are called "Cheated" and "Pandora". The former is as it sounds, a perfect reconstruction of a reverse-engineered simulation of what we expect to see inside a LArTPC. The latter is a Pandora reconstruction of the reverse-engineered simulation. The Pandora reconstruction algorithm is currently the highest quality and most thorough that will be used by DUNE when construction is completed, and will give the best picture of what we will see in that event.

## Choosing Cheated files

While attacking the classification head on using the Pandora files seems to be the most valuable approach, this incidentally conflates two important problems:

- Classifying each event using ML approaches.
- Handling false reconstructions – we avoid this by choosing Cheated files.

This makes the project goals a lot clearer: to classify each **perfectly** reconstructed event using machine learning.

---

# Workflow

Understanding the workflow must include a little more physics background, first about the neutrinos that we are looking for.

## Neutrinos

Neutrally charged, extremely small and inert fundamental leptons that pair with their **charged** lepton. Neutrinos interact via the weak interaction and gravity. By 'pair', we mean that neutrinos have flavour. For example, solar neurinos are released during hydrogen burning through the proton-proton chain:

$$
4\text{H} \longrightarrow ^4\text{He} + 2e^+ + 2\nu_e
$$

2 positrons and electron neutrinos are released. Neutrinos have three flavours ($e, \mu, \tau$) for each charged lepton.

## Understanding event topology

Each interaction looks largely similar. We have a vertex and lots of daughter particles emerging from it – see `Project_Overview.ipynb` for examples. Each of these events can be split into three classes:

1. CC$\nu_e$, charged current electron neutrino interaction. A characteristic primary electron can be spotted.
2. CC$\nu_\mu$, charged current muon neutrino interaction. Similarly, we have a characteristic primary muon.
3. NC$\nu_x$, neutral current. There is no litmus lepton to know the flavour of neutrino.

## Tracks vs Showers

Two important types of particle we see within these events are track-like and shower-like. Tracks are undecaying, inert lines and showers are cascading, chaotic exponentiating decays. The mechanism for an elextromagnetic shower is Bremsstrahlung radiation.

The importance of classifying track and shower particles lies in the fact that we are eventually trying to classify candidate leptons. Muons leave a track-like trail in the detector, and electrons will decay through Bremsstrahlung and create a shower.

## Track vs Shower Likelihood

Engineered 6 features that show mediocre to strong separation between track and shower
