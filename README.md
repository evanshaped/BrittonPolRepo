# Polarization Drift Data Processing

## Introduction
The Polarization Drift Project at UMD's Britton Laboratory aims to quantify noise at various timescales within a 27km burried fiber optic cable, with the hope of making polarization-drift-correction more efficient for quantum networking and polarization-based communication schemes.

Processing data for analysis was made highly difficult by:
1) The use of the Thorlabs PAX1000 digital polarimeter (whose measurements tended to be spaced non-uniformly in time)
2) The use of a single PAX to measure two time-alternating Polarization signals (we had to distinguish between real data and noise which was recorded while the signals were between transitions)

This repository contains python tools to process PAX data and distinguish between real data and noise. It can also visualize and perform preliminary statistical analysis of the data.

## Overview of tool usage
### Incoming data
States of Polarization (SOPs) live on a unit sphere (the Poincare Sphere), and are typically parametrized by three normalized [Stokes Parameters](https://en.wikipedia.org/wiki/Stokes_parameters), s<sub>1</sub>, s<sub>2</sub>, and s<sub>3</sub>. The following examples plot one or multiple of these quantities over time.

Below is a section of raw Stokes data, taken when measuring two time-alternating SOPs as we do in our experiment (note the "switching" between the two SOPs).

![Raw PAX data](screenshots/image1)

The "Time Between Samples" (TBS) is highly inconsistent, and occasionally we suffer exceedingly large lags between samples (one occurence is shown above). This poor performance is not specific to our experiment or polarimeter; [another](https://github.com/evanshaped/PAX-rate-visualization) github repository gives PAX users the tools to quantify just how poor the performance of their polarimeter is by visualizing the TBS distribution.

### Identifying switch times
The code first 
