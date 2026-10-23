# Project Proposal: Dynamic Sweet-Spot Calibration for Multi-Channel Audio via Edge-Vision Listener Tracking

---

## 1. Executive Summary

Traditional multi-channel surround sound systems (such as 5.1 or 7.1 setups) deliver optimal spatial staging and phase coherence only within a tightly constrained physical area known as the "sweet spot." When a listener shifts posture or moves off-center, inter-channel phase alignment degrades, collapsing localization and destabilizing the soundstage.

This project develops an edge-assisted, real-time spatial audio calibration system that uses a single camera to track listener coordinates and continuously adjust multi-channel loudspeaker feeds. By offloading low-latency vision tracking to an edge compute accelerator and applying lightweight geometric acoustic digital signal processing (DSP), the system dynamically anchors the acoustic sweet spot to the listener’s actual position without relying on wearable sensors or specialized beamforming transducer arrays.

---

## 2. Problem Statement

- **Rigid Sweet-Spot Boundaries:** Multi-channel home and desktop audio systems rely on static calibrations (e.g., AVR microphone setups). Any minor physical translation by the listener breaks the intended delay and amplitude balances.
- **Intrusiveness of Existing Adaptive Solutions:** Modern spatial audio head-tracking predominantly targets headphones via wearable IMUs. Hardware-based speaker solutions require expensive, proprietary phased-array soundbars rather than supporting generic, pre-existing loudspeaker setups.
- **Host Compute Interference:** Running high-frame-rate computer vision tasks concurrently with low-latency audio processing on a single host computer introduces scheduling jitter, thermal throttling, and perceptible input-to-sound lag.

---

## 3. Project Objectives

- **Real-Time 3D Pose Estimation at the Edge:** Track listener ear/head coordinates $(x, y, z)$ and orientation relative to a fixed speaker reference coordinate frame using edge neural processing, minimizing host overhead.
- **Artifact-Free Dynamic Audio DSP:** Build a low-latency multi-channel DSP engine that computes time-of-flight (ITD) compensations and off-axis frequency corrections without audible pitch wobbles (Doppler glitches) or zipper noise.
- **Seamless Pass-Through Architecture:** Provide an unobtrusive bridge between multi-channel audio outputs (gaming, cinema, music) and physical speaker hardware.

---

## 4. System Architecture

```
                  +-----------------------------------+
                  |     Monocular Vision Sensor       |
                  +-----------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------+
| Edge AI Processing Unit (e.g., Qualcomm Edge Development Kit)      |
|                                                                    |
|  [ NPU / Vision Pipeline ]                                         |
|  - 3D Landmark / Pose Inference                                    |
|  - Coordinate Transformation (Camera Frame -> Speaker Grid)        |
|  - Temporal Smoothing & Jitter Rejection (Predictive Filtering)    |
|                                                                    |
|  [ Control Signal Dispatch ]                                       |
|  - Real-time parameter stream (Delays, Gain Trims, Off-Axis EQ)     |
+--------------------------------------------------------------------+
                                    | Control Parameters
                                    v
+--------------------------------------------------------------------+
| Multi-Channel Audio Processing Engine                              |
|                                                                    |
|  [ Multi-Channel In (5.1 / 7.1 Stream) ]                           |
|       |                                                            |
|       v                                                            |
|  [ Fractional Delay Lines with Slew-Rate Limiting ]                |
|       |                                                            |
|  [ Inverse-Distance Level Compensation & Off-Axis Filtering ]      |
|       |                                                            |
|       v                                                            |
|  [ Multi-Channel DAC Output ] ----> [ Physical Loudspeaker Setup ] |
+--------------------------------------------------------------------+

```

---

## 5. Technical Approach

### A. Edge Vision Pipeline

- Run an optimized, quantized landmark detection model on an edge development kit to extract facial geometry and ear positions.
- Convert 2D image coordinates and scale into relative 3D space vector offsets.
- Filter measurement noise and frame-rate disparities through predictive extrapolation, providing low-jitter coordinates to the DSP stage even during brief head turns or partial occlusions.

### B. Geometric Acoustic DSP Engine

- **Dynamic Time Alignment:** Determine distance vectors from each speaker to the listener's instantaneous ear position. Apply sub-sample fractional delays across channels to align arrival times at the listener's ears.
- **Slew-Rate Limiting:** Smooth buffer delay transitions over windowed intervals to eliminate pitch modulations and click artifacts caused by sudden head movements.
- **Directional Attenuation Compensation:** Apply gentle shelving filters to correct high-frequency roll-off caused when the listener moves away from a speaker's primary acoustic axis.

---

## 6. Implementation Milestones

| Milestone                                  | Key Deliverable                                                                                                                                 | Focus Area                                                     |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Phase 1: Proof of Concept**              | Software simulation on desktop environment (Python/OSC + multi-channel delay patch) verifying listening impressions and delay-smoothing limits. | DSP stability, artifact elimination                            |
| **Phase 2: Edge Vision Integration**       | Deploying the 3D landmark model onto the edge development board; establishing real-time parameter streaming over local transport.               | Inference latency, FPS optimization, power profiling           |
| **Phase 3: Hardware & Audio Loop Closure** | Routing live multi-channel audio through the DSP pipeline driven directly by the edge board's tracking stream.                                  | End-to-end latency, spatial stability, edge-to-DAC integration |
| **Phase 4: Evaluation & Demonstration**    | Quantitative measurement of latency and acoustic phase coherence; subjective user testing across varying seating positions.                     | Validation and showcase setup                                  |

---

## 7. Expected Outcomes & Impact

- **Autonomous Adaptive Audio:** Demonstrates how off-the-shelf, non-array speakers can achieve dynamic tracking without requiring users to wear headsets or keep their heads pinned to a single point in space.
- **Practical Edge Computing Showcase:** Highlights the utility of edge NPU acceleration for latency-sensitive tasks where host-machine interference must be strictly avoided.
- **Modular Usability:** Sets the foundation for a plug-and-play smart controller usable in desktop workstations, living room home theaters, or accessibility-focused acoustic installations.
