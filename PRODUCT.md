# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary users are clinicians, rehabilitation researchers, and BCI operators running a patient session from a desktop workstation. They need to prepare a patient and task, verify hardware and signal state, record EEG, place motor-imagery markers, observe inference, provide feedback, train or load a model, and adjust learning safeguards without losing situational awareness.

## Product Purpose

BrainBridge coordinates a complete motor-imagery BCI session across EEG acquisition, patient-linked recordings, patient calibration, live inference, VR and orthosis outputs, online reinforcement feedback, and model training. Success means an operator can understand the system state at a glance, complete a session safely, and recover from missing prerequisites without guessing.

## Positioning

BrainBridge connects patient-specific EEG acquisition and learning directly to rehabilitation outputs in one operator workflow: signal, marker, model prediction, VR/orthosis action, corrective feedback, and retraining remain visibly linked instead of living in separate tools.

## Operating Context

The product is used on a desktop during supervised clinical or research sessions. The live workspace must prioritize patient, task, connection, recording, signal quality, inference, and markers. Secondary workflows cover patient records and sessions, model training and loading, and runtime configuration. Portuguese is the current interface language. Some hardware can be absent, so simulation and offline states are normal operating states rather than exceptional failures.

## Capabilities and Constraints

- Preserve all existing API-backed behavior and workflow branches.
- Preserve the four primary areas: live session, patients, training, and settings.
- Preserve EEG visualization, task selection, recording, markers, live prediction, VR/orthosis mirroring, feedback, RL restore, patient/session management, training checks, model loading, and configuration editing.
- The frontend is React 18 with Vite and is packaged as a Tauri desktop application.
- The interface must remain usable when the backend or individual devices are offline.
- Existing uncommitted repository work outside the interface is user-owned and must not be overwritten.

## Brand Commitments

Keep the BrainBridge name and its emphasis on bridging neural intent to rehabilitation action. The interface should feel clinically trustworthy, elegant, calm under pressure, and precise rather than playful or futuristic for its own sake.

## Evidence on Hand

The repository contains the working React interface, API client, live EEG canvas, backend routes, patient/session data flows, model training flows, and device states. No approved logo system, clinical photography, testimonials, or commercial claims are present and none should be fabricated.

## Product Principles

1. State before decoration: the operator always knows what is connected, selected, recording, and ready.
2. One session story: patient, task, signal, prediction, action, and feedback read as a continuous workflow.
3. Progressive complexity: the live task stays immediate while advanced controls remain findable.
4. Clinical confidence: terminology, contrast, focus, and recovery guidance support careful operation.
5. Function is preserved even when visual structure and implementation are replaced.

## Accessibility & Inclusion

Controls must be keyboard reachable, focus-visible, legible at common desktop and narrow widths, distinguish state without relying on color alone, respect reduced motion, and maintain WCAG AA contrast for text and controls.
