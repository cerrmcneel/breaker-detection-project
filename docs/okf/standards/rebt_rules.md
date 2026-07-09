---
type: Regulatory Standard
title: REBT (Spanish Electrical Safety Code)
description: Regulations governing low-voltage domestic electrical installations in Spain.
tags: [compliance, standards, rebt]
timestamp: 2026-06-22T16:50:00Z
---

# REBT Safety Code

This concept formalizes Spain's *Reglamento Electrotécnico para Baja Tensión* (REBT), specifically for domestic *Cuadros Generales de Mando y Protección* (CGMP).

## Key Topological Regulations

1. **IGA (Main Breaker) & DPS (Surge Protection):**
   - The panel must start with a general magnetothermic breaker (IGA) of at least 25A (2 modules wide).
   - Modern standards require a coordinated surge protection device (DPS) connected in parallel or integrated (OVERSURGE).
   - Permanent overvoltage protection devices must protect the entire interior installation and be rated for a maximum overvoltage threshold of $440\text{ V}$.

2. **RCD (Diferencial) Groupings:**
   - Standard RCD sensitivity must be $30\text{ mA}$ for human shock protection.
   - **Rule of 5:** A single RCD can protect a maximum of 5 downstream circuit breakers (MCBs/PIAs). If a panel contains 6 or more circuits, additional RCD sub-groups must be created.
   - For sensitive/electronic appliances (e.g. computer servers, heat pumps), Class A superimmunized differentials (RCD_SI) are required.

3. **Circuit (PIA) Sizing:**
   - **C1 (Lighting):** Max 10A breaker. Wire gauge: $1.5\text{ mm}^2$.
   - **C2 (General Outlets & Fridge):** Max 16A breaker. Wire gauge: $2.5\text{ mm}^2$.
   - **C3 (Stove & Oven):** Max 25A breaker. Wire gauge: $6.0\text{ mm}^2$.
   - **C4 (Washing Machine, Dishwasher, Water Heater):** Max 20A breaker. Wire gauge: $4.0\text{ mm}^2$.
   - **C5 (Kitchen & Bathroom Outlets):** Max 16A breaker. Wire gauge: $2.5\text{ mm}^2$.
   - **C13 (Electric Vehicle Charging):** Dedicated circuit for domestic EV charging infrastructure (ITC-BT-52). For single-family homes, its protections (breaker and coordinated surge protection) can be integrated directly inside the main domestic CGMP.

This rule structure forms the context utilized by the [Schematic Generator](/tools/schematic_generator.md).
