# Physics and modelling of Batteries

## Overview

* In the electrodes and binder
  * Conservation of intercalated lithium and electrons
  * Transport of lithium in the electrodes
    * Nonlinear diffusion depending on lithium concentration
  * Transport of electrons in the electrodes _and_ binder
    * Ohm's law
* In the electrolyte
  * Coupled transport of lithium positive and negative ions and electrons
  * "Concentrated electrolyte theory" (Stefan-Maxwell equations)
* Lithium ion exchange between the electrode particles and the electrolyte
  * Butler-Volmer equation
  * Electric potentials and lithium ion concentrations on either side of the interface

## Equations

* ELECTRODE
  * _For_ lithium concentration: 
    * Transient Poisson equation with heterogeneous concentration-dependent diffusivity
  * _For_ electric potential:
    * Steady-state Poisson equation with heterogeneous electronic conductivity

* BINDER
  * _For_ lithium concentration: 
    * Lithium _does not_ diffuse through binder
  * _For_ electric potential:
    * Steady-state Poisson equation with heterogeneous spatially-dependent [?] electronic conductivity

* ELECTROLYTE
  * Charge neutrality: Concentration of pos. ions = Concentration of neg. ions
  * _For_ positive (bzw. negative) lithium ion concentrations
    * Transient Poisson equation with heterogeneous diffusivity
    * [!] Diffusivity is concentration- _and_ charge-dependent
  * _For_ electric potential:
    * Steady-state Poisson equation with heterogeneous electronic conductivity
    * [!] Conductivity depends on ion concentration and electric potential
  * For a closed model, work with ion or counterion conservation
  * Material parameters depend on ion concentration and temperature

* INTERFACE CONDITIONS
  * Conservation of charge and lithium ions on the interface
  * Jump of fluxes of electrons and ion concentration non-zero
    * (De)intercalation current density
      * Depends on lithium concentrations and electric potentials 

* BOUNDARY CONDITIONS
  * _For_ the electrode
    * On the current-collector
      * _For_ ion concentration
        * Homogeneous Neumann
      * _For_ electric potential
        * Data is voltage: Dirichlet
        * Data is intensity: Integral of fluxes [!]
    * Elsewhere (including electrode-separator)
      * _For_ ion concentration and electric potential
        * Homogeneous Neumann 
  * _For_ the electrolyte
    * Everywhere
      * _For_ ion concentration and electric potential
        * Homogeneous Neumann
  * I guess full Neumann for ion concentration is alright if (nonlinear) couplings

* INITIAL CONDITIONS
  * Initial concentrations
  * What about electric potentials?

## Implementation remarks

* Bad news
  * Advanced Gridap user
  * Linear problem is already rather slow
    * 10 minutes
    * Although transient trimaterial 3D
* Good news
  * It works :)
  * Possible "hand-written" Jacobian
  * Mostly incremental modifications of weak form

## Ideas

* Github repo + project?
* List pairs (incremental steps, unit testing) until whole system
* Meet every now and then to advance together. 1 goal at a time?

## TO-DO

* Output results on all regions
* Renaming
  * electrode - solid  (s)
  * electrolyte - liquid (e)
  * binder (b)
* Optimize degrees

* Create repo

* Jacobian of nonlinear term
* Beware of negative roots

* Check manufactured for bimaterial

## Remarks

* Conductivities are not spatially-dependent, but possibly time-dependent
* Assuming t+ = cnt, 2nd term in (6c) is zero (because of (6b))
* t+ /= cnt at the end of the list
* dmu/dc is measured data (a function of concentration)
* For now, ignore Faraday constant (F)