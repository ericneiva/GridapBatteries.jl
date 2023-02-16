* 3 harmonic averages
* 3 w+
* 3 w-
* 3 jumps (beware of direction of normal)
* 3 forces
* 3 conductivities
* 3 averages

* coupling bilinear forms
* First without c

* j_Γ and g_Γ are not debugged

  dv_ed = get_fe_basis(V_ed)
  dv_bd = get_fe_basis(V_bd)

  du_ed = get_trial_fe_basis(U_ed(0.0))
  du_bd = get_trial_fe_basis(U_bd(0.0))

  b(0.0,du_ed,du_bd,dv_ed,dv_bd,k_ed,k_bd,dΓ_ed_bd,n_Γ_ed_bd)

  dv_ed = get_fe_basis(V_ed)
  dv_bd = get_fe_basis(V_bd)

  du_ed = get_trial_fe_basis(U_ed(0.0))
  du_bd = get_trial_fe_basis(U_bd(0.0))

  c(0.0,dv_ed,dv_bd,k_ed,k_bd,dΓ_ed_bd,n_Γ_ed_bd)

* Remember conditions
* Implement smoothed version
* Use Authomatic Differentiation?

* beta proportional maximum diffusion and maximum of gradient (1/h)

* The directional derivative is computed with the Taylor expansion!!!