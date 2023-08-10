using Gridap
using Gridap.ReferenceFEs
using GridapEmbedded
using GridapPETSc
using GridapPETSc: PETSC
using Gridap.ODEs.ODETools
using Gridap.ODEs.ODETools: Nonlinear, Constant
using Gridap.ODEs.TransientFETools: TransientFEOperatorFromWeakForm
using SparseMatricesCSR
using Test

using LineSearches: BackTracking

function main(n)

  # Embedded geometry

  p = Point(0.3,0.2)
  electrode = plane(x0=p,v=VectorValue(1.0,0.0))
  electrolyte = ! electrode # setdiff(sph_el,sph_ed)

  # Background model

  domain = (0.0,0.45,0.0,0.45)
  partition = (n,n)

  bgmodel = CartesianDiscreteModel(domain,partition)
  h = (domain[2]-domain[1])/n

  Ω_bg = Triangulation(bgmodel)

  # Active and physical triangulations

  cutgeo = cut(bgmodel,electrolyte)

  Ω_A_ed = Triangulation(cutgeo,ACTIVE,electrode  )
  Ω_A_el = Triangulation(cutgeo,ACTIVE,electrolyte)

  Ω_P_ed = Triangulation(cutgeo,PHYSICAL,electrode  )
  Ω_P_el = Triangulation(cutgeo,PHYSICAL,electrolyte)

  Γ_ed_el = EmbeddedBoundary(cutgeo,electrode,electrolyte)
  n_Γ_ed_el = get_normal_vector(Γ_ed_el) # Exterior to electrode

  # writevtk(Ω_P_ed,"omega_ed")
  # writevtk(Ω_P_el,"omega_el")

  # Lebesgue measures

  dim = 2
  order = 1

  # degree = 2*order*dim
  # dΩ_ed = Measure(Ω_P_ed,degree)
  # dΩ_el = Measure(Ω_P_el,degree)

  degree = 3*order+1
  q_ed = Quadrature(momentfitted,cutgeo,electrode,degree)
  q_el = Quadrature(momentfitted,cutgeo,electrolyte,degree)
  dΩ_ed = Measure(Ω_A_ed,q_ed)
  dΩ_el = Measure(Ω_A_el,q_el)

  dΓ_ed_el = Measure(Γ_ed_el,degree)

  # Finite element spaces

  reffe = ReferenceFE(lagrangian,Float64,order)
  strategy = AggregateAllCutCells()

  Vstd_ed = TestFESpace(Ω_A_ed,reffe)
  V_ed    = AgFEMSpace(Vstd_ed,aggregate(strategy,cutgeo,electrode))

  Vstd_el = TestFESpace(Ω_A_el,reffe,dirichlet_tags=[2,4,8])
  V_el    = AgFEMSpace(Vstd_el,aggregate(strategy,cutgeo,electrolyte))

  U_ed = TransientTrialFESpace(V_ed)
  u_ext(x,t::Real) = 1.0
  u_ext(t) = x -> u_ext(x,t)
  U_el = TransientTrialFESpace(V_el,u_ext)

  Y = MultiFieldFESpace([V_ed,V_el])
  X = TransientMultiFieldFESpace([U_ed,U_el])  

  # Weak form

  ### Rmk. Material params. can be spatially-dependent, if needed.

  ## Conductivities

  ### Electrode
  #### [REF] Eq.(10) doi.org/10.1016/j.electacta.2022.140700
  ##### Thermal-electrochemical parameters of a high energy...
  #### [REF] doi.org/10.1149/1945-7111/ab9050
  ##### Development of Experimental Techniques for Parameterization...
  #### Parameters
  α = (0.0,-0.9231,-0.4066,-0.9930,0.0)
  β = (-13.96,0.3216,0.4532,0.8098,0.0)
  γ = (0.0,2.534e-3,3.926e-3,9.924e-2,1.0)
  #### Relation
  log_10_k_ed(x) = α[1] * x + β[1] +
                   α[2] * exp(-(x-β[2])^2/γ[2]) +
                   α[3] * exp(-(x-β[3])^2/γ[3]) +
                   α[4] * exp(-(x-β[4])^2/γ[4]) + 
                   α[5] * exp(-(x-β[5])^2/γ[5])
  k_ed = x -> exp10(log_10_k_ed(x))/(25e-12) # 25e-12 = (5e-6)^2

  dk_ed = x -> k_ed(x) * log(10) * ( α[1] +
    2 * ( α[2]*exp(-(x-β[2])^2/γ[2])*(β[2]-x)/γ[2] + 
          α[3]*exp(-(x-β[3])^2/γ[3])*(β[3]-x)/γ[3] +
          α[4]*exp(-(x-β[4])^2/γ[4])*(β[4]-x)/γ[4] +
          α[5]*exp(-(x-β[5])^2/γ[5])*(β[5]-x)/γ[5] ) )

  ### Electrolyte
  #### [REF] Eq.(18) doi.org/10.1149/2.0571912jes
  #### Parameters
  p = (1.01e3,1.01,-1.56e3,-4.87e2)
  T = 300
  #### Relation
  k_el = x -> (p[1]*exp(p[2]*x)*exp(p[3]/T)*exp(p[4]*x/T)*1.0e-10)/(25e-12) # 25e-12 = (5e-6)^2

  dk_el = x -> k_el(x) * ( p[2] + p[4]/T )

  ## Source and transmission terms

  f(k,x,t::Real) = 0.0
  f(k,t::Real) = x -> f(k,x,t)

  σ = 1.0e-06
  ## Eq. (32) Bonilla-Badia max{0,f}
  s(f) = (f+√(f*f+σ))/2.0
  ds(f) = (1+f/√(f*f+σ))/2.0

  g_Γ = (u⁺,u⁻) -> √(s(u⁻)*s(u⁺)*s(1-u⁺))
  dg⁺_Γ = (u⁺,u⁻) -> (s(u⁻)*(-s(u⁺)*ds(1-u⁺)+s(1-u⁺)*ds(u⁺)))/(2.0*√(s(u⁻)*s(1-u⁺)s(u⁺)))
  dg⁻_Γ = (u⁺,u⁻) -> (√(s(u⁻)*s(1-u⁺)*s(u⁺))*ds(u⁻))/(2*s(u⁻))

  j(u⁺,u⁻) = u⁺ - u⁻

  m(u,v,dΩ) = ∫( ∂t(u)*v )dΩ

  a(u,v,k,dΩ) = ∫( (k∘u)*(∇(u)⋅∇(v)) )dΩ

  da(t,u,du,v,k,dk,dΩ) = 
    ∫( (dk∘u)*du*(∇(u)⋅∇(v)) )dΩ + ∫( (k∘u)*(∇(du)⋅∇(v)) )dΩ

  l(t,v,k,dΩ) = ∫( v*f(k,t) )dΩ
  l₀(v,k,dΩ) = ∫( v*f(k,0.0) )dΩ

  c(t,u⁺,u⁻,v⁺,v⁻,dΓ) = ∫( g_Γ∘(u⁺,u⁻)*j(v⁺,v⁻) )dΓ

  dc(t,u⁺,u⁻,du⁺,du⁻,v⁺,v⁻,dΓ) = ∫( (dg⁺_Γ∘(u⁺,u⁻)*du⁺+dg⁻_Γ∘(u⁺,u⁻)*du⁻)*j(v⁺,v⁻) )dΓ

  # u1, u2 = interpolate_everywhere([u(0.0),u(0.0)],X(0.0))
  # du1, du2 = get_trial_fe_basis(X(0.0))
  # v1, v2 = get_fe_basis(Y)

  res(t,u,v,k,dΩ) = m(u,v,dΩ) + a(u,v,k,dΩ) - l₀(v,k,dΩ)

  jac_t(dut,v,dΩ) = ∫( dut*v )dΩ

  RES(t,(u_ed,u_el),(v_ed,v_el)) = 
    res(t,u_ed,v_ed,k_ed,dΩ_ed) + 
    res(t,u_el,v_el,k_el,dΩ_el) -
      c(t,u_ed,u_el,v_ed,v_el,dΓ_ed_el)
  JAC(t,(u_ed,u_el),(du_ed,du_el),(v_ed,v_el)) = 
    da(t,u_ed,du_ed,v_ed,k_ed,dk_ed,dΩ_ed) +
    da(t,u_el,du_el,v_el,k_el,dk_el,dΩ_el) +
    dc(t,u_ed,u_el,du_ed,du_el,v_ed,v_el,dΓ_ed_el)
  JAC_t(t,(u_ed,u_el),(du_ed,du_el),(v_ed,v_el)) = 
    jac_t(du_ed,v_ed,dΩ_ed) + jac_t(du_el,v_el,dΩ_el)

  # Transient FE Operator and solver
  assem = SparseMatrixAssembler(SparseMatrixCSR{0,PetscScalar,PetscInt},
                                Vector{PetscScalar},
                                evaluate(X,nothing),Y)
  op = TransientFEOperatorFromWeakForm{Nonlinear}(RES,(JAC,JAC_t),assem,(X,∂t(X)),Y,1)
  nls = PETScNonlinearSolver()

  # op = TransientFEOperator(RES,JAC,JAC_t,X,Y)
  # nls = NLSolver(show_trace=true, method=:newton, ftol = 1e-8, iterations=10, linesearch=BackTracking())
  # nls = NLSolver(show_trace=true, method=:anderson, m=0, iterations=30)

  Δt = 0.001/(4^(log2(n/20)))
  θ = 0.5
  ode_solver = ThetaMethod(nls,Δt,θ)

  ## Initial conditions
  u_ed = 0.5
  u_el = 1.0

  uᵢ = interpolate_everywhere([u_ed,u_el],X(0.0))
  t₀ = 0.0
  tₑ = 0.01

  # Solution, errors and postprocessing

  l2(u,dΩ) = ∑( ∫( u*u )dΩ )
  h1(u,k,dΩ) = ∑( ∫( (k∘u)*(∇(u)⋅∇(u)) )dΩ )

  @time createpvd("TransientPoissonAgFEM") do pvd
    ul2 = 0.0; uh1 = 0.0
    pvd[t₀] = createvtk(Ω_P_ed,"res_ed_0")
    writevtk(Ω_P_ed,"res_ed_0",cellfields=["uₕₜ"=>uᵢ[1]])
    writevtk(Ω_P_el,"res_el_0",cellfields=["uₕₜ"=>uᵢ[2]])
    i = 0
    for ti in t₀:Δt:(tₑ-Δt)
      @show ti,ti+Δt
      i = i+1
      uₕₜ = solve(ode_solver,op,uᵢ,ti,ti+Δt)
      for (_u,t) in uₕₜ
        _u_ed,_u_el = _u
        pvd[t] = createvtk(Ω_P_ed,"res_ed_$i")
        writevtk(Ω_P_ed,"res_ed_$i",cellfields=["uₕₜ"=>_u_ed])
        writevtk(Ω_P_el,"res_el_$i",cellfields=["uₕₜ"=>_u_el])
        ul2 = ul2 + l2(_u_ed,dΩ_ed) + l2(_u_el,dΩ_el)
        uh1 = uh1 + h1(_u_ed,k_ed,dΩ_ed) + h1(_u_el,k_el,dΩ_el)
        uᵢ = _u
      end
    end
    ul2 = √(Δt*ul2)
    uh1 = √(Δt*uh1)
    @show ul2
    @show uh1
  end

end

options = "-snes_type newtonls
           -snes_linesearch_type bt 
           -snes_linesearch_damping 0.8
           -snes_linesearch_monitor 
           -snes_rtol 1.0e-08 
           -snes_atol 0.0 
           -pc_type cholesky
           -pc_factor_mat_solver_type mumps
           -ksp_type none
           -ksp_monitor
           -snes_converged_reason 
           -ksp_converged_reason 
           -ksp_error_if_not_converged true"

# options = "-snes_type newtonls
#            -snes_linesearch_type bt 
#            -snes_linesearch_damping 0.8
#            -snes_linesearch_monitor 
#            -snes_rtol 1.0e-08 
#            -snes_atol 0.0 
#            -pc_type cholesky
#            -pc_factor_mat_solver_type mumps
#            -ksp_type none
#            -ksp_monitor
#            -snes_converged_reason 
#            -ksp_converged_reason 
#            -ksp_error_if_not_converged true"

GridapPETSc.with(args=split(options)) do
  main(20)
  main(40)
  # main(80)
  # main(160)
  # main(320)
  # main(640)
end

# ul2 = 0.004499274324688881
# uh1 = 0.0002470357418340545

# ul2 = 0.004499394879430065
# uh1 = 0.00023101171000226763

# ul2 = 0.004499464598786009
# uh1 = 0.0002211672304197376

# ul2 = 0.004499504248244329
# uh1 = 0.0002305132720755624

###############

# ul2 = 0.004499274324688881
# uh1 = 0.0002470357418340545

# ul2 = 0.004499452729195149
# uh1 = 0.0002138012026235924

# ul2 = 0.004499507248071766
# uh1 = 0.0002079995247887595

# ul2 = 0.004499528433702517
# uh1 = 0.0002217589900201261