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

  eps = 1.0e-4
  r_ed = 0.30+eps
  r_el = 3.00+eps
  p = Point(0.0,0.0)

  sph_ed = disk(r_ed,x0=p)
  sph_el = disk(r_el,x0=p)

  electrode   = sph_ed
  electrolyte = setdiff(sph_el,sph_ed)
  all         = sph_el

  # Background model

  domain = (-3.1,3.1,-3.1,3.1)
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

  Γ_out = EmbeddedBoundary(cutgeo,all)
  n_Γ_out = get_normal_vector(Γ_out) # Exterior to electrolyte

  # writevtk(Ω_P_ed,"omega_ed")
  # writevtk(Ω_P_el,"omega_el")

  # Lebesgue measures

  dim = 2
  order = 1

  # degree = 2*order*dim
  # dΩ_ed = Measure(Ω_P_ed,degree)
  # dΩ_el = Measure(Ω_P_el,degree)

  degree = 2*order # Subintegration # 3*order+1
  q_ed = Quadrature(momentfitted,cutgeo,electrode,degree)
  q_el = Quadrature(momentfitted,cutgeo,electrolyte,degree)

  dΩ_ed = Measure(Ω_A_ed,q_ed)
  dΩ_el = Measure(Ω_A_el,q_el)

  # Quadratures for lumped mass matrix
  degree_l = 1
  q_ed_l = Quadrature(momentfitted,cutgeo,electrode,degree_l)
  q_el_l = Quadrature(momentfitted,cutgeo,electrolyte,degree_l)

  dΩ_ed_l = Measure(Ω_A_ed,q_ed_l)
  dΩ_el_l = Measure(Ω_A_el,q_el_l)

  dΓ_ed_el = Measure(Γ_ed_el,degree)
  dΓ_out = Measure(Γ_out,degree)

  # Finite element spaces

  reffe = ReferenceFE(lagrangian,Float64,order)
  strategy = AggregateAllCutCells()

  Vstd_ed = TestFESpace(Ω_A_ed,reffe)
  V_ed    = AgFEMSpace(Vstd_ed,aggregate(strategy,cutgeo,electrode))

  Vstd_el = TestFESpace(Ω_A_el,reffe)
  V_el    = AgFEMSpace(Vstd_el,aggregate(strategy,cutgeo,electrolyte))

  U_ed = TransientTrialFESpace(V_ed)
  U_el = TransientTrialFESpace(V_el)

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
  dg⁻_Γ = (u⁺,u⁻) -> (√(s(u⁻)*s(1-u⁺)*s(u⁺))*ds(u⁻))/(2.0*s(u⁻))

  j(u⁺,u⁻) = u⁺ - u⁻

  m(u,v,dΩ) = ∫( ∂t(u)*v )dΩ

  a(u,v,k,dΩ) = ∫( (k∘u)*(∇(u)⋅∇(v)) )dΩ

  da(t,u,du,v,k,dk,dΩ) = 
    ∫( (dk∘u)*du*(∇(u)⋅∇(v)) )dΩ + ∫( (k∘u)*(∇(du)⋅∇(v)) )dΩ

  l(t,v,k,dΩ) = ∫( v*f(k,t) )dΩ
  l₀(v,k,dΩ) = ∫( v*f(k,0.0) )dΩ

  c(t,u⁺,u⁻,v⁺,v⁻,dΓ) = ∫( g_Γ∘(u⁺,u⁻)*j(v⁺,v⁻) )dΓ

  dc(t,u⁺,u⁻,du⁺,du⁻,v⁺,v⁻,dΓ) = ∫( (dg⁺_Γ∘(u⁺,u⁻)*du⁺+dg⁻_Γ∘(u⁺,u⁻)*du⁻)*j(v⁺,v⁻) )dΓ

  γᵈ = 25.0 # ≈ maximum of k_el
  u_out = 1.0

  aᵈ(u,v,k,n,dΓ) = 
    ∫( (γᵈ/h)*u*v  - (k∘u)*(n⋅∇(u))*v - (k∘v)*(n⋅∇(v))*u )dΓ
  bᵈ(uᵈ,v,k,n,dΓ) = ∫( (γᵈ/h)*uᵈ*v - (k∘v)*(n⋅∇(v))*uᵈ )dΓ

  daᵈ(u,du,v,k,dk,n,dΓ) = ∫( (γᵈ/h)*du*v - 
    (dk∘u)*du*(n⋅∇(u))*v - (k∘u)*(n⋅∇(du))*v - (k∘v)*(n⋅∇(v))*du )dΓ

  # u1, u2 = interpolate_everywhere([u(0.0),u(0.0)],X(0.0))
  # du1, du2 = get_trial_fe_basis(X(0.0))
  # v1, v2 = get_fe_basis(Y)

  res(t,u,v,k,dΩ,dΩ_l) = m(u,v,dΩ_l) + a(u,v,k,dΩ) - l₀(v,k,dΩ)
  rhs(t,u,v,k,dΩ) = l₀(v,k,dΩ) - a(u,v,k,dΩ)
  jac_t(dut,v,dΩ) = ∫( dut*v )dΩ

  RES(t,(u_ed,u_el),(v_ed,v_el)) = 
    res(t,u_ed,v_ed,k_ed,dΩ_ed,dΩ_ed_l) +
    res(t,u_el,v_el,k_el,dΩ_el,dΩ_el_l) -
      c(t,u_ed,u_el,v_ed,v_el,dΓ_ed_el) +
      aᵈ(u_el,v_el,k_el,n_Γ_out,dΓ_out) -
      bᵈ(u_out,v_el,k_el,n_Γ_out,dΓ_out)
  RHS(t,(u_ed,u_el),(v_ed,v_el)) = 
    rhs(t,u_ed,v_ed,k_ed,dΩ_ed) + 
    rhs(t,u_el,v_el,k_el,dΩ_el) +
      c(t,u_ed,u_el,v_ed,v_el,dΓ_ed_el) -
      aᵈ(u_el,v_el,k_el,n_Γ_out,dΓ_out) +
      bᵈ(u_out,v_el,k_el,n_Γ_out,dΓ_out)
  JAC(t,(u_ed,u_el),(du_ed,du_el),(v_ed,v_el)) = 
    da(t,u_ed,du_ed,v_ed,k_ed,dk_ed,dΩ_ed) +
    da(t,u_el,du_el,v_el,k_el,dk_el,dΩ_el) -
    dc(t,u_ed,u_el,du_ed,du_el,v_ed,v_el,dΓ_ed_el) +
    daᵈ(u_el,du_el,v_el,k_el,dk_el,n_Γ_out,dΓ_out) 
  JAC_t(t,(u_ed,u_el),(du_ed,du_el),(v_ed,v_el)) = 
    jac_t(du_ed,v_ed,dΩ_ed_l) + jac_t(du_el,v_el,dΩ_el_l) # Lumped

  # Transient FE Operator and solver
  assem = SparseMatrixAssembler(SparseMatrixCSR{0,PetscScalar,PetscInt},
                                Vector{PetscScalar},
                                evaluate(X,nothing),Y)
  op = TransientFEOperatorFromWeakForm{Nonlinear}(RES,RHS,(JAC,JAC_t),assem,(X,∂t(X)),Y,1)
  nls = PETScNonlinearSolver()

  # op = TransientFEOperator(RES,JAC,JAC_t,X,Y)
  # nls = NLSolver(show_trace=true, method=:newton, ftol = 1e-8, iterations=20, linesearch=BackTracking())
  # nls = NLSolver(show_trace=true, method=:anderson, m=0, iterations=30)

  Δt = 0.01 # *(20/n)
  θ = 1.0
  ode_solver = ThetaMethod(nls,Δt,θ)

  ## Initial conditions
  u_ed = 0.5
  u_el = 1.0

  uᵢ = interpolate_everywhere([u_ed,u_el],X(0.0))
  t₀ = 0.0
  tₑ = 0.5

  # Solution, errors and postprocessing

  l2(u,dΩ) = ∑( ∫( u*u )dΩ )
  h1(u,k,dΩ) = ∑( ∫( (k∘u)*(∇(u)⋅∇(u)) )dΩ )

  @time createpvd("results/TransientPoissonAgFEM") do pvd
    ul2 = 0.0; uh1 = 0.0
    pvd[t₀] = createvtk(Ω_P_ed,"results/res_ed_0")
    writevtk(Ω_P_ed,"results/res_ed_0",cellfields=["uₕₜ"=>uᵢ[1]])
    writevtk(Ω_P_el,"results/res_el_0",cellfields=["uₕₜ"=>uᵢ[2]])
    i = 0
    for ti in t₀:Δt:(tₑ-Δt)
      @show ti,ti+Δt
      i = i+1
      uₕₜ = solve(ode_solver,op,uᵢ,ti,ti+Δt)
      for (_u,t) in uₕₜ
        _u_ed,_u_el = _u
        pvd[t] = createvtk(Ω_P_ed,"results/res_ed_$i")
        writevtk(Ω_P_ed,"results/res_ed_$i",cellfields=["uₕₜ"=>_u_ed])
        writevtk(Ω_P_el,"results/res_el_$i",cellfields=["uₕₜ"=>_u_el])
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

options = "-snes_type nrichardson
           -snes_linesearch_type basic
           -snes_linesearch_damping 1.0
           -npc_snes_type newtonls
           -npc_snes_rtol 1.0e-08
           -npc_snes_atol 0.0
           -snes_rtol 1.0e-08
           -snes_atol 0.0
           -pc_type jacobi
           -ksp_type gmres
           -ksp_monitor
           -snes_converged_reason 
           -ksp_converged_reason 
           -ksp_error_if_not_converged true"

GridapPETSc.with(args=split(options)) do
  # main(20)
  # main(40)
  main(80)
  # main(160)
  # main(320)
  # main(640)
end
