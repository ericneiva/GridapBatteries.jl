module Battery2OnlyConcentration

  using Gridap
  using GridapEmbedded
  using Test

  # Embedded geometry

  R = 0.38

  p1 = Point(-0.5,-0.5,-0.5)
  p2 = Point( 0.5,-0.5,-0.5)
  p3 = Point(-0.5, 0.5,-0.5)
  p4 = Point( 0.5, 0.5,-0.5)
  p5 = Point(-0.5,-0.5, 0.5)
  p6 = Point( 0.5,-0.5, 0.5)
  p7 = Point(-0.5, 0.5, 0.5)
  p8 = Point( 0.5, 0.5, 0.5)

  sph1 = sphere(R,x0=p1)
  sph2 = sphere(R,x0=p2)
  sph3 = sphere(R,x0=p3)
  sph4 = sphere(R,x0=p4)
  sph5 = sphere(R,x0=p5)
  sph6 = sphere(R,x0=p6)
  sph7 = sphere(R,x0=p7)
  sph8 = sphere(R,x0=p8)

  r = 0.24

  v1    = VectorValue(0.0,0.0,1.0)
  cyl1  = cylinder(r,x0=p1,v=v1)
  cyl2  = cylinder(r,x0=p2,v=v1)
  cyl3  = cylinder(r,x0=p3,v=v1)
  cyl4  = cylinder(r,x0=p4,v=v1)

  v2    = VectorValue(1.0,0.0,0.0)
  cyl5  = cylinder(r,x0=p1,v=v2)
  cyl6  = cylinder(r,x0=p3,v=v2)
  cyl7  = cylinder(r,x0=p5,v=v2)
  cyl8  = cylinder(r,x0=p7,v=v2)

  v3    = VectorValue(0.0,1.0,0.0)
  cyl9  = cylinder(r,x0=p1,v=v3)
  cyl10 = cylinder(r,x0=p2,v=v3)
  cyl11 = cylinder(r,x0=p5,v=v3)
  cyl12 = cylinder(r,x0=p6,v=v3)

  _sphs = union(union(union(union(union(union(union(sph1,sph2),sph3),sph4),sph5),sph6),sph7),sph8)
  _cyls = union(union(union(union(union(union(union(union(union(union(union(cyl1,cyl2),cyl3),cyl4),cyl5),cyl6),cyl7),cyl8),cyl9),cyl10),cyl11),cyl12)

  # Warning: Touching can have unexpected consequences and likely break the code
  electrode   = _sphs
  binder      = setdiff(_cyls,electrode)
  electrolyte = ! union(electrode,_cyls)

  # electrode = union(_sphs,_cyls)
  # electrolyte = ! electrode

  # Background model

  n = 20
  domain = (-1.0,1.0,-1.0,1.0,-1.0,1.0)
  partition = (n,n,n)

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

  # Lebesgue measures

  dim = 3
  order = 1
  degree = 2*order*dim

  dΩ_ed = Measure(Ω_P_ed,degree)
  dΩ_el = Measure(Ω_P_el,degree)

  dΓ_ed_el = Measure(Γ_ed_el,degree)

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
  k_ed = x -> exp10(log_10_k_ed(x))/(25e-12) 
  # Q1) 25e-12 Because vals around 10^(-12)?
  # Q2) cm^2 to m^2?

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
  k_el = x -> (p[1]*exp(p[2]*x)*exp(p[3]/T)*exp(p[4]*x/T)*1.0e-10)/(25e-12)
  # Q1) Why 25e-12?
  # Q2) cm^2 to m^2?
  
  dk_el = x -> k_el(x) * ( p[2] + p[4]/T )

  ## Source and transmission terms

  ### Rmk. Homogeneous Neumann BCs

  f(k,x,t::Real) = 0.0
  f(k,t::Real) = x -> f(k,x,t)

  σ = 1.0e-06
  s(f) = (f+√(f*f+σ))/2.0
  ds(f) = (1+f/√(f*f+σ))/2.0

  g_Γ = (u⁺,u⁻) -> √(s(u⁻)*s(u⁺)*s(1-u⁺))
  dg⁺_Γ = (u⁺,u⁻) -> (s(u⁻)*(-s(u⁺)*ds(1-u⁺)+s(1-u⁺)*ds(u⁺)))/(2.0*√(s(u⁻)*s(1-u⁺)s(u⁺)))
  dg⁻_Γ = (u⁺,u⁻) -> (√(s(u⁻)*s(1-u⁺)*s(u⁺))*ds(u⁻))/(2*s(u⁻))

  j(u⁺,u⁻) = u⁺ - u⁻

  m(u,v,dΩ) = ∫( ∂t(u)*v )dΩ

  a(t,u,v,k,dΩ) = ∫( (k∘u)*(∇(u)⋅∇(v)) )dΩ

  da(t,u,du,v,k,dk,dΩ) = 
    ∫( (dk∘u)*du*(∇(u)⋅∇(v)) )dΩ + ∫( (k∘u)*(∇(du)⋅∇(v)) )dΩ

  l(t,v,k,dΩ) = ∫( v*f(k,t) )dΩ

  c(t,u⁺,u⁻,v⁺,v⁻,dΓ) = ∫( g_Γ∘(u⁺,u⁻)*j(v⁺,v⁻) )dΓ

  dc(t,u⁺,u⁻,du⁺,du⁻,v⁺,v⁻,dΓ) = ∫( (dg⁺_Γ∘(u⁺,u⁻)*du⁺+dg⁻_Γ∘(u⁺,u⁻)*du⁻)*j(v⁺,v⁻) )dΓ

  # u1, u2 = interpolate_everywhere([u(0.0),u(0.0)],X(0.0))
  # du1, du2 = get_trial_fe_basis(X(0.0))
  # v1, v2 = get_fe_basis(Y)

  res(t,u,v,k,dΩ) = m(u,v,dΩ) + a(t,u,v,k,dΩ) - l(t,v,k,dΩ)

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

  op = TransientFEOperator(RES,JAC,JAC_t,X,Y)

  using LineSearches: BackTracking
  nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), iterations=15)
  # nls = NLSolver(show_trace=true, method=:anderson, m=0, iterations=30)

  Δt = 0.1
  θ = 1.0
  ode_solver = ThetaMethod(nls,Δt,θ)

  ## Initial conditions
  u_ed = 0.5
  u_el = 1.0

  u₀ = interpolate_everywhere([u_ed,u_el],X(0.0))
  t₀ = 0.0
  T = 1.0

  uₕₜ = solve(ode_solver,op,u₀,t₀,T)

  # Solution, errors and postprocessing

  l2(u,dΩ) = ∑( ∫( u*u )dΩ )
  h1(u,dΩ) = ∑( ∫( u*u + ∇(u)⋅∇(u) )dΩ )

  # using Gridap.ODEs.ODETools
  # cache = nothing
  # uᵢ = interpolate_everywhere([u(0.0),u(0.0)],X(0.0))
  # uᵢ, tᵢ, cache = solve_step!(uᵢ,ode_solver,op,u₀,t₀)

  createpvd("TransientPoissonAgFEM") do pvd
    ul2 = 0.0; uh1 = 0.0
    # pvd[t₀] = createvtk(Ω_P_ed,"res_ed_0",cellfields=["uₕₜ"=>u₀[1]])
    for (i,((_u_ed,_u_el),t)) in enumerate(uₕₜ)
      # pvd[t] = createvtk(Ω_P_ed,"res_ed_$i",cellfields=["uₕₜ"=>_u_ed])
      ul2 = ul2 + l2(_u_ed,dΩ_ed) + l2(_u_el,dΩ_el)
      uh1 = uh1 + h1(_u_ed,dΩ_ed) + h1(_u_el,dΩ_el)
    end
    ul2 = √(Δt*ul2)
    uh1 = √(Δt*uh1) # (!) Not scaled by diffusion
    @show ul2
    @show uh1
  end
  
end # module