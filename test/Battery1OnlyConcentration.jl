module Battery1OnlyConcentration

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

  n = 40
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
 
  ## Manufactured solution

  u(x,t::Real) = ( x[1]^2 + x[2]^2 + x[3]^2 ) + 0.1 * t
  u(t::Real)   = x -> u(x,t)

  ## Conductivities

  k_ed(t::Real) = 1.0
  k_el(t::Real) = 1.0

  ## Jumps, weights and averages

  j(u⁺,u⁻) = u⁺ - u⁻
  w(k¹,k²) = t -> k²(t) / ( k¹(t) + k²(t) )
  ħ(k⁺,k⁻) = t -> ( 2.0 * k⁺(t) * k⁻(t) ) / ( k⁺(t) + k⁻(t) )

  μ(u⁺,u⁻,k⁺,k⁻) = t -> w(k⁺,k⁻)(t) * u⁺ + w(k⁻,k⁺)(t) * u⁻

  ## Source and transmission terms

  f(k,x,t::Real) = ∂t(u)(t)(x) - k(t) * Δ(u(t))(x)
  f(k,t::Real) = x -> f(k,x,t)

  j_Γ(t::Real) = 0.0
  g_Γ = (u⁺,u⁻) -> √(u⁻*u⁺*(1-u⁺))

  m(u,v,dΩ) = ∫( ∂t(u)*v )dΩ

  a(t,u,v,k,dΩ) = ∫( k(t)*(∇(u)⋅∇(v)) )dΩ

  l(t,v,k,dΩ) = ∫( v*f(k,t) )dΩ

  c(t,u⁺,u⁻,v⁺,v⁻,dΓ) = ∫( g_Γ∘(u⁺,u⁻)*j(v⁺,v⁻) )dΓ

  res(t,u,v,k,dΩ) = m(u,v,dΩ) + a(t,u,v,k,dΩ) - l(t,v,k,dΩ)

  jac_t(dut,v,dΩ) = ∫( dut*v )dΩ

  RES(t,(u_ed,u_el),(v_ed,v_el)) = 
    res(t,u_ed,v_ed,k_ed,dΩ_ed) + 
    res(t,u_el,v_el,k_el,dΩ_el) -
      c(t,u_ed,u_el,v_ed,v_el,dΓ_ed_el)
  JAC(t,(u_ed,u_el),(du_ed,du_el),(v_ed,v_el)) = 
    a(t,du_ed,v_ed,k_ed,dΩ_ed) +
    a(t,du_el,v_el,k_el,dΩ_el) # + ADD c TERM
  JAC_t(t,(u_ed,u_el),(du_ed,du_el),(v_ed,v_el)) = 
    jac_t(du_ed,v_ed,dΩ_ed) +
    jac_t(du_el,v_el,dΩ_el)

  # Transient FE Operator and solver

  # op = TransientFEOperator(RES,X,Y) # Jacobian computed with AD (slower)
  op = TransientFEOperator(RES,JAC,JAC_t,X,Y)

  using LineSearches: BackTracking
  nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())

  Δt = 0.5
  θ = 0.5
  ode_solver = ThetaMethod(nls,Δt,θ)

  u₀ = interpolate_everywhere([u(0.0),u(0.0)],X(0.0))
  t₀ = 0.0
  T = 1.0

  uₕₜ = solve(ode_solver,op,u₀,t₀,T)

  # Solution, errors and postprocessing

  l2(u,dΩ) = ∑( ∫( u*u )dΩ )
  h1(u,dΩ) = ∑( ∫( u*u + ∇(u)⋅∇(u) )dΩ )

  createpvd("TransientPoissonAgFEM") do pvd
    el2 = 0.0; eh1 = 0.0; ul2 = 0.0; uh1 = 0.0
    pvd[t₀] = createvtk(Ω_P_ed,"res_ed_0",cellfields=["uₕₜ"=>u₀[1]])
    for (i,((_u_ed,_u_el),t)) in enumerate(uₕₜ)
      pvd[t] = createvtk(Ω_P_ed,"res_ed_$i",cellfields=["uₕₜ"=>_u_ed])
      el2 = el2 + l2(u(t)-_u_ed,dΩ_ed) + l2(u(t)-_u_el,dΩ_el)
      eh1 = eh1 + h1(u(t)-_u_ed,dΩ_ed) + h1(u(t)-_u_el,dΩ_el)
      ul2 = ul2 + l2(_u_ed,dΩ_ed) + l2(_u_el,dΩ_el)
      uh1 = uh1 + h1(_u_ed,dΩ_ed) + h1(_u_el,dΩ_el)
    end
    el2 = √(Δt*el2)
    eh1 = √(Δt*eh1) # (!) Not scaled by diffusion
    @test el2/ul2 < 1.e-8
    @test eh1/uh1 < 1.e-7
  end
  
end # module