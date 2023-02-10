module TransientTrimaterialPoissonAgFEMTests

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
  Ω_A_bd = Triangulation(cutgeo,ACTIVE,binder     )
  Ω_A_el = Triangulation(cutgeo,ACTIVE,electrolyte)

  # writevtk(Ω_A_ed,"trian_O_A_electrode"  )
  # writevtk(Ω_A_bd,"trian_O_A_binder"     )
  # writevtk(Ω_A_el,"trian_O_A_electrolyte")

  Ω_P_ed = Triangulation(cutgeo,PHYSICAL,electrode  )
  Ω_P_bd = Triangulation(cutgeo,PHYSICAL,binder     )
  Ω_P_el = Triangulation(cutgeo,PHYSICAL,electrolyte)

  # writevtk(Ω_P_ed,"trian_O_P_electrode"  )
  # writevtk(Ω_P_bd,"trian_O_P_binder"     )
  # writevtk(Ω_P_el,"trian_O_P_electrolyte")

  Γ_ed_bd = EmbeddedBoundary(cutgeo,electrode,binder)
  Γ_ed_el = EmbeddedBoundary(cutgeo,electrode,electrolyte)
  Γ_bd_el = EmbeddedBoundary(cutgeo,binder,electrolyte)

  # writevtk(Γ_ed_bd,"trian_G_ed_bd")
  # writevtk(Γ_ed_el,"trian_G_ed_el")
  # writevtk(Γ_bd_el,"trian_G_bd_el")

  n_Γ_ed_bd = get_normal_vector(Γ_ed_bd) # Exterior to electrode
  n_Γ_ed_el = get_normal_vector(Γ_ed_el) # Exterior to electrode
  n_Γ_bd_el = get_normal_vector(Γ_bd_el) # Exterior to binder

  # Lebesgue measures

  dim = 3
  order = 1
  degree = 2*order*dim

  dΩ_ed = Measure(Ω_P_ed,degree)
  dΩ_bd = Measure(Ω_P_bd,degree)
  dΩ_el = Measure(Ω_P_el,degree)
  
  dΓ_ed_bd = Measure(Γ_ed_bd,degree)
  dΓ_ed_el = Measure(Γ_ed_el,degree)
  dΓ_bd_el = Measure(Γ_bd_el,degree)

  # Finite element spaces

  reffe = ReferenceFE(lagrangian,Float64,order)
  strategy = AggregateAllCutCells()

  Vstd_ed = TestFESpace(Ω_A_ed,reffe) # No Dirichlet boundary
  V_ed    = AgFEMSpace(Vstd_ed,aggregate(strategy,cutgeo,electrode))

  Vstd_bd = TestFESpace(Ω_A_bd,reffe,dirichlet_tags="boundary")
  V_bd    = AgFEMSpace(Vstd_bd,aggregate(strategy,cutgeo,binder))

  Vstd_el = TestFESpace(Ω_A_el,reffe,dirichlet_tags="boundary")
  V_el    = AgFEMSpace(Vstd_el,aggregate(strategy,cutgeo,electrolyte))

  υ(x,t::Real) = t * ( x[1] + x[2] + x[3] )
  υ(t::Real) = x -> υ(x,t)

  k_ed(t::Real) = 1.0
  k_bd(t::Real) = 1.5
  k_el(t::Real) = 2.0

  # Solution of the problem

  u_ed(x,t::Real) = k_ed(t) * υ(x,t)
  u_bd(x,t::Real) = k_bd(t) * υ(x,t)
  u_el(x,t::Real) = k_el(t) * υ(x,t)

  u_ed(t::Real) = x -> u_ed(x,t)
  u_bd(t::Real) = x -> u_bd(x,t)
  u_el(t::Real) = x -> u_el(x,t)

  U_ed = TransientTrialFESpace(V_ed) # No Dirichlet boundary
  U_bd = TransientTrialFESpace(V_bd,u_bd)
  U_el = TransientTrialFESpace(V_el,u_el)

  Y = MultiFieldFESpace([V_ed,V_bd,V_el])
  X = TransientMultiFieldFESpace([U_ed,U_bd,U_el])  

  # Weak form

  ## Jumps, weights and averages

  j(u⁺,u⁻) = u⁺ - u⁻
  υ(k¹,k²) = t -> k²(t) / ( k¹(t) + k²(t) )
  ħ(k⁺,k⁻) = t -> ( 2.0 * k⁺(t) * k⁻(t) ) / ( k⁺(t) + k⁻(t) )

  μ(u⁺,u⁻,k⁺,k⁻) = t -> υ(k⁺,k⁻)(t) * u⁺ + υ(k⁻,k⁺)(t) * u⁻

  ## Source and transmission terms

  f(k,x,t::Real) = k(t) * ( ∂t(υ)(t)(x) - Δ(υ(t))(x) )
  f(k,t::Real) = x -> f(k,x,t)

  j_Γ(k⁺,k⁻,t::Real) = x -> ( k⁺(t) - k⁻(t) ) * υ(x,t)
  g_Γ(k⁺,k⁻,t::Real) = x -> ( k⁺(t) - k⁻(t) ) * ∇(υ(t))(x)

  γd = 10.0 # Nitsche coefficient for Nitsche's method

  m(u,v,dΩ) = ∫( ∂t(u)*v )dΩ

  a(t,u,v,dΩ) = ∫( ∇(u)⋅∇(v) )dΩ

  b(t,u⁺,u⁻,v⁺,v⁻,k⁺,k⁻,dΓ,n_Γ) = 
    ∫( (γd*ħ(k⁺,k⁻)(t)/h)*j(u⁺,u⁻)*j(v⁺,v⁻)
      - j(v⁺,v⁻)*(n_Γ⋅μ(∇(u⁺),∇(u⁻),k⁺,k⁻)(t))
      - (n_Γ⋅μ(∇(v⁺),∇(v⁻),k⁺,k⁻)(t))*j(u⁺,u⁻) )dΓ

  l(t,v,k,dΩ) = ∫( v*f(k,t) )dΩ

  c(t,v⁺,v⁻,k⁺,k⁻,dΓ,n_Γ) = 
    ∫( (γd*ħ(k⁺,k⁻)(t)/h)*(j_Γ(k⁺,k⁻,t)*j(v⁺,v⁻))
     - (n_Γ⋅μ(∇(v⁺),∇(v⁻),k⁺,k⁻)(t))*j_Γ(k⁺,k⁻,t)
     + (g_Γ(k⁺,k⁻,t)⋅n_Γ)*(υ(k⁻,k⁺)(t)*v⁺ + υ(k⁺,k⁻)(t)*v⁻) )dΓ

  res(t,u,v,k,dΩ) = m(u,v,dΩ) + a(t,u,v,dΩ) - l(t,v,k,dΩ)

  jac_t(dut,v,dΩ) = ∫( dut*v )dΩ

  RES(t,(u_ed,u_bd,u_el),(v_ed,v_bd,v_el)) = 
    res(t,u_ed,v_ed,k_ed,dΩ_ed) + 
    res(t,u_bd,v_bd,k_bd,dΩ_bd) + 
    res(t,u_el,v_el,k_el,dΩ_el) +
      b(t,u_ed,u_bd,v_ed,v_bd,k_ed,k_bd,dΓ_ed_bd,n_Γ_ed_bd) +
      b(t,u_ed,u_el,v_ed,v_el,k_ed,k_el,dΓ_ed_el,n_Γ_ed_el) +
      b(t,u_bd,u_el,v_bd,v_el,k_bd,k_el,dΓ_bd_el,n_Γ_bd_el) -
                c(t,v_ed,v_bd,k_ed,k_bd,dΓ_ed_bd,n_Γ_ed_bd) -
                c(t,v_ed,v_el,k_ed,k_el,dΓ_ed_el,n_Γ_ed_el) -
                c(t,v_bd,v_el,k_bd,k_el,dΓ_bd_el,n_Γ_bd_el)
  JAC(t,(u_ed,u_bd,u_el),(du_ed,du_bd,du_el),(v_ed,v_bd,v_el)) = 
    a(t,du_ed,v_ed,dΩ_ed) +
    a(t,du_bd,v_bd,dΩ_bd) +
    a(t,du_el,v_el,dΩ_el) +
    b(t,du_ed,du_bd,v_ed,v_bd,k_ed,k_bd,dΓ_ed_bd,n_Γ_ed_bd) +
    b(t,du_ed,du_el,v_ed,v_el,k_ed,k_el,dΓ_ed_el,n_Γ_ed_el) +
    b(t,du_bd,du_el,v_bd,v_el,k_bd,k_el,dΓ_bd_el,n_Γ_bd_el)
  JAC_t(t,(u_ed,u_bd,u_el),(du_ed,du_bd,du_el),(v_ed,v_bd,v_el)) = 
    jac_t(du_ed,v_ed,dΩ_ed) +
    jac_t(du_bd,v_bd,dΩ_bd) +
    jac_t(du_el,v_el,dΩ_el)

  # Transient FE Operator and solver

  # op = TransientFEOperator(RES,X,Y) # Jacobian computed with AD (slower)
  op = TransientFEOperator(RES,JAC,JAC_t,X,Y)

  linear_solver = LUSolver()

  Δt = 0.5
  θ = 0.5
  ode_solver = ThetaMethod(linear_solver,Δt,θ)

  u₀ = interpolate_everywhere([u_ed(0.0),u_bd(0.0),u_el(0.0)],X(0.0))
  t₀ = 0.0
  T = 1.0

  uₕₜ = solve(ode_solver,op,u₀,t₀,T)

  # Solution, errors and postprocessing

  l2(u,dΩ) = ∑( ∫( u*u )dΩ )
  h1(u,dΩ) = ∑( ∫( u*u + ∇(u)⋅∇(u) )dΩ )

  createpvd("TransientPoissonAgFEM") do pvd
    el2 = 0.0; eh1 = 0.0; ul2 = 0.0; uh1 = 0.0
    pvd[t₀] = createvtk(Ω_P_ed,"res_ed_0",cellfields=["uₕₜ"=>u₀[1]])
    for (i,((_u_ed,_u_bd,_u_el),t)) in enumerate(uₕₜ)
      pvd[t] = createvtk(Ω_P_ed,"res_ed_$i",cellfields=["uₕₜ"=>_u_ed])
      el2 = el2 + l2(u_ed(t)-_u_ed,dΩ_ed) + l2(u_bd(t)-_u_bd,dΩ_bd) + l2(u_el(t)-_u_el,dΩ_el)
      eh1 = eh1 + h1(u_ed(t)-_u_ed,dΩ_ed) + h1(u_bd(t)-_u_bd,dΩ_bd) + h1(u_el(t)-_u_el,dΩ_el)
      ul2 = ul2 + l2(_u_ed,dΩ_ed) + l2(_u_bd,dΩ_bd) + l2(_u_el,dΩ_el)
      uh1 = uh1 + h1(_u_ed,dΩ_ed) + h1(_u_bd,dΩ_bd) + h1(_u_el,dΩ_el)
    end
    el2 = √(Δt*el2)
    eh1 = √(Δt*eh1) # (!) Not scaled by diffusion
    @test el2/ul2 < 1.e-8
    @test eh1/uh1 < 1.e-7
  end
  
end # module