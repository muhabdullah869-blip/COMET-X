import numpy as np
import rebound
from dynesty import NestedSampler

# =====================================================================
# 1. ORBIT PROPAGATION & COORDINATE TRANSFORMATION
# =====================================================================

def propagate_orbit_rebound(params, times, model_type="outgassing"):
    sim = rebound.Simulation()
    sim.units = ('days', 'AU', 'Msun')
    
    sim.add(m=1.0) # Sun
    sim.add(m=0.000954388, a=5.2044, e=0.0489, inc=0.0227) # Jupiter
    
    a, e, inc, Omega, omega, M = params[:6]
    sim.add(m=0.0, a=a, e=e, inc=inc, Omega=Omega, omega=omega, M=M)
    
    if model_type == "outgassing":
        A2_outg = params[6]
        def cometary_outgassing_force(reb_sim):
            p = reb_sim.contents.particles[-1]
            r = np.sqrt(p.x**2 + p.y**2 + p.z**2)
            if r == 0:
                return
            g_r = 0.11126 * (1.0 / r)**2.15
            p.ax += A2_outg * g_r * (p.vx / r)
            p.ay += A2_outg * g_r * (p.vy / r)
            p.az += A2_outg * g_r * (p.vz / r)
        sim.additional_forces = cometary_outgassing_force
        
    sim.integrator = "IAS15"
    
    modeled_positions = []
    if len(times) == 0:
        return np.array(modeled_positions)
        
    try:
        sim.integrate(times[0])
        for t in times:
            sim.integrate(t)
            p = sim.particles[-1]
            modeled_positions.append([p.x, p.y, p.z])
    except Exception:
        return None
        
    return np.array(modeled_positions)

def xyz_to_radec(xyz_array):
    radec = []
    for pos in xyz_array:
        x, y, z = pos
        r_xy = np.sqrt(x**2 + y**2)
        ra = np.arctan2(y, x)
        dec = np.arctan2(z, r_xy)
        radec.append([ra, dec])
    return np.array(radec)


# =====================================================================
# 2. BAYESIAN LIKELIHOOD & PRIOR FUNCTIONS
# =====================================================================

def prior_transform(cube, model_type="outgassing"):
    params = np.zeros_like(cube)
    params[0] = cube[0] * 0.4 + 2.0   # a [AU] (2.0 to 2.4)
    params[1] = cube[1] * 0.05 + 0.08 # e (0.08 to 0.13)
    params[2] = cube[2] * 0.02 + 0.04 # inc [rad]
    params[3] = cube[3] * 0.2 + 1.1   # Omega
    params[4] = cube[4] * 0.2 + 0.7   # omega
    params[5] = cube[5] * 0.2 - 0.1   # M
    
    if model_type == "outgassing":
        log_min, log_max = -13.0, -10.0
        params[6] = 10.0 ** (cube[6] * (log_max - log_min) + log_min)
        
    return params

def log_likelihood(params, times, observed_astrometry, covariance_matrices, model_type="outgassing"):
    modeled_xyz = propagate_orbit_rebound(params, times, model_type=model_type)
    if modeled_xyz is None:
        return -np.inf 
    
    modeled_radec = xyz_to_radec(modeled_xyz)
    log_l = 0.0
    for i, t in enumerate(times):
        residual = modeled_radec[i] - observed_astrometry[i]
        residual[0] = (residual[0] + np.pi) % (2 * np.pi) - np.pi
        
        try:
            inv_cov = np.linalg.inv(covariance_matrices[i])
            chi2 = residual.T @ inv_cov @ residual
            _, log_det = np.linalg.slogdet(2.0 * np.pi * covariance_matrices[i])
            log_l += -0.5 * (chi2 + log_det)
        except Exception:
            return -np.inf
            
    if np.isnan(log_l):
        return -np.inf
    return log_l


# =====================================================================
# 3. SYNTHETIC DATA GENERATOR
# =====================================================================

def generate_synthetic_data(true_A2, times, nominal_elements):
    sim = rebound.Simulation()
    sim.units = ('days', 'AU', 'Msun')
    sim.add(m=1.0)
    sim.add(m=0.000954388, a=5.2044, e=0.0489, inc=0.0227)
    
    a, e, inc, Omega, omega, M = nominal_elements
    sim.add(m=0.0, a=a, e=e, inc=inc, Omega=Omega, omega=omega, M=M)
    
    def injected_outgassing(reb_sim):
        p = reb_sim.contents.particles[-1]
        r = np.sqrt(p.x**2 + p.y**2 + p.z**2)
        if r == 0:
            return
        g_r = 0.11126 * (1.0 / r)**2.15
        p.ax += true_A2 * g_r * (p.vx / r)
        p.ay += true_A2 * g_r * (p.vy / r)
        p.az += true_A2 * g_r * (p.vz / r)
        
    sim.additional_forces = injected_outgassing
    sim.integrator = "IAS15"
    
    sim.integrate(times[0])
    astrometry = []
    covariances = []
    sigma_astrom = 0.5 / 206265.0 # Ground-based uncertainty (~0.5 arcsec)
    
    for t in times:
        sim.integrate(t)
        p = sim.particles[-1]
        r_xy = np.sqrt(p.x**2 + p.y**2)
        ra_true = np.arctan2(p.y, p.x)
        dec_true = np.arctan2(p.z, r_xy)
        
        ra_obs = ra_true + np.random.normal(0, sigma_astrom)
        dec_obs = dec_true + np.random.normal(0, sigma_astrom)
        astrometry.append([ra_obs, dec_obs])
        covariances.append(np.diag([sigma_astrom**2, sigma_astrom**2]))
        
    return np.array(astrometry), covariances


# =====================================================================
# 4. MAIN INJECTION-RECOVERY SWEEP EXECUTION
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("COMET-X: Automated Injection-Recovery Sensitivity Sweep")
    print("=" * 70)
    
    times = np.linspace(0, 365.25 * 2, 15) # 15 epochs over 2 years
    true_elements = [2.2, 0.10, 0.05, 1.2, 0.8, 0.0]
    
    # Logarithmic grid of A2 values to test detection limits
    injected_A2_grid = [1.0e-12, 1.0e-11, 1.0e-10, 5.0e-10]
    sweep_results = []
    
    for A2_inj in injected_A2_grid:
        print(f"\n>>> Running Sweep for Injected A2 = {A2_inj:.2e} <<<")
        astrometry, covariances = generate_synthetic_data(A2_inj, times, true_elements)
        
        # Fit Model A (Pure Gravity)
        print("  Fitting Model A (Pure Gravity)...")
        sampler_A = NestedSampler(
            lambda c: log_likelihood(c, times, astrometry, covariances, model_type="gravity"),
            lambda c: prior_transform(c, model_type="gravity"),
            ndim=6, nlive=50
        )
        sampler_A.run_nested(dlogz=2.0, print_progress=False)
        logZ_A = sampler_A.results.logz[-1]
        
        # Fit Model C (Gravity + Outgassing)
        print("  Fitting Model C (Gravity + Outgassing)...")
        sampler_C = NestedSampler(
            lambda c: log_likelihood(c, times, astrometry, covariances, model_type="outgassing"),
            lambda c: prior_transform(c, model_type="outgassing"),
            ndim=7, nlive=50
        )
        sampler_C.run_nested(dlogz=2.0, print_progress=False)
        logZ_C = sampler_C.results.logz[-1]
        
        delta_lnZ = logZ_C - logZ_A
        sweep_results.append((A2_inj, logZ_A, logZ_C, delta_lnZ))
        print(f"  Result -> ln(Z_A): {logZ_A:.2f} | ln(Z_C): {logZ_C:.2f} | Delta ln(Z): {delta_lnZ:.2f}")

    # Final Summary Table
    print("\n" + "=" * 70)
    print("INJECTION-RECOVERY SWEEP SUMMARY TABLE:")
    print(f"{'Injected A2':<15} | {'ln(Z_A)':<12} | {'ln(Z_C)':<12} | {'Delta ln(Z)':<12}")
    print("-" * 70)
    for A2_val, z_a, z_c, dz in sweep_results:
        print(f"{A2_val:<15.2e} | {z_a:<12.2f} | {z_c:<12.2f} | {dz:<12.2f}")
    print("=" * 70)
    
Verified pipeline # Instead of default initialization, update your dynesty sampler to:
sampler = dynesty.DynamicNestedSampler(
    loglikelihood, 
    prior_transform, 
    ndim=7, # 6 orbital elements + 1 outgassing/Yarkovsky parameter
    bound='multi', # Multi-ellipsoid bounds handle complex posteriors better
    sample='rwalk', # Use random walk instead of uniform sampling to fix the warning
    walks=25 # Increase walks to ensure the sampler doesn't get stuck
)