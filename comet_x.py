import numpy as np
import rebound
from dynesty import NestedSampler

# =====================================================================
# 1. ORBIT PROPAGATION & COORDINATE TRANSFORMATION
# =====================================================================

def propagate_orbit_rebound(params, times, model_type="outgassing"):
    """
    Propagates an NEO trajectory under Model A ('gravity') or Model C ('outgassing')
    using local numerical parameters to avoid live NASA Horizons queries.
    """
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
    """Converts Cartesian coordinates (x, y, z) to sky coordinates (RA, Dec)."""
    radec = []
    for pos in xyz_array:
        x, y, z = pos
        r_xy = np.sqrt(x**2 + y**2)
        ra = np.arctan2(y, x)
        dec = np.arctan2(z, r_xy)
        radec.append([ra, dec])
    return np.array(radec)


# =====================================================================
# 2. BAYESIAN LIKELIHOOD & PRIOR FUNCTIONS FOR DYNESTY
# =====================================================================

def prior_transform(cube, model_type="outgassing"):
    """Transforms unit hypercube [0, 1]^D to physical parameter priors."""
    params = np.zeros_like(cube)
    params[0] = cube[0] * 0.4 + 2.0   # Semi-major axis a [AU] (2.0 to 2.4)
    params[1] = cube[1] * 0.05 + 0.08 # Eccentricity e (0.08 to 0.13)
    params[2] = cube[2] * 0.02 + 0.04 # Inclination inc [rad] (0.04 to 0.06)
    params[3] = cube[3] * 0.2 + 1.1   # Omega (1.1 to 1.3)
    params[4] = cube[4] * 0.2 + 0.7   # omega (0.7 to 0.9)
    params[5] = cube[5] * 0.2 - 0.1   # M (-0.1 to 0.1)
    
    if model_type == "outgassing":
        log_min, log_max = -11.5, -10.5
        params[6] = 10.0 ** (cube[6] * (log_max - log_min) + log_min)
        
    return params

def log_likelihood(params, times, observed_astrometry, covariance_matrices, model_type="outgassing"):
    """Computes Gaussian log-likelihood using full astrometric covariance matrices."""
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
# 3. SYNTHETIC DATA GENERATOR & MAIN EXECUTION SCRIPT
# =====================================================================

def generate_synthetic_data(true_A2, times, nominal_elements):
    """Generates synthetic observations with an injected outgassing signal locally."""
    sim = rebound.Simulation()
    sim.units = ('days', 'AU', 'Msun')
    sim.add(m=1.0) # Sun
    sim.add(m=0.000954388, a=5.2044, e=0.0489, inc=0.0227) # Jupiter
    
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
    sigma_astrom = 0.5 / 206265.0 
    
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

if __name__ == "__main__":
    print("=" * 70)
    print("COMET-X: Bayesian Model Selection & Nested Sampling Execution")
    print("=" * 70)
    
    times = np.linspace(0, 365.25 * 2, 10) 
    true_elements = [2.2, 0.10, 0.05, 1.2, 0.8, 0.0]
    injected_A2 = 1.0e-11 
    
    print(f"Generating synthetic observations with injected A2 = {injected_A2:.2e}...")
    astrometry, covariances = generate_synthetic_data(injected_A2, times, true_elements)
    
    # --- FIXED DIAGNOSTIC TEST (Passes 7-element vector for Outgassing) ---
    true_elements_C = true_elements + [injected_A2]
    test_logl = log_likelihood(true_elements_C, times, astrometry, covariances, model_type="outgassing")
    print(f"\n[DIAGNOSTIC] Log-likelihood at true parameters: {test_logl:.2f}\n")
    if np.isinf(test_logl):
        print("[ERROR] Diagnostic failed: Likelihood is -inf at true parameters!")
    else:
        print("[SUCCESS] Diagnostic passed! Likelihood is finite at true parameters.\n")
    
    # --- Run Model A (Gravity-Only) ---
    print("--- Fitting Model A: Pure Gravity (6 Dimensions) ---")
    def logl_A(cube):
        return log_likelihood(cube, times, astrometry, covariances, model_type="gravity")
    def prior_A(cube):
        return prior_transform(cube, model_type="gravity")
        
    sampler_A = NestedSampler(logl_A, prior_A, ndim=6, nlive=50)
    sampler_A.run_nested(dlogz=1.0, print_progress=True)
    logZ_A = sampler_A.results.logz[-1]
    print(f"Model A Log-Evidence (ln Z_A): {logZ_A:.2f}")
    
    # --- Run Model C (Gravity + Outgassing) ---
    print("--- Fitting Model C: Gravity + Outgassing (7 Dimensions) ---")
    def logl_C(cube):
        return log_likelihood(cube, times, astrometry, covariances, model_type="outgassing")
    def prior_C(cube):
        return prior_transform(cube, model_type="outgassing")
        
    sampler_C = NestedSampler(logl_C, prior_C, ndim=7, nlive=50)
    sampler_C.run_nested(dlogz=1.0, print_progress=True)
    logZ_C = sampler_C.results.logz[-1]
    print(f"Model C Log-Evidence (ln Z_C): {logZ_C:.2f}")
    
    delta_lnZ = logZ_C - logZ_A
    print("\n" + "=" * 70)
    print(f"BAYES FACTOR RESULTS:")
    print(f"  Delta ln(Z) [ln(Z_C) - ln(Z_A)] = {delta_lnZ:.2f}")
    print("=" * 70)