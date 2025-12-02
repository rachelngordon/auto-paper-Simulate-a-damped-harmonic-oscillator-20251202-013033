# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def simulate(m, c, k, x0, v0, t):
    dt = t[1] - t[0]
    x = np.empty_like(t)
    v = np.empty_like(t)
    x[0] = x0
    v[0] = v0
    for i in range(1, len(t)):
        def f(state):
            x_, v_ = state
            dx = v_
            dv = -(c / m) * v_ - (k / m) * x_
            return np.array([dx, dv])
        state = np.array([x[i - 1], v[i - 1]])
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state_next = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x[i], v[i] = state_next
    return x, v

def main():
    m = 1.0
    k = 1.0
    x0 = 1.0
    v0 = 0.0
    t = np.arange(0, 20, 0.01)
    cs = {
        "underdamped": 0.2,
        "critical": 2.0,
        "overdamped": 3.0
    }
    results = {}
    for regime, c in cs.items():
        x, v = simulate(m, c, k, x0, v0, t)
        results[regime] = (x, v)
    # Time response plot
    plt.figure()
    for regime, (x, _) in results.items():
        plt.plot(t, x, label=regime)
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement')
    plt.title('Damped Harmonic Oscillator - Time Response')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('time_response_vs_damping.png')
    plt.close()
    # Phase space plot
    plt.figure()
    for regime, (x, v) in results.items():
        plt.plot(x, v, label=regime)
    plt.xlabel('Displacement')
    plt.ylabel('Velocity')
    plt.title('Phase Space Trajectories')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('phase_space.png')
    plt.close()
    # Primary numeric answer (critical damping ratio)
    answer = 1.0
    print('Answer:', answer)

if __name__ == '__main__':
    main()

